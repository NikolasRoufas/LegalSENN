# run_train.py
# Deterministic, resumable training runner used by the experimental pipeline.
#
# Default protocol:
#   AdamW (torch defaults, weight_decay=0.01), lr 2e-5, batch 16, no LR schedule
#   (as in the released train.py), max 10 epochs, early stopping on validation
#   Macro-F1 with patience 3, checkpoint with best validation Macro-F1 retained.
#   The test split is evaluated exactly once, after training, with the selected
#   checkpoint, and written to a separate file that selection scripts never read.

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from legalsenn.model import build_model
from legalsenn import scotus

NUM_LABELS = 13


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def check_gpu() -> None:
    """Refuse to run on any GPU other than the one named by LEGALSENN_GPU_BUS (PCI bus number)."""
    want = os.environ.get("LEGALSENN_GPU_BUS")
    if want is None:
        return
    p = torch.cuda.get_device_properties(0)
    if int(p.pci_bus_id) != int(want):
        raise SystemExit(f"GPU guard: device 0 is on PCI bus {p.pci_bus_id}, expected {want}")


def git_commit() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return os.environ.get("LEGALSENN_COMMIT", "unknown")


def metrics_from_logits(logits: np.ndarray, y: np.ndarray) -> dict:
    pred = logits.argmax(-1)
    labels = list(range(NUM_LABELS))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "micro_f1": float(f1_score(y, pred, labels=labels, average="micro", zero_division=0)),
        "per_class_f1": [float(x) for x in f],
        "per_class_precision": [float(x) for x in p],
        "per_class_recall": [float(x) for x in r],
        "per_class_support": [int(x) for x in s],
    }


@torch.no_grad()
def predict_logits(model, input_ids, attn, device, batch_size, amp):
    model.eval()
    out = []
    for i in range(0, len(input_ids), batch_size):
        ids = input_ids[i:i + batch_size].to(device, non_blocking=True)
        am = attn[i:i + batch_size].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            lg = model(input_ids=ids, attention_mask=am)["logits"]
        out.append(lg.float().cpu().numpy())
    return np.concatenate(out, 0)


def build_inputs(arch, ids_list, args, tok):
    if arch == "hier":
        return scotus.hier_inputs(ids_list, args.max_segments, args.segment_length,
                                  tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)
    return scotus.flat_inputs(ids_list, args.max_length, tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["senn", "concept", "bert", "hier"], required=True)
    ap.add_argument("--num_concepts", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--max_segments", type=int, default=64)
    ap.add_argument("--segment_length", type=int, default=128)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--micro_batch", type=int, default=16, help="gradient accumulation if < batch_size")
    ap.add_argument("--eval_batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--experiment", default="")
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.run_dir, "DONE")):
        print(f"[skip] {args.run_dir} already complete")
        return
    os.makedirs(args.run_dir, exist_ok=True)
    set_determinism(args.seed)
    device = torch.device("cuda")
    check_gpu()
    amp = not args.no_amp

    tok = scotus.get_tokenizer()
    data = scotus.tokenized_corpus(args.cache_dir, tok)
    X = {s: build_inputs(args.arch, data[s]["ids"], args, tok) for s in scotus.SPLITS}
    Y = {s: torch.from_numpy(data[s]["label"]) for s in scotus.SPLITS}

    model = build_model(args.arch, scotus.ENCODER_NAME, NUM_LABELS, num_concepts=args.num_concepts,
                        max_segments=args.max_segments).to(device)
    if args.grad_ckpt:
        model.encoder.gradient_checkpointing_enable()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_head = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("encoder."))

    cfg = {
        "experiment": args.experiment,
        "arch": args.arch,
        "encoder_name": scotus.ENCODER_NAME,
        "encoder_revision": scotus.ENCODER_REVISION,
        "dataset": f"{scotus.DATASET_NAME}/{scotus.DATASET_CONFIG}",
        "dataset_revision": scotus.DATASET_REVISION,
        "split_fingerprints": {s: data[s]["fingerprint"] for s in scotus.SPLITS},
        "num_labels": NUM_LABELS,
        "num_concepts": args.num_concepts if args.arch in ("senn", "concept") else None,
        "max_length": args.max_length if args.arch != "hier" else args.max_segments * args.segment_length,
        "max_segments": args.max_segments if args.arch == "hier" else None,
        "segment_length": args.segment_length if args.arch == "hier" else None,
        "seed": args.seed,
        "epochs_max": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "micro_batch": args.micro_batch,
        "lr": args.lr,
        "optimizer": "AdamW(torch default betas=(0.9,0.999), eps=1e-8, weight_decay=0.01)",
        "lr_schedule": "constant",
        "dropout": 0.1,
        "mixed_precision": "bf16 autocast" if amp else "fp32",
        "selection_metric": "validation macro_f1",
        "tokenizer": f"{scotus.ENCODER_NAME} (fast WordPiece, uncased)",
        "trainable_params": n_trainable,
        "head_params": n_head,
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "host": platform.node(),
    }
    import transformers
    cfg["transformers"] = transformers.__version__
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(args.seed)
    n_train = len(Y["train"])
    accum = max(1, args.batch_size // args.micro_batch)

    best_f1, best_epoch, bad = -1.0, 0, 0
    t_start = time.time()
    metrics_path = os.path.join(args.run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n_train, generator=g)
        tot_loss, tot_n = 0.0, 0
        for b in range(0, n_train, args.batch_size):
            idx = perm[b:b + args.batch_size]
            opt.zero_grad(set_to_none=True)
            for m in range(0, len(idx), args.micro_batch):
                mi = idx[m:m + args.micro_batch]
                ids = X["train"][0][mi].to(device, non_blocking=True)
                am = X["train"][1][mi].to(device, non_blocking=True)
                y = Y["train"][mi].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    logits = model(input_ids=ids, attention_mask=am)["logits"]
                loss = crit(logits.float(), y)
                (loss * len(mi) / len(idx)).backward()
                tot_loss += loss.item() * len(mi)
                tot_n += len(mi)
            opt.step()
        train_time = time.time() - t0

        val_logits = predict_logits(model, *X["validation"], device, args.eval_batch, amp)
        vm = metrics_from_logits(val_logits, Y["validation"].numpy())
        rec = {"epoch": epoch, "train_loss": tot_loss / tot_n, "train_time_s": train_time,
               "val_accuracy": vm["accuracy"], "val_macro_f1": vm["macro_f1"],
               "val_macro_precision": vm["macro_precision"], "val_macro_recall": vm["macro_recall"]}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

        if vm["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = vm["macro_f1"], epoch, 0
            torch.save(model.state_dict(), os.path.join(args.run_dir, "best_model.pt"))
            np.savez_compressed(os.path.join(args.run_dir, "val_logits.npz"),
                                logits=val_logits, labels=Y["validation"].numpy())
            best_val = vm
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[early stop] epoch {epoch}, best epoch {best_epoch}")
                break

    train_runtime = time.time() - t_start

    # Validation summary is written (and therefore frozen) before the test split is touched.
    with open(os.path.join(args.run_dir, "val_summary.json"), "w") as f:
        json.dump({"selected_epoch": best_epoch, "epochs_run": epoch, "train_runtime_s": train_runtime,
                   **{f"val_{k}": v for k, v in best_val.items()}}, f, indent=2)

    state = torch.load(os.path.join(args.run_dir, "best_model.pt"), map_location="cpu")
    model.load_state_dict(state)
    test_logits = predict_logits(model, *X["test"], device, args.eval_batch, amp)
    tm = metrics_from_logits(test_logits, Y["test"].numpy())
    np.savez_compressed(os.path.join(args.run_dir, "test_logits.npz"), logits=test_logits, labels=Y["test"].numpy())
    with open(os.path.join(args.run_dir, "test_summary.json"), "w") as f:
        json.dump({f"test_{k}": v for k, v in tm.items()}, f, indent=2)

    with open(os.path.join(args.run_dir, "DONE"), "w") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    print(f"[done] best epoch {best_epoch} val_f1={best_f1:.6f} test_f1={tm['macro_f1']:.6f} "
          f"runtime={train_runtime:.0f}s")


if __name__ == "__main__":
    main()
