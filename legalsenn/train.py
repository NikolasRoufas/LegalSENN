# train.py
# Spaces-only indentation (4 spaces). Server-friendly logging + artifacts.

import argparse
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from legalsenn.model import LegalSENN
from legalsenn.data import LegalTextDataset, build_label_map, save_label_map, load_label_map


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Keep MPS fallback in case you ever run locally on Mac again
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return torch.device("mps")
    return torch.device("cpu")


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=axis, keepdims=True)


def load_id2name_json(path: str) -> Dict[int, str]:
    """Loads {"0":"name",...} -> {0:"name",...}"""
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def save_jsonl(rows: List[dict], path: str) -> None:
    safe_mkdir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def confusion_matrix(y_true: List[int], y_pred: List[int], n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def save_confusion_csv(cm: np.ndarray, id2label: Dict[int, str], path: str) -> None:
    safe_mkdir(os.path.dirname(path) or ".")
    labels = [id2label[i] for i in range(len(id2label))]
    with open(path, "w") as f:
        f.write("true\\pred," + ",".join(labels) + "\n")
        for i in range(cm.shape[0]):
            row = [labels[i]] + [str(int(x)) for x in cm[i].tolist()]
            f.write(",".join(row) + "\n")


def classification_report_simple(
    y_true: List[int], y_pred: List[int], id2label: Dict[int, str]
) -> dict:
    """
    Lightweight per-class precision/recall/f1 + macro/weighted averages.
    (No sklearn dependency)
    """
    n = len(id2label)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    report = {"per_class": {}}
    f1s = []
    weights = []

    for c in range(n):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        support = int((y_true == c).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        report["per_class"][id2label[c]] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(support),
        }
        f1s.append(f1)
        weights.append(support)

    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    weighted_f1 = float(np.average(f1s, weights=weights)) if sum(weights) > 0 else 0.0

    report["accuracy"] = acc
    report["macro_f1"] = macro_f1
    report["weighted_f1"] = weighted_f1
    report["num_samples"] = int(len(y_true))
    return report


def summarize_concepts_by_label(
    y_true: List[int],
    contributions: np.ndarray,  # (N, C)
    id2label: Dict[int, str],
    id2concept: Dict[int, str],
    top_k: int = 8,
) -> dict:
    """
    For each gold label: average contribution vector, return top concepts by |avg|.
    """
    y_true = np.asarray(y_true, dtype=int)
    out = {}

    for label_id in range(len(id2label)):
        idx = np.where(y_true == label_id)[0]
        if len(idx) == 0:
            continue
        avg = contributions[idx].mean(axis=0)  # (C,)
        order = np.argsort(-np.abs(avg))[: min(top_k, avg.shape[0])]
        out[id2label[label_id]] = [
            {"concept": id2concept[int(i)], "avg_contribution": float(avg[int(i)])}
            for i in order
        ]
    return out


def build_display_label_names(
    label2id: Dict[str, int],
    label_semantics_path: Optional[str] = None
) -> Dict[int, str]:
    """
    Build id2label for reporting.
    If label_semantics.json exists AND your CSV labels are numeric strings ("0","1",...),
    it will map to human-readable legal labels.
    Otherwise, it falls back to the original label string keys.
    """
    id2label_fallback = {int(v): str(k) for k, v in label2id.items()}

    if not label_semantics_path:
        return id2label_fallback
    if not os.path.exists(label_semantics_path):
        return id2label_fallback

    sem = load_id2name_json(label_semantics_path)

    # If your label strings are numeric, map through semantics.
    # Example: label2id keys: "0","1",...
    all_numeric = True
    for k in label2id.keys():
        if not str(k).isdigit():
            all_numeric = False
            break

    if all_numeric:
        # label ids are 0..K-1 already; semantics keys should match those ids
        id2label = {}
        for lbl_str, idx in label2id.items():
            i = int(idx)
            sem_key = int(lbl_str)  # "0" -> 0
            id2label[i] = sem.get(sem_key, id2label_fallback[i])
        return id2label

    # Otherwise: fallback (your labels are already names)
    return id2label_fallback


# -------------------------
# Train/Eval
# -------------------------

def train_epoch(model, dataloader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for input_ids, attention_mask, labels in tqdm(dataloader, desc="Train", leave=False):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"]

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return float(avg_loss), float(avg_acc)


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    y_true: List[int] = []
    y_pred: List[int] = []
    logits_list: List[np.ndarray] = []
    contrib_list: List[np.ndarray] = []

    for input_ids, attention_mask, labels in tqdm(dataloader, desc="Eval", leave=False):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"]
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        preds = logits.argmax(dim=-1)

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        logits_list.append(logits.detach().cpu().numpy())

        contrib = out.get("contributions", None)
        if contrib is not None:
            contrib_list.append(contrib.detach().cpu().numpy())

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if total_samples else 0.0

    logits_np = np.concatenate(logits_list, axis=0) if logits_list else None
    contrib_np = np.concatenate(contrib_list, axis=0) if contrib_list else None

    return float(avg_loss), float(avg_acc), y_true, y_pred, logits_np, contrib_np


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--encoder_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")

    # Accept both --epochs and --num_epochs (so your command works)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)

    parser.add_argument("--num_concepts", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--text_col", type=str, default="text")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    # Optional naming files
    parser.add_argument("--label_semantics_path", type=str, default=None)
    parser.add_argument("--concept_names_path", type=str, default=None)

    parser.add_argument("--save_eval_artifacts", action="store_true")
    parser.add_argument("--run_id", type=str, default=None, help="Optional run folder name")

    args = parser.parse_args()

    # Resolve epochs
    if args.epochs is not None:
        num_epochs = int(args.epochs)
    else:
        num_epochs = int(args.num_epochs)

    set_seed(args.seed)

    # Create run folder
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, run_id)
    safe_mkdir(out_dir)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Run dir: {out_dir}")

    # Build or load label map (from training CSV)
    label_map_path = os.path.join(out_dir, "label_map.json")
    if os.path.exists(label_map_path):
        label2id = load_label_map(label_map_path)
    else:
        label2id = build_label_map(args.train_csv, label_col=args.label_col)
        save_label_map(label2id, label_map_path)

    num_labels = len(label2id)

    # Build display names for reporting
    id2label = build_display_label_names(label2id, args.label_semantics_path)

    # Concept names
    id2concept = None
    if args.concept_names_path and os.path.exists(args.concept_names_path):
        id2concept = load_id2name_json(args.concept_names_path)
    else:
        # fallback generic names
        id2concept = {i: f"Concept {i}" for i in range(args.num_concepts)}

    # Datasets
    train_dataset = LegalTextDataset(
        csv_path=args.train_csv,
        tokenizer_name=args.encoder_name,
        label2id=label2id,
        max_length=args.max_length,
        text_col=args.text_col,
        label_col=args.label_col,
    )
    val_dataset = LegalTextDataset(
        csv_path=args.val_csv,
        tokenizer_name=args.encoder_name,
        label2id=label2id,
        max_length=args.max_length,
        text_col=args.text_col,
        label_col=args.label_col,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = LegalSENN(
        encoder_name=args.encoder_name,
        num_labels=num_labels,
        num_concepts=args.num_concepts,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Save config
    config = {
        "encoder_name": args.encoder_name,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "num_labels": num_labels,
        "num_concepts": args.num_concepts,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "label_col": args.label_col,
        "text_col": args.text_col,
        "seed": args.seed,
        "epochs": num_epochs,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    def log_metrics(d: dict) -> None:
        with open(metrics_path, "a") as f:
            f.write(json.dumps(d) + "\n")

    best_val_score = -1.0  # we'll track macro_f1

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

        val_loss, val_acc, y_true, y_pred, logits_np, contrib_np = eval_epoch(
            model, val_loader, criterion, device
        )
        report = classification_report_simple(y_true, y_pred, id2label)
        val_macro_f1 = report["macro_f1"]
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f} | Val macroF1: {val_macro_f1:.4f}")

        log_metrics({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": report["weighted_f1"],
        })

        # Save artifacts
        if args.save_eval_artifacts:
            epoch_dir = os.path.join(out_dir, f"epoch_{epoch}")
            safe_mkdir(epoch_dir)

            with open(os.path.join(epoch_dir, "classification_report.json"), "w") as f:
                json.dump(report, f, indent=2)

            cm = confusion_matrix(y_true, y_pred, n=num_labels)
            save_confusion_csv(cm, id2label, os.path.join(epoch_dir, "confusion_matrix.csv"))

            # Save per-example predictions with top-3 labels
            rows = []
            if logits_np is not None:
                probs = softmax_np(logits_np, axis=-1)
                top3 = np.argsort(-probs, axis=1)[:, :3]
                for i in range(len(y_true)):
                    rows.append({
                        "idx": int(i),
                        "gold_id": int(y_true[i]),
                        "gold_label": id2label[int(y_true[i])],
                        "pred_id": int(y_pred[i]),
                        "pred_label": id2label[int(y_pred[i])],
                        "confidence": float(probs[i, int(y_pred[i])]),
                        "top3": [
                            {"label": id2label[int(j)], "prob": float(probs[i, int(j)])}
                            for j in top3[i]
                        ],
                    })
            save_jsonl(rows, os.path.join(epoch_dir, "predictions.jsonl"))

            # Explanation summaries if contributions exist
            if contrib_np is not None:
                label2concepts = summarize_concepts_by_label(
                    y_true=y_true,
                    contributions=contrib_np,
                    id2label=id2label,
                    id2concept=id2concept,
                    top_k=8,
                )
                with open(os.path.join(epoch_dir, "label_top_concepts.json"), "w") as f:
                    json.dump(label2concepts, f, indent=2)

                tau = 0.01
                avg_active = float((np.abs(contrib_np) > tau).sum(axis=1).mean())
                expl_stats = {"avg_active_concepts_tau_0p01": avg_active}
                with open(os.path.join(epoch_dir, "explanation_stats.json"), "w") as f:
                    json.dump(expl_stats, f, indent=2)

        # Save best checkpoint by macro F1 (better for imbalance)
        if val_macro_f1 > best_val_score:
            best_val_score = val_macro_f1
            ckpt_path = os.path.join(out_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved new best model to {ckpt_path} (macro_f1={best_val_score:.4f})")


if __name__ == "__main__":
    main()
