# explain_eval.py
# (1) Concept Deletion Test (causal)
# (2) Explanation Stability under sentence dropout (robustness)

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from legalsenn.model import LegalSENN
from legalsenn.data import LegalTextDataset, load_label_map
from legalsenn.utils_reporting import load_id2name_json, softmax_np


# -----------------------------
# Helpers
# -----------------------------

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def simple_sentence_split(text: str) -> List[str]:
    """
    Lightweight sentence splitter that works decently on legal text.
    Avoids external deps. Splits on . ? ! followed by whitespace.
    """
    text = text.strip()
    if not text:
        return []
    # Keep delimiters by splitting on punctuation boundaries
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def drop_sentences(text: str, drop_rate: float, seed: int) -> str:
    """
    Drop a fraction of sentences randomly. Always keeps at least 1 sentence.
    """
    sents = simple_sentence_split(text)
    if len(sents) <= 1 or drop_rate <= 0:
        return text
    rng = np.random.RandomState(seed)
    n = len(sents)
    keep = np.ones(n, dtype=bool)
    k_drop = int(round(drop_rate * n))
    k_drop = min(k_drop, n - 1)  # keep >= 1
    if k_drop > 0:
        drop_idx = rng.choice(n, size=k_drop, replace=False)
        keep[drop_idx] = False
    kept = [sents[i] for i in range(n) if keep[i]]
    return " ".join(kept)


def jaccard(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


# -----------------------------
# Core inference ops
# -----------------------------

@torch.no_grad()
def forward_with_override_concepts(model: LegalSENN,
                                   input_ids: torch.Tensor,
                                   attention_mask: torch.Tensor,
                                   concept_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs model forward, obtains:
    - probs: (K,)
    - contributions: (C,)

    If concept_mask is provided (C,), concepts with mask=0 are zeroed
    BEFORE classifier to measure causal impact of removing concepts.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out["logits"]  # (1, K)
    contrib = out.get("contributions", None)  # (1, C)

    if contrib is None:
        raise RuntimeError("Model output does not contain 'contributions'. Cannot run explainability experiments.")

    contrib = contrib.clone()  # (1, C)

    if concept_mask is not None:
        mask = torch.tensor(concept_mask, dtype=contrib.dtype, device=contrib.device).view(1, -1)
        contrib = contrib * mask
        # IMPORTANT: we must recompute logits from modified contrib.
        # Your model uses a linear head: logits = classifier(contributions)
        logits = model.classifier(contrib)

    probs = softmax_np(logits.detach().cpu().numpy(), axis=-1).squeeze(0)  # (K,)
    contrib_np = contrib.detach().cpu().numpy().squeeze(0)  # (C,)
    return probs, contrib_np


def topk_concepts_for_pred(model: LegalSENN,
                           pred_id: int,
                           contrib: np.ndarray,
                           topk: int) -> List[int]:
    """
    Faithful, class-specific concept ranking: |w_y * c|
    """
    W = model.classifier.weight.detach().cpu().numpy()  # (K, C)
    Wy = W[pred_id]  # (C,)
    effects = Wy * contrib
    order = np.argsort(-np.abs(effects))
    return [int(i) for i in order[:topk]]


# -----------------------------
# Experiment 1: Deletion test
# -----------------------------

@torch.no_grad()
def concept_deletion_curves(model: LegalSENN,
                            dataset: LegalTextDataset,
                            device: torch.device,
                            max_points: int,
                            delete_steps: int,
                            topk_rank: int,
                            seed: int = 42) -> Dict:
    """
    For each sample:
    - get predicted class
    - rank concepts by |w_y*c|
    - progressively delete top concepts
    - measure confidence drop
    Baselines:
    - random deletion
    - lowest-importance deletion
    """
    rng = np.random.RandomState(seed)

    # choose subset indices
    N = len(dataset)
    idxs = np.arange(N)
    rng.shuffle(idxs)
    idxs = idxs[: min(max_points, N)]

    # curves: list of (step -> confidence) aggregated
    # step t corresponds to deleting t/delete_steps fraction of concepts
    all_top = []
    all_rand = []
    all_low = []

    C = None

    for idx in tqdm(idxs, desc="DeletionTest", leave=False):
        item = dataset[idx]
        input_ids = item[0].unsqueeze(0).to(device)
        attention_mask = item[1].unsqueeze(0).to(device)

        # base forward
        probs, contrib = forward_with_override_concepts(model, input_ids, attention_mask, concept_mask=None)
        pred = int(probs.argmax())
        base_conf = float(probs[pred])

        if C is None:
            C = int(contrib.shape[0])

        # rank concepts (class-specific)
        rank = topk_concepts_for_pred(model, pred, contrib, topk=C)  # full order

        # low importance order
        low_rank = list(reversed(rank))

        # random order
        rand_rank = list(rank)
        rng.shuffle(rand_rank)

        # steps
        step_confs_top = []
        step_confs_low = []
        step_confs_rand = []

        for s in range(delete_steps + 1):
            # fraction deleted
            k_del = int(round((s / delete_steps) * C))
            k_del = min(k_del, C)

            # build masks
            mask_top = np.ones(C, dtype=np.float32)
            mask_low = np.ones(C, dtype=np.float32)
            mask_rand = np.ones(C, dtype=np.float32)

            for j in range(k_del):
                mask_top[rank[j]] = 0.0
                mask_low[low_rank[j]] = 0.0
                mask_rand[rand_rank[j]] = 0.0

            p_top, _ = forward_with_override_concepts(model, input_ids, attention_mask, concept_mask=mask_top)
            p_low, _ = forward_with_override_concepts(model, input_ids, attention_mask, concept_mask=mask_low)
            p_rand, _ = forward_with_override_concepts(model, input_ids, attention_mask, concept_mask=mask_rand)

            step_confs_top.append(float(p_top[pred]))
            step_confs_low.append(float(p_low[pred]))
            step_confs_rand.append(float(p_rand[pred]))

        all_top.append(step_confs_top)
        all_low.append(step_confs_low)
        all_rand.append(step_confs_rand)

    all_top = np.asarray(all_top)
    all_low = np.asarray(all_low)
    all_rand = np.asarray(all_rand)

    out = {
        "num_samples": int(len(all_top)),
        "num_concepts": int(C if C is not None else 0),
        "delete_steps": int(delete_steps),
        "x_frac_deleted": [float(s / delete_steps) for s in range(delete_steps + 1)],
        "mean_conf_top": all_top.mean(axis=0).tolist(),
        "mean_conf_low": all_low.mean(axis=0).tolist(),
        "mean_conf_rand": all_rand.mean(axis=0).tolist(),
        "std_conf_top": all_top.std(axis=0).tolist(),
        "std_conf_low": all_low.std(axis=0).tolist(),
        "std_conf_rand": all_rand.std(axis=0).tolist(),
    }
    return out


# -----------------------------
# Experiment 2: Stability test
# -----------------------------

@torch.no_grad()
def explanation_stability(model: LegalSENN,
                          dataset: LegalTextDataset,
                          raw_texts: List[str],
                          tokenizer_name: str,
                          device: torch.device,
                          max_points: int,
                          drop_rate: float,
                          repeats: int,
                          topk: int,
                          seed: int = 123) -> Dict:
    """
    For each sample:
    - compute top-k concept indices for predicted label (original)
    - perturb input by dropping sentences
    - compute top-k concept indices again
    - measure Jaccard overlap
    Also track label flip rate under perturbation.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    rng = np.random.RandomState(seed)
    N = len(dataset)
    idxs = np.arange(N)
    rng.shuffle(idxs)
    idxs = idxs[: min(max_points, N)]

    overlaps = []
    label_flips = 0
    total = 0

    for idx in tqdm(idxs, desc="StabilityTest", leave=False):
        item = dataset[idx]
        input_ids = item[0].unsqueeze(0).to(device)
        attention_mask = item[1].unsqueeze(0).to(device)

        probs, contrib = forward_with_override_concepts(model, input_ids, attention_mask, concept_mask=None)
        pred = int(probs.argmax())

        orig_top = topk_concepts_for_pred(model, pred, contrib, topk=topk)

        text = raw_texts[idx]
        for r in range(repeats):
            pert = drop_sentences(text, drop_rate=drop_rate, seed=seed + idx * 1000 + r)
            enc = tok(
                pert,
                truncation=True,
                padding="max_length",
                max_length=dataset.max_length if hasattr(dataset, "max_length") else 256,
                return_tensors="pt",
            )
            pid = enc["input_ids"].to(device)
            pam = enc["attention_mask"].to(device)

            p2, c2 = forward_with_override_concepts(model, pid, pam, concept_mask=None)
            pred2 = int(p2.argmax())
            if pred2 != pred:
                label_flips += 1

            top2 = topk_concepts_for_pred(model, pred, c2, topk=topk)  # keep SAME class for stability
            overlaps.append(jaccard(orig_top, top2))
            total += 1

    out = {
        "num_base_samples": int(len(idxs)),
        "repeats_per_sample": int(repeats),
        "drop_rate": float(drop_rate),
        "topk": int(topk),
        "mean_jaccard": float(np.mean(overlaps)) if overlaps else 0.0,
        "std_jaccard": float(np.std(overlaps)) if overlaps else 0.0,
        "label_flip_rate": float(label_flips / max(1, total)),
    }
    return out


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True, help="outputs/<RUN_ID>")
    parser.add_argument("--split_csv", type=str, required=True, help="CSV to evaluate (e.g., test.csv)")
    parser.add_argument("--encoder_name", type=str, required=True)
    parser.add_argument("--label_map", type=str, required=True, help="outputs/<RUN_ID>/label_map.json")

    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)

    # deletion test
    parser.add_argument("--run_deletion", action="store_true")
    parser.add_argument("--delete_steps", type=int, default=10)
    parser.add_argument("--max_points", type=int, default=400)

    # stability test
    parser.add_argument("--run_stability", action="store_true")
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--topk", type=int, default=6)

    parser.add_argument("--out_dir", type=str, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = os.path.join(args.model_dir, "best_model.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    out_dir = args.out_dir or os.path.join(args.model_dir, "explain_eval")
    safe_mkdir(out_dir)

    # mappings
    label2id = load_label_map(args.label_map)
    num_labels = len(label2id)

    # dataset
    dataset = LegalTextDataset(
        csv_path=args.split_csv,
        tokenizer_name=args.encoder_name,
        label2id=label2id,
        max_length=args.max_length,
        text_col=args.text_col,
        label_col=args.label_col,
    )

    # Load raw texts for stability test
    # We'll read the CSV again safely using pandas (dataset already uses pandas internally)
    import pandas as pd
    df = pd.read_csv(args.split_csv)
    raw_texts = df[args.text_col].astype(str).tolist()

    # model
    # infer num_concepts from concept_names.json if exists in repo root
    concept_names_path = "concept_names.json"
    if os.path.exists(concept_names_path):
        concept_names = load_id2name_json(concept_names_path)
        num_concepts = len(concept_names)
    else:
        num_concepts = 16

    model = LegalSENN(
        encoder_name=args.encoder_name,
        num_labels=num_labels,
        num_concepts=num_concepts,
    ).to(device)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    results = {}

    if args.run_deletion:
        deletion = concept_deletion_curves(
            model=model,
            dataset=dataset,
            device=device,
            max_points=args.max_points,
            delete_steps=args.delete_steps,
            topk_rank=num_concepts,
        )
        results["deletion_test"] = deletion
        with open(os.path.join(out_dir, "deletion_test.json"), "w") as f:
            json.dump(deletion, f, indent=2)

        # Also save a CSV for easy plotting later
        csv_path = os.path.join(out_dir, "deletion_test.csv")
        with open(csv_path, "w") as f:
            f.write("frac_deleted,mean_top,mean_rand,mean_low,std_top,std_rand,std_low\n")
            for i, x in enumerate(deletion["x_frac_deleted"]):
                f.write(
                    f"{x},"
                    f"{deletion['mean_conf_top'][i]},"
                    f"{deletion['mean_conf_rand'][i]},"
                    f"{deletion['mean_conf_low'][i]},"
                    f"{deletion['std_conf_top'][i]},"
                    f"{deletion['std_conf_rand'][i]},"
                    f"{deletion['std_conf_low'][i]}\n"
                )

    if args.run_stability:
        stability = explanation_stability(
            model=model,
            dataset=dataset,
            raw_texts=raw_texts,
            tokenizer_name=args.encoder_name,
            device=device,
            max_points=args.max_points,
            drop_rate=args.drop_rate,
            repeats=args.repeats,
            topk=args.topk,
        )
        results["stability_test"] = stability
        with open(os.path.join(out_dir, "stability_test.json"), "w") as f:
            json.dump(stability, f, indent=2)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved explainability eval artifacts to: {out_dir}")
    if args.run_deletion:
        print(" - deletion_test.json / deletion_test.csv")
    if args.run_stability:
        print(" - stability_test.json")
    print(" - summary.json")


if __name__ == "__main__":
    main()
