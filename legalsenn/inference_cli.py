# inference_cli.py
# Single-document inference with faithful SENN self-explanation (best label only)

import argparse
import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer

from legalsenn.model import LegalSENN
from legalsenn.data import load_label_map
from legalsenn.utils_reporting import load_id2name_json, softmax_np


def read_first_document(path: str) -> str:
    """Reads only the FIRST document from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    # If multiple docs exist, take only the first
    return raw.split("\n\n")[0]


def normalize(vals: np.ndarray) -> np.ndarray:
    vals = np.abs(vals)
    s = vals.sum()
    if s == 0:
        return np.zeros_like(vals)
    return vals / s


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True,
                        help="outputs/<RUN_ID> containing best_model.pt + label_map.json")
    parser.add_argument("--text_file", type=str, required=True,
                        help="Text file (only FIRST document will be used)")
    parser.add_argument("--topk_concepts", type=int, default=6)

    parser.add_argument("--encoder_name", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--label_semantics_path", type=str, default="label_semantics.json")
    parser.add_argument("--concept_names_path", type=str, default="concept_names.json")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    ckpt_path = os.path.join(args.model_dir, "best_model.pt")
    label_map_path = os.path.join(args.model_dir, "label_map.json")
    config_path = os.path.join(args.model_dir, "config.json")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"Missing label_map.json in {args.model_dir}")

    # Encoder name
    encoder_name = args.encoder_name
    if encoder_name is None and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        encoder_name = cfg.get("encoder_name", None)
    if encoder_name is None:
        raise ValueError("encoder_name not provided and not found in config.json")

    # Load mappings
    label2id = load_label_map(label_map_path)
    num_labels = len(label2id)

    label_names = load_id2name_json(args.label_semantics_path) \
        if os.path.exists(args.label_semantics_path) else None

    concept_names = load_id2name_json(args.concept_names_path) \
        if os.path.exists(args.concept_names_path) else None

    num_concepts = len(concept_names) if concept_names is not None else 16

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)

    # Model
    model = LegalSENN(
        encoder_name=encoder_name,
        num_labels=num_labels,
        num_concepts=num_concepts,
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ---- Read ONE document ----
    text = read_first_document(args.text_file)

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = out["logits"].detach().cpu().numpy()
    probs = softmax_np(logits, axis=-1).squeeze(0)

    pred_id = int(probs.argmax())
    pred_prob = float(probs[pred_id])

    def label_name(i: int) -> str:
        if label_names is None:
            return str(i)
        return label_names.get(i, str(i))

    print("\n=== FINAL DECISION ===")
    print(f"Label      : {label_name(pred_id)}")
    print(f"Confidence : {pred_prob:.4f}")

    # ---- Faithful SENN self-explanation ----
    if "contributions" not in out or concept_names is None:
        print("\n(No self-explanations available.)")
        return

    c = out["contributions"].detach().cpu().numpy().squeeze(0)  # (C,)
    W = model.classifier.weight.detach().cpu().numpy()          # (K, C)
    Wy = W[pred_id]                                             # (C,)

    # Exact logit decomposition
    effects = Wy * c

    support_idx = np.argsort(-effects)
    oppose_idx = np.argsort(effects)

    k = min(args.topk_concepts, len(effects))

    sup_ids = support_idx[:k]
    opp_ids = oppose_idx[:k]

    sup_imp = normalize(effects[sup_ids])
    opp_imp = normalize(effects[opp_ids])

    print("\n=== SELF-EXPLANATION (faithful, best label) ===")

    print("\nTop concepts SUPPORTING this decision:")
    for i, imp in zip(sup_ids, sup_imp):
        print(
            f"  {concept_names[int(i)]:45s} "
            f"effect={effects[int(i)]:+8.4f} "
            f"importance={imp:6.3f}"
        )

    print("\nTop concepts OPPOSING this decision:")
    for i, imp in zip(opp_ids, opp_imp):
        print(
            f"  {concept_names[int(i)]:45s} "
            f"effect={effects[int(i)]:+8.4f} "
            f"importance={imp:6.3f}"
        )


if __name__ == "__main__":
    main()
