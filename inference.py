# inference.py

import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from model import LegalSENN
from data import load_label_map


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Inference with LegalSENN")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model dir")
    parser.add_argument("--text_file", type=str, help="Path to a text file with the trial")
    parser.add_argument("--text", type=str, help="Raw text passed directly as argument")
    parser.add_argument("--top_k", type=int, default=3, help="Show top-k labels")

    args = parser.parse_args()

    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        raise ValueError("Provide either --text_file or --text")

    # Load config and label map
    config_path = os.path.join(args.model_dir, "config.json")
    label_map_path = os.path.join(args.model_dir, "label_map.json")
    model_path = os.path.join(args.model_dir, "best_model.pt")

    with open(config_path, "r") as f:
        config = json.load(f)

    label2id = load_label_map(label_map_path)
    id2label = {v: k for k, v in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(config["encoder_name"], use_fast=True)

    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=config["max_length"],
        return_tensors="pt",
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    device = get_device()
    print(f"Using device: {device}")
    if device.type == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    model = LegalSENN(
        encoder_name=config["encoder_name"],
        num_labels=config["num_labels"],
        num_concepts=config["num_concepts"],
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    out = model.predict(input_ids=input_ids, attention_mask=attention_mask)

    probs = out["probs"][0].cpu()
    preds = out["preds"][0].item()
    contributions = out["contributions"][0].cpu().tolist()

    # Top-k labels
    topk = min(args.top_k, probs.size(0))
    topk_probs, topk_indices = torch.topk(probs, k=topk)

    print("\n=== Top-k predicted labels ===")
    for p, idx in zip(topk_probs, topk_indices):
        label = id2label[idx.item()]
        print(f"{label:25s}  prob={p.item():.4f}")

    # Concept-level explanation
    print("\n=== Concept contributions (SENN head) ===")
    contrib_abs = [(i, abs(c)) for i, c in enumerate(contributions)]
    contrib_abs.sort(key=lambda x: x[1], reverse=True)

    for i, _ in contrib_abs[:10]:
        print(f"Concept {i:02d}: contribution={contributions[i]:.4f}")


if __name__ == "__main__":
    main()
