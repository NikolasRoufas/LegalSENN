# inference.py
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from legalsenn.model import LegalSENN
from legalsenn.data import LegalTextDataset, load_label_map
from legalsenn.utils_reporting import load_id2name_json, save_jsonl, softmax_np


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")

    parser.add_argument("--encoder_name", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)

    # IMPORTANT: your dataset requires this
    parser.add_argument("--label_map", type=str, required=True, help="Path to outputs/<RUN_ID>/label_map.json")

    parser.add_argument("--label_semantics_path", type=str, default="label_semantics.json")
    parser.add_argument("--concept_names_path", type=str, default="concept_names.json")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--out_jsonl", type=str, default="outputs/inference_predictions.jsonl")
    parser.add_argument("--topk_concepts", type=int, default=8)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load maps
    label2id = load_label_map(args.label_map)
    num_labels = len(label2id)

    label_names = load_id2name_json(args.label_semantics_path) if os.path.exists(args.label_semantics_path) else None
    concept_names = load_id2name_json(args.concept_names_path) if os.path.exists(args.concept_names_path) else None
    num_concepts = len(concept_names) if concept_names is not None else 16

    # Dataset (now with label2id!)
    dataset = LegalTextDataset(
        csv_path=args.csv,
        tokenizer_name=args.encoder_name,
        label2id=label2id,
        max_length=args.max_length,
        text_col=args.text_col,
        label_col=args.label_col,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = LegalSENN(
        encoder_name=args.encoder_name,
        num_labels=num_labels,
        num_concepts=num_concepts,
    ).to(device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    def label_name(i: int) -> str:
        if label_names is None:
            return str(i)
        return label_names.get(i, str(i))

    rows = []
    global_idx = 0

    for input_ids, attention_mask, labels in tqdm(loader, desc="Infer"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"].detach().cpu().numpy()
        probs = softmax_np(logits, axis=-1)

        pred_ids = probs.argmax(axis=-1)
        conf = probs[np.arange(len(pred_ids)), pred_ids]

        contrib = out.get("contributions", None)
        contrib_np = contrib.detach().cpu().numpy() if contrib is not None else None

        for b in range(len(pred_ids)):
            gold_id = int(labels[b].item())
            pred_id = int(pred_ids[b])

            row = {
                "idx": global_idx,
                "gold_id": gold_id,
                "gold_label": label_name(gold_id),
                "pred_id": pred_id,
                "pred_label": label_name(pred_id),
                "confidence": float(conf[b]),
            }

            top3 = np.argsort(-probs[b])[:3]
            row["top3"] = [
                {"label": label_name(int(j)), "prob": float(probs[b, int(j)])}
                for j in top3
            ]

            if contrib_np is not None and concept_names is not None:
                c = contrib_np[b]
                order = np.argsort(-np.abs(c))[: min(args.topk_concepts, len(c))]
                row["top_concepts"] = [
                    {"concept": concept_names[int(i)], "contribution": float(c[int(i)])}
                    for i in order
                ]

            rows.append(row)
            global_idx += 1

    save_jsonl(rows, args.out_jsonl)
    print(f"Saved: {args.out_jsonl}")


if __name__ == "__main__":
    main()
