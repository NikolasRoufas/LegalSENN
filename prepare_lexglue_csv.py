# prepare_lexglue_csv.py
"""
Prepare CSV files (train/val/test) from LexGLUE tasks.

Uses Hugging Face Datasets: `coastalcph/lex_glue`.

Supported tasks (subsets):
  - ecthr_a      (multi-label, labels as list[int])
  - ecthr_b      (multi-label)
  - scotus       (single-label, label as int)
  - eurlex       (multi-label)
  - ledgar       (single-label)
  - unfair_tos   (multi-label)
  - case_hold    (multiple choice QA, not used here)

Output CSVs (per split):
  - text:  full document text (string)
  - label: primary label (int) as string (for single-label training)

For multi-label tasks, by default we keep only the first label as "primary"
to remain compatible with a single-label classifier. If you later want
true multi-label training, we can adapt this.
"""

import argparse
import os
import csv

from datasets import load_dataset


def normalize_text(example_text):
    """
    LexGLUE sometimes stores `text` as a list of paragraphs (ecthr_*),
    other times as a single string (scotus, eurlex, ledgar, etc.).
    This function always returns a single string.
    """
    if isinstance(example_text, list):
        # Join paragraphs with two newlines for readability.
        return "\n\n".join(str(t) for t in example_text)
    return str(example_text)


def extract_label(example, task, primary_label_only=True):
    """
    Return a single label (int) from a LexGLUE example, as a string.

    - For single-label tasks (scotus, ledgar):
        uses `example["label"]` (int).
    - For multi-label tasks (ecthr_a, ecthr_b, eurlex, unfair_tos):
        uses first element of example["labels"] by default.

    NOTE: This is a simplification for now to match single-label training.
    """
    if task in ["scotus", "ledgar"]:
        # single-label: `label` is an int
        lbl = example["label"]
        return str(int(lbl))

    elif task in ["ecthr_a", "ecthr_b", "eurlex", "unfair_tos"]:
        # multi-label: `labels` is a list[int] (possibly empty)
        labels = example.get("labels") or example.get("label")
        if labels is None:
            # Fallback
            return "0"
        if not isinstance(labels, list):
            # Sometimes HF shows label as a list-like even if single
            labels = [labels]
        if len(labels) == 0:
            # No label assigned, treat as "0"
            return "0"

        if primary_label_only:
            return str(int(labels[0]))
        else:
            # If you later want full multi-label, you could return
            # a space-separated string or JSON-encoded list here.
            return " ".join(str(int(l)) for l in labels)

    else:
        raise ValueError(f"Unsupported task: {task}")


def save_split_to_csv(dataset_split, task, out_path, primary_label_only=True):
    """
    Save one split (train/validation/test) of a LexGLUE subset to CSV.
    Columns: text,label
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])

        for ex in dataset_split:
            text = normalize_text(ex["text"])
            label = extract_label(ex, task, primary_label_only=primary_label_only)
            writer.writerow([text, label])

    print(f"[✔] Saved {len(dataset_split)} examples to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare LexGLUE task as CSVs")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="LexGLUE task: ecthr_a, ecthr_b, scotus, eurlex, ledgar, unfair_tos, case_hold (not recommended)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/lexglue_csv",
        help="Directory to store train/val/test CSVs",
    )
    parser.add_argument(
        "--primary_label_only",
        action="store_true",
        help="For multi-label tasks, keep only the first label as primary (default).",
    )

    args = parser.parse_args()

    task = args.task
    out_dir = args.output_dir
    primary_label_only = args.primary_label_only

    print(f"[*] Loading LexGLUE task: {task}")
    dataset = load_dataset("coastalcph/lex_glue", task)

    # The HF splits are usually: train, validation, test
    available_splits = list(dataset.keys())
    print(f"[*] Available splits: {available_splits}")

    # Map HF splits to CSV filenames
    split_name_map = {
        "train": "train.csv",
        "validation": "val.csv",
        "test": "test.csv",
    }

    for split, filename in split_name_map.items():
        if split not in dataset:
            print(f"[!] Split '{split}' not found for this task, skipping.")
            continue

        split_data = dataset[split]
        out_path = os.path.join(out_dir, task, filename)
        save_split_to_csv(split_data, task, out_path, primary_label_only=primary_label_only)

    print("\n[Done] CSV preparation completed.")


if __name__ == "__main__":
    main()
