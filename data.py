# data.py

import json
from typing import Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class LegalTextDataset(Dataset):
    """
    Simple Dataset for legal text classification from a CSV file.

    CSV format:
        text_col:  column with the raw text
        label_col: string or int label (jurisdiction / issue area)
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer_name: str,
        label2id: Dict[str, int],
        max_length: int = 512,
        text_col: str = "text",
        label_col: str = "label",
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.text_col = text_col
        self.label_col = label_col
        self.label2id = label2id
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        label_str = str(row[self.label_col])
        label_id = self.label2id[label_str]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)        # (T,)
        attention_mask = enc["attention_mask"].squeeze(0)

        return input_ids, attention_mask, torch.tensor(label_id, dtype=torch.long)


def build_label_map(csv_path: str, label_col: str = "label") -> Dict[str, int]:
    """
    Build label2id mapping from the training CSV.
    """
    df = pd.read_csv(csv_path)
    labels = sorted(df[label_col].astype(str).unique().tolist())
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    return label2id


def save_label_map(label2id: Dict[str, int], path: str) -> None:
    with open(path, "w") as f:
        json.dump(label2id, f, indent=2)


def load_label_map(path: str) -> Dict[str, int]:
    with open(path, "r") as f:
        return json.load(f)
