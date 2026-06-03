# utils_reporting.py
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def load_id2name_json(path: str) -> Dict[int, str]:
    """Loads {"0": "name", ...} -> {0: "name", ...}."""
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=axis, keepdims=True)


def topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    k = min(k, values.shape[-1])
    return np.argsort(-values, axis=-1)[..., :k]


def save_jsonl(rows: List[dict], path: str) -> None:
    safe_mkdir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_confusion_csv(cm: np.ndarray, id2label: Dict[int, str], path: str) -> None:
    """
    Saves confusion matrix with human-readable labels.
    Rows = true, cols = pred.
    """
    safe_mkdir(os.path.dirname(path) or ".")
    labels = [id2label[i] for i in range(len(id2label))]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + labels)
        for i, row in enumerate(cm):
            writer.writerow([labels[i]] + list(map(int, row)))


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_report_simple(
    y_true: Sequence[int], y_pred: Sequence[int], id2label: Dict[int, str]
) -> dict:
    """
    Lightweight per-class precision/recall/f1 + macro/weighted averages.
    No sklearn dependency.
    """
    n = len(id2label)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    report = {"per_class": {}}
    supports = []
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
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        }
        supports.append(support)
        f1s.append(f1)
        weights.append(support)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    weighted_f1 = float(np.average(f1s, weights=weights)) if sum(weights) > 0 else 0.0
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0

    report["accuracy"] = acc
    report["macro_f1"] = macro_f1
    report["weighted_f1"] = weighted_f1
    report["num_samples"] = int(len(y_true))
    return report


def summarize_concepts_by_label(
    y_true: Sequence[int],
    contributions: np.ndarray,  # shape (N, C)
    id2label: Dict[int, str],
    id2concept: Dict[int, str],
    top_k: int = 8,
) -> dict:
    """
    For each class label: average contribution vector, return top concepts by |avg|.
    Useful for paper tables: "Label -> typical concepts"
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
