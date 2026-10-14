# scotus.py
# LexGLUE-SCOTUS loading, pinned to a dataset revision, plus cached full-document
# tokenisation so that every experiment builds model inputs from the same tokens.

import hashlib
import os
import pickle
from typing import Dict, List

import numpy as np
import torch

DATASET_NAME = "coastalcph/lex_glue"
DATASET_CONFIG = "scotus"
# Pinned Hub revision used by the experimental pipeline.
DATASET_REVISION = "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"
ENCODER_NAME = "nlpaueb/legal-bert-base-uncased"
ENCODER_REVISION = "15b570cbf88259610b082a167dacc190124f60f6"

SPLITS = ("train", "validation", "test")


def load_scotus(revision: str = DATASET_REVISION) -> Dict[str, dict]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, revision=revision)
    return {s: {"text": list(ds[s]["text"]), "label": np.asarray(ds[s]["label"], dtype=np.int64)} for s in SPLITS}


def split_fingerprint(texts: List[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(hashlib.sha256(t.encode("utf-8")).digest())
    return h.hexdigest()


def get_tokenizer(name: str = ENCODER_NAME, revision: str = ENCODER_REVISION):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True)


def tokenized_corpus(cache_dir: str, tokenizer=None) -> Dict[str, dict]:
    """
    Returns {split: {"text", "label", "ids": [np.int32 array], "offsets": [np.int32 (n,2)]}}
    where ids/offsets cover the FULL document without special tokens.
    """
    path = os.path.join(cache_dir, f"scotus_tok_{DATASET_REVISION[:8]}_{ENCODER_REVISION[:8]}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    tokenizer = tokenizer or get_tokenizer()
    data = load_scotus()
    for s in SPLITS:
        enc = tokenizer(data[s]["text"], add_special_tokens=False, return_offsets_mapping=True)
        data[s]["ids"] = [np.asarray(x, dtype=np.int32) for x in enc["input_ids"]]
        data[s]["offsets"] = [np.asarray(x, dtype=np.int32) for x in enc["offset_mapping"]]
        data[s]["fingerprint"] = split_fingerprint(data[s]["text"])
    os.makedirs(cache_dir, exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".tmp", path)
    return data


def flat_inputs(ids_list, max_length: int, cls_id: int, sep_id: int, pad_id: int):
    """[CLS] + first (max_length-2) tokens + [SEP] + padding; identical to HF truncation."""
    n = len(ids_list)
    input_ids = np.full((n, max_length), pad_id, dtype=np.int64)
    attn = np.zeros((n, max_length), dtype=np.int64)
    for i, ids in enumerate(ids_list):
        body = ids[: max_length - 2]
        seq = np.concatenate([[cls_id], body, [sep_id]])
        input_ids[i, : len(seq)] = seq
        attn[i, : len(seq)] = 1
    return torch.from_numpy(input_ids), torch.from_numpy(attn)


def hier_inputs(ids_list, max_segments: int, segment_length: int, cls_id: int, sep_id: int, pad_id: int):
    """Split each document into consecutive chunks of (segment_length-2) tokens, each wrapped in [CLS]/[SEP]."""
    n = len(ids_list)
    body_len = segment_length - 2
    input_ids = np.full((n, max_segments, segment_length), pad_id, dtype=np.int64)
    attn = np.zeros((n, max_segments, segment_length), dtype=np.int64)
    for i, ids in enumerate(ids_list):
        for s in range(max_segments):
            chunk = ids[s * body_len: (s + 1) * body_len]
            if len(chunk) == 0:
                break
            seq = np.concatenate([[cls_id], chunk, [sep_id]])
            input_ids[i, s, : len(seq)] = seq
            attn[i, s, : len(seq)] = 1
    return torch.from_numpy(input_ids), torch.from_numpy(attn)
