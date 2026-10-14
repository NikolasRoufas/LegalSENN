"""
Top-6 Absolute Effect Share for concept models, on validation and test:
    Top6Share(x) = sum_{i in S6(x)} |effect_i(x, y)| / sum_j |effect_j(x, y)|
with y the predicted class. Reported as an explanation-concentration diagnostic only.
Writes <run_dir>/concentration.json.
"""
import argparse
import json
import os

import numpy as np
import torch

from legalsenn import scotus
from legalsenn.explanations import ModelWrap


def top6_share(logits, C, Wc):
    y = logits.argmax(1)
    eff = np.abs(Wc[y] * C)
    s = np.sort(eff, axis=1)[:, ::-1]
    return s[:, :6].sum(1) / s.sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--cache_dir", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = scotus.get_tokenizer()
    data = scotus.tokenized_corpus(a.cache_dir, tok)
    for rd in a.run_dirs:
        out = os.path.join(rd, "concentration.json")
        if os.path.exists(out):
            continue
        mw = ModelWrap(rd, torch.device("cuda"), tok)
        res = {}
        for split in ("validation", "test"):
            seqs = [np.concatenate([[mw.cls_id], ids[: mw.L - 2], [mw.sep_id]]).astype(np.int64)
                    for ids in data[split]["ids"]]
            r = mw.run(seqs)
            sh = top6_share(r["logits"], r["contrib"], mw.Wc)
            res[f"{split}_mean_top6_abs_effect_share"] = float(sh.mean())
            res[f"{split}_sd_top6_abs_effect_share"] = float(sh.std(ddof=1))
        json.dump(res, open(out, "w"), indent=1)
        print(rd, res)
        del mw
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
