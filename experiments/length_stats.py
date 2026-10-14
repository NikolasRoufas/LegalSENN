"""
Phase 7.1: document-length analysis of LexGLUE-SCOTUS in Legal-BERT WordPiece tokens,
and a structural check of what the first 128/256/512 tokens contain.

Content heuristic (documented, approximate): the start of the opinion body is located
as the first match of a typical opening ("delivered the opinion of the Court",
"PER CURIAM", "MR. JUSTICE ... delivered", "opinion of the Court") ; tokens before that
point are treated as caption / syllabus / counsel material.
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from legalsenn import scotus

OPINION_START = re.compile(
    r"(delivered the opinion of the court|per curiam|announced the judgment of the court|"
    r"delivered an opinion|opinion of the court\b)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = scotus.tokenized_corpus(a.cache_dir)
    rows = []
    for split in scotus.SPLITS:
        for i, (text, ids, off) in enumerate(zip(data[split]["text"], data[split]["ids"], data[split]["offsets"])):
            n = len(ids)
            m = OPINION_START.search(text)
            body_char = m.end() if m else None
            body_tok = int(np.searchsorted(off[:, 0], body_char)) if m else None
            r = {"split": split, "doc": i, "label": int(data[split]["label"][i]), "n_tokens": n,
                 "n_chars": len(text), "opinion_start_found": bool(m), "opinion_start_token": body_tok}
            for L in (128, 256, 512):
                vis = min(n, L - 2)
                r[f"visible_frac_{L}"] = vis / n if n else 1.0
                r[f"body_tokens_visible_{L}"] = (max(0, vis - body_tok) if m else None)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "document_lengths.csv"), index=False)
    summ = {}
    for split, g in list(df.groupby("split")) + [("all", df)]:
        n = g["n_tokens"]
        s = {"n_docs": int(len(g)), "mean": float(n.mean()), "median": float(n.median()),
             **{f"p{q}": float(np.percentile(n, q)) for q in (25, 75, 90, 95)},
             "min": int(n.min()), "max": int(n.max()),
             **{f"pct_longer_than_{L}": float((n > L - 2).mean() * 100) for L in (128, 256, 512)},
             **{f"mean_visible_frac_{L}": float(g[f"visible_frac_{L}"].mean()) for L in (128, 256, 512)},
             **{f"median_visible_frac_{L}": float(g[f"visible_frac_{L}"].median()) for L in (128, 256, 512)},
             "pct_opinion_start_found": float(g["opinion_start_found"].mean() * 100),
             "median_opinion_start_token": float(g["opinion_start_token"].median())}
        f = g[g.opinion_start_found]
        for L in (128, 256, 512):
            s[f"pct_docs_with_no_opinion_body_in_first_{L}"] = float((f[f"body_tokens_visible_{L}"] == 0).mean() * 100)
            s[f"mean_opinion_body_tokens_in_first_{L}"] = float(f[f"body_tokens_visible_{L}"].mean())
        summ[split] = s
    json.dump(summ, open(os.path.join(a.out, "length_summary.json"), "w"), indent=1)
    print(json.dumps(summ["all"], indent=1))


if __name__ == "__main__":
    main()
