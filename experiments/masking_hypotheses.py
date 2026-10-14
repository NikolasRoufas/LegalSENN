"""
Compare concept-masking outcomes under alternative concept-ranking rules. Re-uses the
saved decision-layer quantities (logits_before_after.npz: contributions c, classifier W, b)
of each masking run, so no model forward pass is needed. Each rule chooses the top-6 and
bottom-6 sets differently:

  abs_effect      |W[y,i] c_i|               (used in masking.py)
  signed_effect   W[y,i] c_i (top = largest, bottom = most negative)
  abs_contrib     |c_i|                      (ranking used by the released inference.py)
  abs_activation  not recoverable here (h not stored) - omitted
  gold_class      |W[gold,i] c_i|            (ranking for the gold instead of the predicted class)

Writes <out>/masking_hypotheses.csv and .json.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from legalsenn.explanations import softmax


def run_rule(lg, C, Wc, b, gold, rule, rng):
    res = {"top6": [], "bottom6": [], "random6": [], "top6_flip": [], "bottom6_flip": [], "random6_flip": []}
    for d in range(len(lg)):
        y = int(np.argmax(lg[d])); c = C[d]
        if rule == "abs_effect":
            score = np.abs(Wc[y] * c)
        elif rule == "signed_effect":
            score = Wc[y] * c
        elif rule == "abs_contrib":
            score = np.abs(c)
        elif rule == "gold_class":
            score = np.abs(Wc[gold[d]] * c)
        order = np.argsort(-score, kind="stable")
        p0 = softmax(lg[d])[y]
        for name, S in (("top6", order[:6]), ("bottom6", order[-6:]), ("random6", rng.choice(len(c), 6, replace=False))):
            l2 = lg[d] - Wc[:, S] @ c[S]
            res[name].append(p0 - softmax(l2)[y])
            res[f"{name}_flip"].append(int(np.argmax(l2) != y))
    return {k: float(np.mean(v)) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masking_root", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from legalsenn import scotus
    from experiments.explain_suite import sample_indices
    gold_all = scotus.tokenized_corpus(a.cache_dir)["test"]["label"]
    hist = sample_indices(len(gold_all), 42, 200)
    rows = []
    for d in sorted(glob.glob(os.path.join(a.masking_root, "*"))):
        f = os.path.join(d, "logits_before_after.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        for subset, idx in (("full_test", np.arange(len(gold_all))), ("historical_sample200", hist)):
            for rule in ("abs_effect", "signed_effect", "abs_contrib", "gold_class"):
                r = run_rule(z["before"][idx], z["contrib"][idx], z["W"], z["b"], gold_all[idx], rule,
                             np.random.RandomState(7))
                rows.append({"run_id": os.path.basename(d), "subset": subset, "rule": rule, **r})
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "masking_hypotheses.csv"), index=False, float_format="%.10g")
    df["config"] = df.run_id.str.rsplit("_", n=1).str[0]
    summ = df.groupby(["config", "subset", "rule"])[["top6", "bottom6", "random6", "top6_flip", "bottom6_flip", "random6_flip"]].agg(["mean", "std"])
    summ.to_csv(os.path.join(a.out, "masking_hypotheses_summary.csv"), float_format="%.6g")
    print(summ.xs("historical_sample200", level="subset").round(4).to_string())


if __name__ == "__main__":
    main()
