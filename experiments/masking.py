"""
Concept-masking audit (decision-layer intervention).

For every test document, concepts are ranked by |effect_i(x, y)| with
effect_i(x, y) = W[y, i] * c_i(x) and y the original predicted class. Masking sets
c_i(x) = 0 for the masked set S before the linear classifier, i.e.
    logits'(x) = logits(x) - W[:, S] c_S(x).
This is a computational (decision-layer) intervention, not an input-level test.

Conditions: top-1, top-3, top-6, bottom-6, random-6 (N_RANDOM independent draws per doc).
Outputs per run: per_document.csv (one row per doc x condition, random = mean over draws),
random_draws.npz (every random draw), logits_before_after.npz, summary.json.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from legalsenn import scotus, windows as W
from legalsenn.explanations import ModelWrap, softmax

N_RANDOM = 100


def masked(logits, Wc, c, S):
    return logits - Wc[:, S] @ c[S]


def analyse(logits, Wc, c, y, S):
    """All quantities for one document and one masked set S."""
    eff = Wc[y] * c
    tot_abs = np.abs(eff).sum()
    lg2 = masked(logits, Wc, c, S)
    p, p2 = softmax(logits), softmax(lg2)
    others = np.delete(np.arange(len(logits)), y)
    comp0 = others[np.argmax(logits[others])]
    comp2 = others[np.argmax(lg2[others])]
    d = lg2 - logits
    # counterfactual: only the predicted-class logit changes, competitors held fixed
    lg_own = logits.copy(); lg_own[y] = lg2[y]
    p_own = softmax(lg_own)
    return {
        "conf_orig": p[y], "conf_masked": p2[y], "conf_drop": p[y] - p2[y],
        "flip": int(np.argmax(lg2) != y), "new_pred": int(np.argmax(lg2)),
        "logit_y_orig": logits[y], "logit_y_masked": lg2[y], "delta_logit_y": d[y],
        "margin_orig": logits[y] - logits[comp0], "margin_masked": lg2[y] - lg2[comp2],
        "max_competitor_delta_logit": d[others].max(), "mean_competitor_delta_logit": d[others].mean(),
        "runner_up_orig": int(comp0), "runner_up_delta_logit": d[comp0],
        "conf_drop_own_class_only": p[y] - p_own[y],
        "conf_drop_from_competitors": (p[y] - p2[y]) - (p[y] - p_own[y]),
        "abs_effect_share": np.abs(eff[S]).sum() / tot_abs,
        "signed_effect_sum": eff[S].sum(),
        "signed_effect_share": eff[S].sum() / tot_abs,  # stable denominator: total absolute effect
        "abs_c_share": np.abs(c[S]).sum() / np.abs(c).sum(),
        "competitor_abs_weighted_share": np.abs(Wc[others][:, S] * c[S]).sum() / np.abs(Wc[others] * c).sum(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")
    tok = scotus.get_tokenizer()
    data = scotus.tokenized_corpus(args.cache_dir, tok)["test"]
    mw = ModelWrap(args.run_dir, device, tok)
    assert mw.has_concepts
    seqs = [np.concatenate([[mw.cls_id], ids[: mw.L - 2], [mw.sep_id]]).astype(np.int64) for ids in data["ids"]]
    res = mw.run(seqs)
    logits_all, C_all = res["logits"], res["contrib"]
    K = C_all.shape[1]
    Wc = mw.Wc
    rows, rand_draws = [], []
    after = {k: [] for k in ("top1", "top3", "top6", "bottom6")}
    recon = []
    for d in range(len(seqs)):
        lg, c = logits_all[d], C_all[d]
        y = int(np.argmax(lg))
        eff = Wc[y] * c
        recon.append(abs(eff.sum() + mw.bc[y] - lg[y]))
        order = np.argsort(-np.abs(eff), kind="stable")
        # gauge check: logits are softmax-invariant to W[:, i] -> W[:, i] + u_i for all classes,
        # which changes effect_i(x, y) by u_i c_i(x). Compare the top-6 under (a) class-mean-centred
        # weights and (b) contrastive effects (predicted class vs runner-up), both invariant to u.
        others = np.delete(np.arange(len(lg)), y)
        r_up = others[np.argmax(lg[others])]
        top_c = np.argsort(-np.abs((Wc[y] - Wc.mean(0)) * c), kind="stable")[:6]
        top_r = np.argsort(-np.abs((Wc[y] - Wc[r_up]) * c), kind="stable")[:6]
        gauge = {"jaccard_top6_vs_centered": W.jaccard(order[:6], top_c),
                 "jaccard_top6_vs_contrastive_runnerup": W.jaccard(order[:6], top_r)}
        sets = {"top1": order[:1], "top3": order[:3], "top6": order[:6], "bottom6": order[-6:]}
        for name, S in sets.items():
            r = analyse(lg, Wc, c, y, S)
            rows.append({"doc": d, "condition": name, "pred": y, "gold": int(data["label"][d]),
                         "masked_concepts": " ".join(map(str, S.tolist())), **r,
                         **(gauge if name == "top6" else {})})
            after[name].append(masked(lg, Wc, c, S))
        rng = np.random.RandomState(1_000 + d)
        draws = [analyse(lg, Wc, c, y, rng.choice(K, 6, replace=False)) for _ in range(N_RANDOM)]
        dd = pd.DataFrame(draws)
        rand_draws.append(dd[["conf_drop", "flip", "abs_effect_share", "signed_effect_share"]].to_numpy())
        rr = dd.mean(numeric_only=True).to_dict()
        rr["conf_drop_sd_over_draws"] = float(dd["conf_drop"].std(ddof=1))
        rows.append({"doc": d, "condition": "random6", "pred": y, "gold": int(data["label"][d]),
                     "masked_concepts": "", **rr})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "per_document.csv"), index=False, float_format="%.10g")
    np.savez_compressed(os.path.join(args.out, "random_draws.npz"), draws=np.stack(rand_draws),
                        columns=np.array(["conf_drop", "flip", "abs_effect_share", "signed_effect_share"]))
    np.savez_compressed(os.path.join(args.out, "logits_before_after.npz"), before=logits_all, contrib=C_all,
                        W=Wc, b=mw.bc, **{k: np.stack(v) for k, v in after.items()})

    from experiments.explain_suite import sample_indices
    hist = set(sample_indices(len(seqs), 42, 200).tolist())
    summ = {"run_dir": args.run_dir, "K": K, "n_docs": len(seqs), "max_recon_err": float(max(recon)),
            "n_random_draws": N_RANDOM, "conditions": {}}
    for subset, mask in (("full_test", np.ones(len(df), bool)), ("historical_sample200", df["doc"].isin(hist).values)):
        g = df[mask].groupby("condition")
        summ["conditions"][subset] = {
            cond: {k: float(v) for k, v in grp.mean(numeric_only=True).items() if k not in ("doc", "pred", "gold")}
            | {"n": int(len(grp))} for cond, grp in g}
    json.dump(summ, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    open(os.path.join(args.out, "DONE"), "w").write("ok\n")
    print(json.dumps({c: {k: round(v[k], 4) for k in ("conf_drop", "flip", "abs_effect_share")}
                      for c, v in summ["conditions"]["full_test"].items()}, indent=1))


if __name__ == "__main__":
    main()
