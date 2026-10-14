"""
Phases 3, 5 and 6: input-level faithfulness, fixed-window stability and post-hoc
comparisons for one trained model, on a fixed prespecified test sample.

Outputs (JSONL, one row per document x method x condition; resumable per document):
  <out>/base.jsonl          original prediction, confidence, logits, contributions per doc
  <out>/attributions.jsonl  unit scores of every method on the unperturbed window
  <out>/faithfulness.jsonl  comprehensiveness / sufficiency rows (ranked, random, bottom)
  <out>/stability.jsonl     fixed-window sentence-removal stability rows
  <out>/legacy_stability.jsonl  re-run of the released (full-text, re-truncating) protocol
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch

from legalsenn import scotus, windows as W
from legalsenn.explanations import (ModelWrap, explain_units, faithfulness_doc, rank_units, sub_window,
                               TOPK_CONCEPTS)

DROP_RATES = (0.10, 0.20, 0.30)
REPEATS = 3
TOPK_UNITS = 3


def sample_indices(n_test: int, seed: int, size: int):
    """Same rule as the released explain_eval.py: RandomState(seed).shuffle(arange(N))[:size]."""
    rng = np.random.RandomState(seed)
    idx = np.arange(n_test)
    rng.shuffle(idx)
    return idx[:size]


def done_docs(path):
    s = set()
    if os.path.exists(path):
        for line in open(path):
            s.add(json.loads(line)["doc"])
    return s


def append(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ----- released (legacy) stability protocol, reproduced for comparison -----
def legacy_split(text):
    text = text.strip()
    return [p.strip() for p in re.split(r'(?<=[\.\?\!])\s+', text) if p.strip()] if text else []


def legacy_drop(text, rate, seed):
    sents = legacy_split(text)
    if len(sents) <= 1 or rate <= 0:
        return text, sents, np.ones(len(sents), bool)
    rng = np.random.RandomState(seed)
    n = len(sents)
    keep = np.ones(n, dtype=bool)
    k = min(int(round(rate * n)), n - 1)
    if k > 0:
        keep[rng.choice(n, size=k, replace=False)] = False
    return " ".join(s for s, kk in zip(sents, keep) if kk), sents, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--sample_size", type=int, default=200)
    ap.add_argument("--parts", default="faith,stab,legacy")
    ap.add_argument("--methods", default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")
    tok = scotus.get_tokenizer()
    data = scotus.tokenized_corpus(args.cache_dir, tok)["test"]
    mw = ModelWrap(args.run_dir, device, tok)
    methods = ([] if args.methods == "none" else args.methods.split(",") if args.methods else
               (["ce_occlusion", "ce_ig", "ig", "shap", "lime"] if mw.has_concepts else ["ig", "shap", "lime"]))
    parts = set(args.parts.split(","))
    idxs = sample_indices(len(data["label"]), args.sample_seed, args.sample_size)
    meta = {"run_dir": args.run_dir, "config": mw.cfg, "sample_seed": args.sample_seed,
            "sample_indices": [int(i) for i in idxs], "methods": methods, "drop_rates": DROP_RATES,
            "repeats": REPEATS, "topk_units": TOPK_UNITS, "topk_concepts": TOPK_CONCEPTS,
            "budgets": [0.05, 0.10, 0.20, 0.30], "n_random_controls": 10,
            "shap_nsamples": 512, "lime_num_samples": 512, "ig_steps": 32,
            "precision": "fp32 weights, TF32 matmul", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    import shap as _shap, captum as _captum  # noqa
    meta["versions"] = {"shap": _shap.__version__, "captum": _captum.__version__, "torch": torch.__version__}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)

    paths = {k: os.path.join(args.out, f"{k}.jsonl") for k in
             ("base", "attributions", "faithfulness", "stability", "legacy_stability")}
    marker = "stability" if "stab" in parts else ("faithfulness" if "faith" in parts else "legacy_stability")
    completed = done_docs(paths[marker])
    t0 = time.time()
    for n_done, di in enumerate(idxs):
        di = int(di)
        if di in completed:
            continue
        w = W.build_window(data["text"][di], data["ids"][di], data["offsets"][di], mw.L)
        full = W.pack(w.ids, np.ones(w.n, bool), mw.cls_id, mw.sep_id)
        base = mw.run([full])
        p0, lg0 = base["probs"][0], base["logits"][0]
        y = int(np.argmax(p0))
        brow = {"doc": di, "gold": int(data["label"][di]), "pred": y, "conf": float(p0[y]),
                "logits": lg0.tolist(), "n_visible": w.n, "n_units": w.n_units, "n_doc_tokens": w.n_doc_tokens,
                "truncated_last_unit": w.truncated_last}
        if mw.has_concepts:
            c0 = base["contrib"][0]
            brow["contrib"] = c0.tolist()
            brow["top_concepts"] = mw.top_concepts(c0, y).tolist()
            # exact additive reconstruction check: logit_y = sum_i effect_i + b_y
            brow["recon_err"] = float(abs(mw.effects(c0, y).sum() + mw.bc[y] - lg0[y]))
        rows_attr, rows_f, rows_s, rows_l = [], [], [], []
        orders = {}
        for mi, m in enumerate(methods if parts & {"faith", "stab"} else []):
            sc, diag = explain_units(mw, w, y, m, seed=di * 100 + mi, base=base)
            orders[m] = rank_units(sc)
            rows_attr.append({"doc": di, "method": m, "scores": sc.tolist(), **diag})
            if "faith" in parts:
                rng = np.random.RandomState(10_000 + di)  # same random controls for every method
                for r in faithfulness_doc(mw, w, y, p0, lg0, orders[m], rng):
                    rows_f.append({"doc": di, "method": m, "pred": y, **r})

        if "stab" in parts:
            for ri, rate in enumerate(DROP_RATES):
                for rep in range(REPEATS):
                    rng = np.random.RandomState(123 + di * 1000 + rep * 10 + ri)
                    dropped, keep = W.drop_units(w, rate, rng)
                    sw, orig_units = sub_window(w, keep)
                    pb = mw.run([W.pack(w.ids, keep, mw.cls_id, mw.sep_id)])
                    p2 = pb["probs"][0]
                    y2 = int(np.argmax(p2))
                    common = {"doc": di, "rate": rate, "rep": rep, "n_units": w.n_units,
                              "n_dropped": int(len(dropped)), "n_surv_units": int(sw.n_units),
                              "tokens_kept": int(keep.sum()), "pred": y, "pred_pert": y2,
                              "flip": int(y2 != y), "conf_orig": float(p0[y]), "conf_pert_same_class": float(p2[y]),
                              "conf_change": float(p0[y] - p2[y])}
                    if mw.has_concepts:
                        c2 = pb["contrib"][0]
                        t_orig = mw.top_concepts(base["contrib"][0], y)
                        rows_s.append({**common, "method": "concept_top6",
                                       "jaccard": W.jaccard(t_orig, mw.top_concepts(c2, y)),
                                       "jaccard_newpred_target": W.jaccard(t_orig, mw.top_concepts(c2, y2)),
                                       "chance": W.expected_jaccard_random(len(c2), TOPK_CONCEPTS)})
                    k = TOPK_UNITS
                    for mi, m in enumerate(methods):
                        if sw.n_units <= k:
                            rows_s.append({**common, "method": m, "jaccard": None, "chance": None,
                                           "skipped": "n_surv_units<=k"})
                            continue
                        sc2, _ = explain_units(mw, sw, y, m, seed=di * 100 + mi + 7, base=pb)
                        surv = set(orig_units.tolist())
                        top_orig = [u for u in orders[m] if u in surv][:k]
                        top_pert = orig_units[rank_units(sc2)[:k]]
                        rows_s.append({**common, "method": m, "jaccard": W.jaccard(top_orig, top_pert),
                                       "chance": W.expected_jaccard_random(sw.n_units, k)})

        if "legacy" in parts and mw.has_concepts:
            text = data["text"][di]
            orig_end_char = int(data["offsets"][di][min(len(data["ids"][di]), mw.L - 2) - 1, 1])
            for ri, rate in enumerate(DROP_RATES):
                for rep in range(REPEATS):
                    pert, sents, keep = legacy_drop(text, rate, seed=123 + di * 1000 + rep)
                    enc = tok(pert, add_special_tokens=False, return_offsets_mapping=True)
                    pids = np.asarray(enc["input_ids"][: mw.L - 2])
                    poff = np.asarray(enc["offset_mapping"][: mw.L - 2])
                    # map perturbed characters back to original positions to measure text drift
                    pos, starts, cur = [], [], 0
                    search = 0
                    for s, kk in zip(sents, keep):
                        o = text.find(s, search)
                        search = o + len(s)
                        if kk:
                            starts.append((cur, o)); cur += len(s) + 1
                    pstarts = np.asarray([a for a, _ in starts]); ostarts = np.asarray([b for _, b in starts])
                    j = np.searchsorted(pstarts, poff[:, 0], side="right") - 1
                    orig_pos = ostarts[j] + (poff[:, 0] - pstarts[j])
                    intr = float(np.mean(orig_pos >= orig_end_char)) if len(pids) else 0.0
                    pb = mw.run([np.concatenate([[mw.cls_id], pids, [mw.sep_id]])])
                    y2 = int(np.argmax(pb["probs"][0]))
                    t_orig = mw.top_concepts(base["contrib"][0], y)
                    rows_l.append({"doc": di, "rate": rate, "rep": rep, "flip": int(y2 != y),
                                   "jaccard": W.jaccard(t_orig, mw.top_concepts(pb["contrib"][0], y)),
                                   "frac_tokens_from_outside_window": intr,
                                   "n_sentences_full_doc": len(sents)})

        append(paths["base"], [brow])
        append(paths["attributions"], rows_attr)
        append(paths["faithfulness"], rows_f)
        append(paths["legacy_stability"], rows_l)
        append(paths["stability"], rows_s)  # written last: marks the document complete
        el = time.time() - t0
        print(f"doc {n_done + 1}/{len(idxs)} (test idx {di}) units={w.n_units} elapsed={el:.0f}s", flush=True)
    open(os.path.join(args.out, "DONE"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")


if __name__ == "__main__":
    main()
