"""
Phase 1: validation-only hyperparameter selection.

  collect : gather every completed run into run_level/all_runs.csv (full precision)
  select  : choose K (or L) by highest mean validation Macro-F1 over the five seeds.
            The decision file is written from validation columns ONLY; test columns are
            never loaded by this code path.

Prespecified tie-breaking rule (fixed before any run finished): if two configurations
have mean validation Macro-F1 equal to within 1e-9, prefer (1) higher mean validation
accuracy, then (2) the smaller value (fewer concepts / shorter input).
"""
import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

SEEDS = [13, 21, 42, 87, 100]


def collect(root, val_only=False):
    rows = []
    for rd in sorted(glob.glob(os.path.join(root, "runs", "*"))):
        if not os.path.exists(os.path.join(rd, "DONE")):
            continue
        cfg = json.load(open(os.path.join(rd, "config.json")))
        val = json.load(open(os.path.join(rd, "val_summary.json")))
        test = {} if val_only else json.load(open(os.path.join(rd, "test_summary.json")))
        r = {"run_id": os.path.basename(rd), "arch": cfg["arch"], "K": cfg.get("num_concepts"),
             "L": cfg["max_length"], "seed": cfg["seed"], "selected_epoch": val["selected_epoch"],
             "epochs_run": val["epochs_run"], "train_runtime_s": val["train_runtime_s"],
             "trainable_params": cfg["trainable_params"], "head_params": cfg["head_params"],
             "lr": cfg["lr"], "batch_size": cfg["batch_size"], "optimizer": cfg["optimizer"],
             "tokenizer": cfg["tokenizer"], "git_commit": cfg["git_commit"],
             "checkpoint": os.path.join(rd, "best_model.pt"),
             "checkpoint_sha256_16": sha16(os.path.join(rd, "best_model.pt"))}
        for k, v in val.items():
            if k.startswith("val_") and not isinstance(v, list):
                r[k] = v
        for k, v in test.items():
            if isinstance(v, list):
                for i, x in enumerate(v):
                    r[f"{k}_{i}"] = x
            else:
                r[k] = v
        rows.append(r)
    return pd.DataFrame(rows)


_sha_cache = {}


def sha16(path):
    """First 16 hex chars of the checkpoint's SHA-256 (cached by size+mtime)."""
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime)
    cache_file = path + ".sha256"
    if os.path.exists(cache_file):
        return open(cache_file).read().strip()[:16]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    open(cache_file, "w").write(h.hexdigest())
    return h.hexdigest()[:16]


def select(df_val, factor):
    """df_val must contain only validation columns + the factor column + seed."""
    g = df_val.groupby(factor)
    stats = pd.DataFrame({
        "n_seeds": g["seed"].nunique(),
        "mean_val_macro_f1": g["val_macro_f1"].mean(),
        "sd_val_macro_f1": g["val_macro_f1"].std(ddof=1),
        "mean_val_accuracy": g["val_accuracy"].mean(),
        "sd_val_accuracy": g["val_accuracy"].std(ddof=1),
    }).reset_index()
    if not (stats["n_seeds"] == len(SEEDS)).all():
        raise SystemExit(f"incomplete seeds:\n{stats}")
    best = stats["mean_val_macro_f1"].max()
    cand = stats[stats["mean_val_macro_f1"] >= best - 1e-9]
    cand = cand.sort_values(["mean_val_accuracy", factor], ascending=[False, True])
    return int(cand.iloc[0][factor]), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "select"])
    ap.add_argument("--root", default="workdir")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--factor", choices=["K", "L"])
    ap.add_argument("--fixed", type=int, help="fixed value of the other factor (L for K sweep, K for L sweep)")
    ap.add_argument("--values", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.cmd == "collect":
        df = collect(a.root)
        df.to_csv(os.path.join(a.out_dir, "all_runs.csv"), index=False, float_format="%.12g")
        print(f"{len(df)} completed runs")
        return
    df = collect(a.root, val_only=True)  # test_summary.json files are never opened here
    other = "L" if a.factor == "K" else "K"
    values = [int(v) for v in a.values.split(",")]
    sub = df[(df.arch == "senn") & (df[other] == a.fixed) & (df[a.factor].isin(values)) & (df.seed.isin(SEEDS))]
    val_cols = ["run_id", a.factor, "seed", "val_macro_f1", "val_accuracy", "selected_epoch"]
    chosen, stats = select(sub[val_cols].copy(), a.factor)
    decision = {
        "factor": a.factor, "candidates": values, "fixed": {other: a.fixed}, "seeds": SEEDS,
        "selected": chosen, "criterion": "highest mean validation Macro-F1 over 5 seeds",
        "tie_break": "within 1e-9: higher mean validation accuracy, then smaller value",
        "per_value_validation_stats": stats.to_dict(orient="records"),
        "runs": sub[val_cols].to_dict(orient="records"),
        "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": "Computed from validation metrics only; test metrics were not read.",
    }
    path = os.path.join(a.out_dir, f"selection_{a.factor}.json")
    json.dump(decision, open(path, "w"), indent=1)
    print(stats.to_string(index=False))
    print(f"SELECTED {a.factor} = {chosen} -> {path}")


if __name__ == "__main__":
    main()
