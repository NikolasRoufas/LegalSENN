"""
Aggregate every pipeline result into machine-readable summaries, report tables
(4 decimals) and figures. All numbers are recomputed from the stored run-level and
per-document outputs; nothing is typed in by hand.

Usage: python -m experiments.aggregate --root ROOT --out results_summary
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

from experiments.select_hparams import SEEDS, collect
from legalsenn import statistics as S
from legalsenn.windows import expected_jaccard_random

ARCH_NAME = {"bert": "Legal-BERT", "concept": "Legal-BERT + Concept Layer", "senn": "SENN-inspired (full)",
             "hier": "Hierarchical Legal-BERT"}


def f4(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def msd(x):
    x = np.asarray(x, float)
    return f"{x.mean():.4f} ± {x.std(ddof=1):.4f}" if len(x) > 1 else f4(x.mean())


def md_table(df, path, title=None, note=None):
    lines = []
    if title:
        lines.append(f"**{title}**\n")
    cols = list(df.columns)
    lines.append("| " + " | ".join(map(str, cols)) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    if note:
        lines.append(f"\n{note}")
    open(path, "w").write("\n".join(lines) + "\n")


class Agg:
    def __init__(self, root, out):
        self.root, self.out = root, out
        for d in ("run_level", "per_document", "summaries", "tables", "figures", "configs", "logs"):
            os.makedirs(os.path.join(out, d), exist_ok=True)
        self.summary = {}

    def p(self, *a):
        return os.path.join(self.out, *a)

    def save_json(self, name, obj):
        json.dump(obj, open(self.p("summaries", name), "w"), indent=1, default=float)

    # ------------------------------------------------------------------ runs
    def runs(self):
        df = collect(self.root)
        conc = []
        for rid in df.run_id:
            f = os.path.join(self.root, "runs", rid, "concentration.json")
            conc.append(json.load(open(f)) if os.path.exists(f) else {})
        df = pd.concat([df, pd.DataFrame(conc)], axis=1)
        df["checkpoint_id"] = df.run_id + "@" + df.checkpoint_sha256_16
        df.to_csv(self.p("run_level", "all_runs.csv"), index=False, float_format="%.12g")
        for rid in df.run_id:  # epoch-level logs and configs
            rd = os.path.join(self.root, "runs", rid)
            os.makedirs(self.p("run_level", "epochs"), exist_ok=True)
            shutil.copy(os.path.join(rd, "metrics.jsonl"), self.p("run_level", "epochs", f"{rid}.jsonl"))
            shutil.copy(os.path.join(rd, "config.json"), self.p("configs", f"{rid}.json"))
        self.df = df
        return df

    def sel(self, factor):
        f = os.path.join(self.root, "results", "selection", f"selection_{factor}.json")
        if os.path.exists(f):
            shutil.copy(f, self.p("summaries", f"selection_{factor}.json"))
            return json.load(open(f))["selected"]
        return None

    # ------------------------------------------------------------------ phase 1
    def sweep(self, factor, values, fixed, name):
        df = self.df
        other = "L" if factor == "K" else "K"
        sub = df[(df.arch == "senn") & (df[other] == fixed) & df[factor].isin(values)].sort_values([factor, "seed"])
        if sub.empty:
            return
        chosen = self.sel(factor)
        rows = []
        for _, r in sub.iterrows():
            rows.append({factor: int(r[factor]), "Seed": int(r.seed), "Val Acc": f4(r.val_accuracy),
                         "Val Macro-F1": f4(r.val_macro_f1), "Test Acc": f4(r.test_accuracy),
                         "Test Macro-F1": f4(r.test_macro_f1), "Epoch": int(r.selected_epoch),
                         "Checkpoint": r.checkpoint_id})
        md_table(pd.DataFrame(rows), self.p("tables", f"{name}_seed_level.md"),
                 f"{name}: every run (Model C, {other} = {fixed})")
        summ, srows = {}, []
        for v, g in sub.groupby(factor):
            d = {"n_seeds": int(len(g))}
            for m in ("val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1",
                      "validation_mean_top6_abs_effect_share", "test_mean_top6_abs_effect_share", "train_runtime_s"):
                if m in g and g[m].notna().all():
                    d[m] = S.mean_sd(g[m])
            summ[int(v)] = d
            row = {factor: int(v), "n": len(g), "Val Acc": msd(g.val_accuracy), "Val Macro-F1": msd(g.val_macro_f1),
                   "Test Acc": msd(g.test_accuracy), "Test Macro-F1": msd(g.test_macro_f1)}
            if factor == "K" and "test_mean_top6_abs_effect_share" in g and g.test_mean_top6_abs_effect_share.notna().all():
                row["Top-6 Abs. Effect Share (test)"] = msd(g.test_mean_top6_abs_effect_share)
            row["Selected"] = "**selected**" if chosen == int(v) else ""
            srows.append(row)
        md_table(pd.DataFrame(srows), self.p("tables", f"{name}_summary.md"),
                 f"{name}: mean ± SD over seeds; selection by validation Macro-F1 only",
                 "Test columns are retrospective: they were not used for selection.")
        self.summary[name] = {"selected": chosen, "per_value": summ, "fixed": {other: fixed}}
        self.save_json(f"{name}.json", self.summary[name])

    # ------------------------------------------------------------------ phase 2
    def ablation(self, K, L):
        df = self.df
        pick = {"bert": df[(df.arch == "bert") & (df.L == L)], "concept": df[(df.arch == "concept") & (df.K == K) & (df.L == L)],
                "senn": df[(df.arch == "senn") & (df.K == K) & (df.L == L)]}
        if any(len(v) == 0 for v in pick.values()):
            return
        rows, summ = [], {}
        labels = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config", "scotus_issue_areas.json")))
        pc_rows = []
        for arch, g in pick.items():
            g = g.sort_values("seed")
            for _, r in g.iterrows():
                rows.append({"Model": ARCH_NAME[arch], "Seed": int(r.seed), "Val Acc": f4(r.val_accuracy),
                             "Val Macro-F1": f4(r.val_macro_f1), "Test Acc": f4(r.test_accuracy),
                             "Test Precision": f4(r.test_macro_precision), "Test Recall": f4(r.test_macro_recall),
                             "Test Macro-F1": f4(r.test_macro_f1), "Runtime (s)": f"{r.train_runtime_s:.0f}",
                             "Trainable params": f"{int(r.trainable_params):,}", "Checkpoint": r.checkpoint_id})
                pc_rows.append({"Model": ARCH_NAME[arch], "Seed": int(r.seed),
                                **{labels[str(i)]: f4(r[f"test_per_class_f1_{i}"]) for i in range(13)}})
            summ[arch] = {m: S.mean_sd(g[m]) for m in ("val_accuracy", "val_macro_f1", "test_accuracy",
                                                         "test_macro_precision", "test_macro_recall", "test_macro_f1",
                                                         "train_runtime_s")}
            summ[arch]["trainable_params"] = int(g.trainable_params.iloc[0])
            summ[arch]["per_class_f1_mean"] = [float(g[f"test_per_class_f1_{i}"].mean()) for i in range(13)]
        md_table(pd.DataFrame(rows), self.p("tables", "ablation_seed_level.md"), f"Ablation: every run (K = {K}, L = {L})")
        md_table(pd.DataFrame(pc_rows), self.p("tables", "ablation_per_class_f1.md"), "Ablation: test per-class F1")
        srows = []
        for arch in ("bert", "concept", "senn"):
            g = pick[arch]
            srows.append({"Model": ARCH_NAME[arch], "Val Macro-F1": msd(g.val_macro_f1), "Test Acc": msd(g.test_accuracy),
                          "Test Precision": msd(g.test_macro_precision), "Test Recall": msd(g.test_macro_recall),
                          "Test Macro-F1": msd(g.test_macro_f1)})
        md_table(pd.DataFrame(srows), self.p("tables", "ablation_summary.md"), "Ablation: mean ± SD over 5 seeds")
        comps, pvals = {}, {}
        for a_, b_ in (("bert", "concept"), ("bert", "senn"), ("concept", "senn")):
            m = pick[a_].merge(pick[b_], on="seed", suffixes=("_a", "_b")).sort_values("seed")
            for metric in ("test_macro_f1", "test_accuracy", "val_macro_f1"):
                key = f"{b_}_minus_{a_}:{metric}"
                comps[key] = S.paired(m[f"{metric}_a"], m[f"{metric}_b"])
                comps[key]["seeds"] = m.seed.tolist()
                if metric == "test_macro_f1":
                    pvals[key] = comps[key]["p_paired_t"]
        holm = S.holm(pvals)
        for k, v in holm.items():
            comps[k]["p_holm"] = v
        crow = []
        for k, v in comps.items():
            if not k.endswith("test_macro_f1"):
                continue
            crow.append({"Comparison (test Macro-F1)": k.split(":")[0].replace("_minus_", " − "),
                         "Per-seed differences": ", ".join(f"{x:+.4f}" for x in v["differences"]),
                         "Mean diff": f"{v['mean_diff']:+.4f}", "95% CI": f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]",
                         "t(4)": f"{v['t']:.3f}", "p (paired t)": f"{v['p_paired_t']:.4f}",
                         "p (Holm)": f"{v['p_holm']:.4f}", "p (exact Wilcoxon)": f"{v['p_wilcoxon_exact']:.4f}",
                         "d_z": f"{v['cohens_dz']:.2f}", "+/−": f"{v['n_positive']}/{v['n_negative']}"})
        md_table(pd.DataFrame(crow), self.p("tables", "ablation_paired.md"), "Ablation: seed-paired comparisons",
                 "With n = 5 pairs the smallest attainable two-sided exact Wilcoxon p-value is 0.0625.")
        self.summary["ablation"] = {"K": K, "L": L, "models": summ, "paired": comps}
        self.save_json("ablation.json", self.summary["ablation"])

    # ------------------------------------------------------------------ phase 7/8
    def long_context(self, K):
        df = self.df
        out = {}
        b = {L: df[(df.arch == "bert") & (df.L == L)].sort_values("seed") for L in (256, 512)}
        hier = df[df.arch == "hier"].sort_values("seed")
        rows = []
        for name, g in (("Legal-BERT 256", b[256]), ("Legal-BERT 512", b[512]), ("Hierarchical Legal-BERT 64×128", hier)):
            for _, r in g.iterrows():
                rows.append({"Model": name, "Seed": int(r.seed), "Val Acc": f4(r.val_accuracy),
                             "Val Macro-F1": f4(r.val_macro_f1), "Test Acc": f4(r.test_accuracy),
                             "Test Macro-F1": f4(r.test_macro_f1), "Test micro-F1": f4(r.test_micro_f1),
                             "Runtime (s)": f"{r.train_runtime_s:.0f}", "Checkpoint": r.checkpoint_id})
            if len(g):
                out[name] = {m: S.mean_sd(g[m]) for m in ("val_macro_f1", "test_accuracy", "test_macro_f1",
                                                            "test_micro_f1", "train_runtime_s")}
        if rows:
            md_table(pd.DataFrame(rows), self.p("tables", "long_context_seed_level.md"), "Long-document references: every run")
        paired = {}
        for label, a, c in (("SENN K={} 512 − 256".format(K), df[(df.arch == "senn") & (df.K == K) & (df.L == 256)],
                             df[(df.arch == "senn") & (df.K == K) & (df.L == 512)]),
                            ("Legal-BERT 512 − 256", b[256], b[512]),
                            ("Hierarchical − Legal-BERT 256", b[256], hier),
                            ("Hierarchical − Legal-BERT 512", b[512], hier)):
            m = a.merge(c, on="seed", suffixes=("_a", "_b")).sort_values("seed")
            if len(m) < 2:
                continue
            res = S.paired(m.test_macro_f1_a, m.test_macro_f1_b)
            res["seeds"] = m.seed.tolist()
            res["macro_f1_a"] = m.test_macro_f1_a.tolist(); res["macro_f1_b"] = m.test_macro_f1_b.tolist()
            res["welch_unpaired"] = S.welch(m.test_macro_f1_a, m.test_macro_f1_b)
            paired[label] = res
            tab = pd.DataFrame({"Seed": m.seed, "Macro-F1 (a)": m.test_macro_f1_a.map(f4),
                                "Macro-F1 (b)": m.test_macro_f1_b.map(f4),
                                "Paired difference (b − a)": (m.test_macro_f1_b - m.test_macro_f1_a).map(lambda x: f"{x:+.4f}")})
            md_table(tab, self.p("tables", f"paired_{label.split()[0].lower()}_{label.replace(' ', '_').replace('−', 'minus').replace('=', '')}.md"),
                     f"Paired test Macro-F1: {label}",
                     f"Mean diff {res['mean_diff']:+.4f}, SD {res['sd_diff']:.4f}, SE {res['se_diff']:.4f}, "
                     f"95% CI [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}], paired t(4) = {res['t']:.3f}, "
                     f"p = {res['p_paired_t']:.4f}; exact Wilcoxon p = {res['p_wilcoxon_exact']:.4f}.")
        ls = os.path.join(self.root, "results", "long_context", "length_summary.json")
        out["length_summary"] = json.load(open(ls)) if os.path.exists(ls) else None
        out["paired"] = paired
        out["lexglue_reference"] = json.load(open(os.path.join(os.path.dirname(__file__), "long_context", "lexglue_reference.json")))
        self.summary["long_context"] = out
        self.save_json("long_context.json", out)
        if os.path.exists(ls):
            shutil.copy(os.path.join(self.root, "results", "long_context", "document_lengths.csv"),
                        self.p("per_document", "document_lengths.csv"))

    # ------------------------------------------------------------------ phase 4
    def masking(self, K, L):
        rows, per_run = [], {}
        for d in sorted(glob.glob(os.path.join(self.root, "results", "masking", "*"))):
            if not os.path.exists(os.path.join(d, "DONE")):
                continue
            rid = os.path.basename(d)
            s = json.load(open(os.path.join(d, "summary.json")))
            per_run[rid] = s
            pdoc = pd.read_csv(os.path.join(d, "per_document.csv"))
            pdoc.insert(0, "run_id", rid)
            os.makedirs(self.p("per_document", "masking"), exist_ok=True)
            pdoc.to_csv(self.p("per_document", "masking", f"{rid}.csv.gz"), index=False, float_format="%.10g")
        if not per_run:
            return
        out = {}
        groups = {}
        for rid in per_run:
            arch, kk, ll, seed = rid.split("_")
            groups.setdefault(f"{arch}_{kk}_{ll}", []).append(rid)
        for g, rids in groups.items():
            out[g] = {}
            for subset in ("full_test", "historical_sample200"):
                out[g][subset] = {}
                for cond in ("top1", "top3", "top6", "random6", "bottom6"):
                    vals = {m: [per_run[r]["conditions"][subset][cond][m] for r in rids]
                            for m in per_run[rids[0]]["conditions"][subset][cond] if m != "n"}
                    out[g][subset][cond] = {m: S.mean_sd(v) for m, v in vals.items()}
                    out[g][subset][cond]["n_docs"] = per_run[rids[0]]["conditions"][subset][cond]["n"]
                    out[g][subset][cond]["n_seeds"] = len(rids)
            r = []
            for cond in ("top1", "top3", "top6", "random6", "bottom6"):
                c = out[g]["historical_sample200"][cond]; cf = out[g]["full_test"][cond]
                r.append({"Condition": cond, "Masked": {"top1": 1, "top3": 3}.get(cond, 6),
                          "Conf. drop (200)": f"{c['conf_drop']['mean']:.4f} ± {c['conf_drop']['sd']:.4f}",
                          "Flip (200)": f"{c['flip']['mean']:.4f} ± {c['flip']['sd']:.4f}",
                          "Conf. drop (1400)": f"{cf['conf_drop']['mean']:.4f} ± {cf['conf_drop']['sd']:.4f}",
                          "Flip (1400)": f"{cf['flip']['mean']:.4f} ± {cf['flip']['sd']:.4f}",
                          "Abs. effect share": f"{cf['abs_effect_share']['mean']:.4f}",
                          "Signed effect share": f"{cf['signed_effect_share']['mean']:+.4f}",
                          "Drop via own class": f"{cf['conf_drop_own_class_only']['mean']:+.4f}",
                          "Drop via competitors": f"{cf['conf_drop_from_competitors']['mean']:+.4f}",
                          "Max competitor Δlogit": f"{cf['max_competitor_delta_logit']['mean']:+.4f}"})
            md_table(pd.DataFrame(r), self.p("tables", f"masking_{g}.md"),
                     f"Concept masking ({g}); mean ± SD over {len(rids)} seeds",
                     "Decision-layer intervention (computational faithfulness), not an input-level test.")
            # gauge check (top-6 overlap with gauge-invariant rankings)
            gj = {}
            for rid in rids:
                pdoc = pd.read_csv(os.path.join(self.root, "results", "masking", rid, "per_document.csv"))
                t6 = pdoc[pdoc.condition == "top6"]
                if "jaccard_top6_vs_centered" in t6:
                    gj.setdefault("centered", []).append(t6.jaccard_top6_vs_centered.mean())
                    gj.setdefault("contrastive_runnerup", []).append(t6.jaccard_top6_vs_contrastive_runnerup.mean())
            out[g]["gauge_top6_jaccard"] = {k: S.mean_sd(v) for k, v in gj.items()}
        self.summary["masking"] = out
        self.save_json("masking.json", out)

    # ------------------------------------------------------------------ phases 3, 5, 6
    def explain(self, K, L):
        base_dir = os.path.join(self.root, "results", "explain")
        runs = sorted(d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.exists(os.path.join(d, "DONE")))
        if not runs:
            return
        F, ST, B = [], [], []
        for d in runs:
            rid = os.path.basename(d)
            arch = rid.split("_")[0]
            seed = int(rid.split("_s")[-1])
            for name, lst in (("faithfulness", F), ("stability", ST), ("base", B)):
                x = pd.read_json(os.path.join(d, f"{name}.jsonl"), lines=True)
                x.insert(0, "seed", seed); x.insert(0, "arch", arch); x.insert(0, "run_id", rid)
                lst.append(x)
        F, ST, B = pd.concat(F), pd.concat(ST), pd.concat(B)
        os.makedirs(self.p("per_document", "explain"), exist_ok=True)
        F.to_csv(self.p("per_document", "explain", "faithfulness.csv.gz"), index=False, float_format="%.10g")
        ST.to_csv(self.p("per_document", "explain", "stability.csv.gz"), index=False, float_format="%.10g")
        B.drop(columns=[c for c in ("logits", "contrib") if c in B]).to_csv(
            self.p("per_document", "explain", "base_predictions.csv.gz"), index=False, float_format="%.10g")
        for d in runs:
            for extra in ("attributions.jsonl", "meta.json"):
                src = os.path.join(d, extra)
                dst = self.p("per_document", "explain", f"{os.path.basename(d)}_{extra}")
                if extra == "meta.json":
                    shutil.copy(src, dst)
                else:
                    os.system(f"gzip -c {src} > {dst}.gz")
        out = {"faithfulness": self._faith(F, B), "stability": self._stab(ST)}
        lg = []
        for d in sorted(glob.glob(os.path.join(self.root, "results", "legacy123", "*"))):
            if os.path.exists(os.path.join(d, "DONE")):
                x = pd.read_json(os.path.join(d, "legacy_stability.jsonl"), lines=True)
                x.insert(0, "run_id", os.path.basename(d)); lg.append(x)
        if lg:
            lg = pd.concat(lg)
            lg.to_csv(self.p("per_document", "explain", "legacy_stability_sample123.csv.gz"), index=False)
            out["legacy_protocol"] = self._legacy(lg)
        out["reference_config_K32_L256"] = self._k32_stability()
        out["within_class_reference"] = self._within_class(B)
        self.summary["explain"] = out
        self.save_json("explanations.json", out)

    def _within_class(self, B):
        """
        Reference point for concept-index Jaccard: overlap between the top-6 concept sets of two
        DIFFERENT documents of the same model. High same-class overlap means a stable top-6 set
        is largely a property of the predicted class rather than of the individual document.
        """
        out, rows = {}, []
        for (arch, seed), g in B[B.arch.isin(["senn", "concept"])].groupby(["arch", "seed"]):
            tops = [set(t) for t in g.top_concepts]
            preds = g.pred.to_numpy()
            same, diff = [], []
            for i in range(len(tops)):
                for j in range(i + 1, len(tops)):
                    jac = len(tops[i] & tops[j]) / len(tops[i] | tops[j])
                    (same if preds[i] == preds[j] else diff).append(jac)
            out.setdefault(arch, {})[int(seed)] = {"same_class_pairs": float(np.mean(same)),
                                                   "different_class_pairs": float(np.mean(diff)),
                                                   "n_same": len(same), "n_diff": len(diff)}
        for arch, d in out.items():
            sm = [v["same_class_pairs"] for v in d.values()]; df_ = [v["different_class_pairs"] for v in d.values()]
            rows.append({"Model": ARCH_NAME[arch], "Top-6 Jaccard, different docs, same predicted class": msd(sm),
                         "Top-6 Jaccard, different docs, different class": msd(df_),
                         "Chance (K=16, k=6)": f"{expected_jaccard_random(16, 6):.4f}"})
        md_table(pd.DataFrame(rows), self.p("tables", "stability_within_class_reference.md"),
                 "Reference: concept top-6 overlap between different documents (200-doc sample, mean ± SD over seeds)")
        return out

    def _k32_stability(self):
        """Fixed-window and released-protocol concept stability at the reference configuration (K=32, L=256)."""
        fw, lg = [], []
        for d in sorted(glob.glob(os.path.join(self.root, "results", "stability_k32", "*"))):
            if os.path.exists(os.path.join(d, "DONE")):
                x = pd.read_json(os.path.join(d, "stability.jsonl"), lines=True); x.insert(0, "run_id", os.path.basename(d)); fw.append(x)
        for d in sorted(glob.glob(os.path.join(self.root, "results", "legacy123_k32", "*"))):
            if os.path.exists(os.path.join(d, "DONE")):
                x = pd.read_json(os.path.join(d, "legacy_stability.jsonl"), lines=True); x.insert(0, "run_id", os.path.basename(d)); lg.append(x)
        if not fw or not lg:
            return None
        fw, lg = pd.concat(fw), pd.concat(lg)
        fw = fw[fw.method == "concept_top6"].assign(rate=lambda z: z.rate.round(2))
        lg = lg.assign(rate=lg.rate.round(2))
        fw.to_csv(self.p("per_document", "explain", "stability_fixed_window_senn_K32_L256.csv.gz"), index=False)
        lg.to_csv(self.p("per_document", "explain", "legacy_stability_senn_K32_L256_sample123.csv.gz"), index=False)
        out, rows = {}, []
        for rate in (0.1, 0.2, 0.3):
            f, l = fw[fw.rate == rate], lg[lg.rate == rate]
            r = {"fixed_window": {"jaccard_all": S.mean_sd(f.jaccard), "jaccard_unchanged": S.mean_sd(f[f.flip == 0].jaccard),
                                  "jaccard_changed": S.mean_sd(f[f.flip == 1].jaccard) if (f.flip == 1).sum() > 1 else None,
                                  "n_changed": int((f.flip == 1).sum()), "flip_rate": float(f.flip.mean()), "n": int(len(f))},
                 "legacy": {"jaccard_all": S.mean_sd(l.jaccard), "jaccard_unchanged": S.mean_sd(l[l.flip == 0].jaccard),
                            "jaccard_changed": S.mean_sd(l[l.flip == 1].jaccard) if (l.flip == 1).sum() > 1 else None,
                            "flip_rate": float(l.flip.mean()), "n": int(len(l)),
                            "frac_tokens_from_outside_window": S.mean_sd(l.frac_tokens_from_outside_window),
                            "pct_with_outside_text": float((l.frac_tokens_from_outside_window > 0).mean() * 100)}}
            out[str(rate)] = r
            fwr, lgr = r["fixed_window"], r["legacy"]
            rows.append({"Drop": f"{int(rate * 100)}%",
                         "Released protocol: Jaccard / flip": f"{lgr['jaccard_all']['mean']:.4f} ± {lgr['jaccard_all']['sd']:.4f} / {lgr['flip_rate']:.4f}",
                         "Released: window tokens from outside": f"{lgr['frac_tokens_from_outside_window']['mean']:.4f}",
                         "Fixed window: Jaccard all": f"{fwr['jaccard_all']['mean']:.4f} ± {fwr['jaccard_all']['sd']:.4f}",
                         "Fixed: unchanged": f"{fwr['jaccard_unchanged']['mean']:.4f} (n={fwr['n'] - fwr['n_changed']})",
                         "Fixed: changed": (f"{fwr['jaccard_changed']['mean']:.4f} (n={fwr['n_changed']})" if fwr["jaccard_changed"] else f"— (n={fwr['n_changed']})"),
                         "Fixed: flip": f"{fwr['flip_rate']:.4f}"})
        md_table(pd.DataFrame(rows), self.p("tables", "stability_reference_config.md"),
                 "Concept top-6 stability at the reference configuration (SENN K=32, L=256; 5 seeds × 200 docs × 3 repeats)",
                 f"Chance Jaccard for K=32, k=6: {expected_jaccard_random(32, 6):.10f}. Released protocol uses its original "
                 "sample (RandomState(123)); fixed-window uses the RandomState(42) sample.")
        return out

    def _faith(self, F, B):
        out = {}
        # per run, per method, per budget: mean over docs of ranked/random/bottom
        g = F.groupby(["arch", "seed", "method", "budget", "control"])[["comp", "suff", "flip_without", "flip_only",
                                                                        "logit_drop_without", "logit_drop_only"]].mean()
        g = g.unstack("control")
        run = pd.DataFrame({
            "comp_ranked": g[("comp", "ranked")], "comp_random": g[("comp", "random")], "comp_bottom": g[("comp", "bottom")],
            "suff_ranked": g[("suff", "ranked")], "suff_random": g[("suff", "random")], "suff_bottom": g[("suff", "bottom")],
            "flip_without_ranked": g[("flip_without", "ranked")], "flip_without_random": g[("flip_without", "random")],
            "logit_drop_without_ranked": g[("logit_drop_without", "ranked")],
        }).reset_index()
        run["excess_comp"] = run.comp_ranked - run.comp_random
        run["suff_advantage"] = run.suff_random - run.suff_ranked
        run.to_csv(self.p("summaries", "faithfulness_run_level.csv"), index=False, float_format="%.10g")
        # document-level CI within each run (seed 42 as the illustrative run) and across-seed summary
        rows = []
        for (arch, method, b), gg in run.groupby(["arch", "method", "budget"]):
            r = {"Model": ARCH_NAME[arch], "Method": method, "Budget": f"{int(b * 100)}%", "n seeds": len(gg)}
            for m in ("comp_ranked", "comp_random", "comp_bottom", "excess_comp", "suff_ranked", "suff_random",
                      "suff_advantage", "flip_without_ranked", "flip_without_random"):
                r[m] = msd(gg[m])
            if len(gg) > 1:
                lo, hi = S.t_ci(gg.excess_comp)
                r["excess_comp 95% CI (seeds)"] = f"[{lo:+.4f}, {hi:+.4f}]"
            rows.append(r)
            out.setdefault(arch, {}).setdefault(method, {})[str(b)] = {
                m: S.mean_sd(gg[m]) for m in ("comp_ranked", "comp_random", "comp_bottom", "excess_comp",
                                              "suff_ranked", "suff_random", "suff_bottom", "suff_advantage",
                                              "flip_without_ranked", "flip_without_random", "logit_drop_without_ranked")}
        tab = pd.DataFrame(rows)
        md_table(tab, self.p("tables", "faithfulness_all.md"),
                 "Input-level faithfulness (200 test docs; mean ± SD over seeds of per-seed document means)",
                 "comp = p(ŷ|x) − p(ŷ|x∖R); suff = p(ŷ|x) − p(ŷ|R); excess_comp = ranked − random; "
                 "suff_advantage = random − ranked (higher = better).")
        # aggregate over budgets (AOPC-style mean over the 4 budgets) per run
        aopc = run.groupby(["arch", "seed", "method"])[["excess_comp", "suff_advantage", "comp_ranked", "suff_ranked"]].mean().reset_index()
        aopc.to_csv(self.p("summaries", "faithfulness_aopc_run_level.csv"), index=False, float_format="%.10g")
        # main question: relevance network (C) vs concept layer (B), seed-paired, per method
        cmp = {}
        crow = []
        pv = {}
        for method in sorted(set(aopc.method)):
            a = aopc[(aopc.arch == "concept") & (aopc.method == method)]
            c = aopc[(aopc.arch == "senn") & (aopc.method == method)]
            m = a.merge(c, on="seed", suffixes=("_B", "_C"))
            if len(m) < 2:
                continue
            for metric in ("excess_comp", "suff_advantage"):
                res = S.paired(m[f"{metric}_B"], m[f"{metric}_C"]); res["seeds"] = m.seed.tolist()
                cmp[f"{method}:{metric}"] = res
                pv[f"{method}:{metric}"] = res["p_paired_t"]
        hol = S.holm(pv)
        for k, v in cmp.items():
            v["p_holm"] = hol[k]
            crow.append({"Method : metric (AOPC over budgets)": k, "C − B per seed": ", ".join(f"{x:+.4f}" for x in v["differences"]),
                         "Mean": f"{v['mean_diff']:+.4f}", "95% CI": f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]",
                         "p (paired t)": f"{v['p_paired_t']:.4f}", "p (Holm)": f"{v['p_holm']:.4f}",
                         "p (Wilcoxon)": f"{v['p_wilcoxon_exact']:.4f}"})
        if crow:
            md_table(pd.DataFrame(crow), self.p("tables", "faithfulness_relevance_network.md"),
                     "Does the relevance network (C) improve input-level faithfulness over the concept layer (B)?")
        out["C_vs_B_paired"] = cmp
        # within-model method comparison: concept-grounded vs post-hoc, and every method vs random (one-sample over seeds)
        vs_random = {}
        for (arch, method), gg in aopc.groupby(["arch", "method"]):
            if len(gg) > 1:
                vs_random[f"{arch}:{method}"] = {"excess_comp": S.paired(np.zeros(len(gg)), gg.excess_comp),
                                                 "suff_advantage": S.paired(np.zeros(len(gg)), gg.suff_advantage)}
        out["vs_random_over_seeds"] = vs_random
        # within-model method comparisons (seed-paired): concept-grounded vs post-hoc rationales
        mrow, mcmp, mp = [], {}, {}
        pairs = [("ce_occlusion", "shap"), ("ce_occlusion", "lime"), ("ce_occlusion", "ig"), ("ce_ig", "ig")]
        for arch in ("concept", "senn"):
            for m1, m2 in pairs:
                a = aopc[(aopc.arch == arch) & (aopc.method == m1)]
                b = aopc[(aopc.arch == arch) & (aopc.method == m2)]
                m = a.merge(b, on="seed", suffixes=("_1", "_2"))
                if len(m) < 2:
                    continue
                for metric in ("excess_comp", "suff_advantage"):
                    key = f"{arch}:{m2}-{m1}:{metric}"
                    mcmp[key] = S.paired(m[f"{metric}_1"], m[f"{metric}_2"])
                    mp[key] = mcmp[key]["p_paired_t"]
        for k, v in S.holm(mp).items():
            mcmp[k]["p_holm"] = v
            arch, diff, metric = k.split(":")
            mrow.append({"Model": ARCH_NAME[arch], "Difference": diff.replace("-", " − "), "Metric (AOPC)": metric,
                         "Mean": f"{mcmp[k]['mean_diff']:+.4f}",
                         "95% CI": f"[{mcmp[k]['ci_low']:+.4f}, {mcmp[k]['ci_high']:+.4f}]",
                         "p (paired t)": f"{mcmp[k]['p_paired_t']:.4f}", "p (Holm)": f"{v:.4f}",
                         "+/−": f"{mcmp[k]['n_positive']}/{mcmp[k]['n_negative']}"})
        if mrow:
            md_table(pd.DataFrame(mrow), self.p("tables", "faithfulness_methods_paired.md"),
                     "Concept-grounded vs post-hoc rationales on the same models (seed-paired; positive = post-hoc better)")
        out["methods_paired"] = mcmp
        md_rows = []
        for (arch, method), gg in aopc.groupby(["arch", "method"]):
            md_rows.append({"Model": ARCH_NAME[arch], "Method": method, "n seeds": len(gg),
                            "Excess comp. (AOPC)": msd(gg.excess_comp), "Suff. advantage (AOPC)": msd(gg.suff_advantage),
                            "Comp. ranked": msd(gg.comp_ranked), "Suff. ranked": msd(gg.suff_ranked)})
        md_table(pd.DataFrame(md_rows), self.p("tables", "faithfulness_aopc.md"),
                 "Input-level faithfulness averaged over the four budgets (mean ± SD over 5 seeds)",
                 "Excess comprehensiveness = ranked − random; sufficiency advantage = random − ranked; higher is better for both.")
        # matched-class analysis: documents where B and C (same seed) predict the same label
        bb = B[["arch", "seed", "doc", "pred"]]
        piv = bb.pivot_table(index=["seed", "doc"], columns="arch", values="pred").reset_index()
        if {"concept", "senn"} <= set(piv.columns):
            piv["same_BC"] = piv.concept == piv.senn
            agree = piv.groupby("seed").same_BC.mean().to_dict()
            same = piv[piv.same_BC][["seed", "doc"]]
            Fm = F.merge(same, on=["seed", "doc"])
            gm = Fm.groupby(["arch", "seed", "method", "control"])[["comp", "suff"]].mean().unstack("control")
            mm = pd.DataFrame({"excess_comp": gm[("comp", "ranked")] - gm[("comp", "random")],
                               "suff_advantage": gm[("suff", "random")] - gm[("suff", "ranked")]}).reset_index()
            mres = {}
            for method in ("ce_occlusion", "ce_ig", "ig", "shap", "lime"):
                a = mm[(mm.arch == "concept") & (mm.method == method)]; c = mm[(mm.arch == "senn") & (mm.method == method)]
                m = a.merge(c, on="seed", suffixes=("_B", "_C"))
                if len(m) > 1:
                    mres[method] = {k: S.paired(m[f"{k}_B"], m[f"{k}_C"]) for k in ("excess_comp", "suff_advantage")}
            out["matched_class"] = {"B_C_prediction_agreement_by_seed": agree, "paired": mres}
        # figures
        try:
            self._fig_faith(run)
        except Exception as e:  # figures are optional; tables are authoritative
            print("figure error", e)
        return out

    def _stab(self, ST):
        out = {}
        st = ST.copy()
        st["rate"] = st.rate.round(2)
        st["stratum"] = np.where(st.flip == 1, "changed", "unchanged")
        rows = []
        for (arch, method, rate), g in st.groupby(["arch", "method", "rate"]):
            g = g[g.jaccard.notna()]
            if g.empty:
                continue
            res = {}
            for stratum, gs in (("all", g), ("unchanged", g[g.stratum == "unchanged"]), ("changed", g[g.stratum == "changed"])):
                per_seed = gs.groupby("seed").jaccard.mean()
                res[stratum] = {"n_perturbations": int(len(gs)), "n_docs": int(gs.doc.nunique()),
                                "jaccard_pooled_mean": float(gs.jaccard.mean()) if len(gs) else None,
                                "jaccard_pooled_sd": float(gs.jaccard.std(ddof=1)) if len(gs) > 1 else None,
                                "jaccard_seed_means": S.mean_sd(per_seed) if len(per_seed) > 1 else None,
                                "doc_bootstrap_ci_pooled": S.doc_bootstrap_ci(gs.groupby(["seed", "doc"]).jaccard.mean())
                                if len(gs) > 1 else None,
                                "chance_mean": float(gs.chance.mean()) if len(gs) else None}
            res["flip_rate"] = S.mean_sd(g.groupby("seed").flip.mean()) if g.seed.nunique() > 1 else float(g.flip.mean())
            res["conf_change"] = S.mean_sd(g.groupby("seed").conf_change.mean()) if g.seed.nunique() > 1 else float(g.conf_change.mean())
            if "jaccard_newpred_target" in g and g.jaccard_newpred_target.notna().any():
                ch = g[g.stratum == "changed"]
                res["changed_with_new_pred_target"] = float(ch.jaccard_newpred_target.mean()) if len(ch) else None
            out.setdefault(arch, {}).setdefault(method, {})[str(rate)] = res
            def cell(s):
                if not res[s]["n_perturbations"]:
                    return "— (n=0)"
                return f"{res[s]['jaccard_pooled_mean']:.4f} ± {res[s]['jaccard_pooled_sd'] if res[s]['jaccard_pooled_sd'] is not None else float('nan'):.4f} (n={res[s]['n_perturbations']})"
            fr = res["flip_rate"]["mean"] if isinstance(res["flip_rate"], dict) else res["flip_rate"]
            rows.append({"Model": ARCH_NAME[arch], "Explanation": method, "Drop": f"{int(rate * 100)}%",
                         "Jaccard all": cell("all"), "Jaccard unchanged label": cell("unchanged"),
                         "Jaccard changed label": cell("changed"), "Chance": f"{res['all']['chance_mean']:.4f}",
                         "Flip rate": f"{fr:.4f}"})
        md_table(pd.DataFrame(rows), self.p("tables", "stability_fixed_window.md"),
                 "Fixed-window stability (target = original prediction; pooled over seeds × docs × 3 repeats; mean ± SD)",
                 "concept_top6: top-6 concept Jaccard (chance = exact hypergeometric expectation for K, k=6). "
                 "Other rows: top-3 visible-sentence Jaccard in the shared rationale space (chance per doc).")
        return out

    def _legacy(self, lg):
        out = {}
        rows = []
        lg = lg.assign(rate=lg.rate.round(2))
        for rate, g in lg.groupby("rate"):
            ch = g[g.flip == 1]; un = g[g.flip == 0]
            out[str(rate)] = {"jaccard": S.mean_sd(g.jaccard), "flip_rate": float(g.flip.mean()),
                              "jaccard_unchanged": float(un.jaccard.mean()) if len(un) else None,
                              "jaccard_changed": float(ch.jaccard.mean()) if len(ch) else None,
                              "frac_tokens_from_outside_window": S.mean_sd(g.frac_tokens_from_outside_window),
                              "pct_perturbations_with_any_outside_text": float((g.frac_tokens_from_outside_window > 0).mean() * 100),
                              "n": int(len(g)), "n_runs": int(g.run_id.nunique())}
            rows.append({"Drop": f"{int(rate * 100)}%", "Jaccard": f"{g.jaccard.mean():.4f} ± {g.jaccard.std(ddof=1):.4f}",
                         "Flip rate": f"{g.flip.mean():.4f}",
                         "Perturbed-window tokens from beyond original window": f"{g.frac_tokens_from_outside_window.mean():.4f}"})
        md_table(pd.DataFrame(rows), self.p("tables", "stability_legacy_protocol.md"),
                 "Released (re-truncating) stability protocol re-run on its original sample (RandomState(123), 200 docs)")
        return out

    def _fig_faith(self, run):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        methods = ["ce_occlusion", "ce_ig", "ig", "shap", "lime"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for arch, ls in (("concept", "--"), ("senn", "-"), ("bert", ":")):
            for mi, m in enumerate(methods):
                g = run[(run.arch == arch) & (run.method == m)].groupby("budget")
                if not len(g):
                    continue
                mu = g.excess_comp.mean()
                axes[0].plot(mu.index * 100, mu.values, ls, color=f"C{mi}", label=f"{arch}:{m}")
                mu = g.suff_advantage.mean()
                axes[1].plot(mu.index * 100, mu.values, ls, color=f"C{mi}")
        for ax, t in zip(axes, ("Excess comprehensiveness (ranked − random)", "Sufficiency advantage (random − ranked)")):
            ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("Rationale budget (% visible tokens)"); ax.set_title(t, fontsize=10)
        axes[0].legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(self.p("figures", "faithfulness_budgets.png"), dpi=160); plt.close(fig)

    def figures(self, K, L):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        df = self.df
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, (factor, vals, other, fixed) in zip(axes, (("K", (8, 16, 32, 64), "L", 256), ("L", (128, 256, 512), "K", K))):
            sub = df[(df.arch == "senn") & (df[other] == fixed) & df[factor].isin(vals)]
            if sub.empty:
                continue
            for col, lab, mk in (("val_macro_f1", "validation", "o"), ("test_macro_f1", "test (retrospective)", "s")):
                g = sub.groupby(factor)[col]
                ax.errorbar(g.mean().index, g.mean().values, yerr=g.std(ddof=1).values, marker=mk, capsize=3, label=lab)
                ax.scatter(sub[factor], sub[col], s=8, alpha=0.4)
            ax.set_xscale("log", base=2); ax.set_xticks(vals); ax.set_xticklabels(vals)
            ax.set_xlabel(factor); ax.set_ylabel("Macro-F1"); ax.legend(fontsize=7)
            ax.set_title(f"Model C, {other} = {fixed} (mean ± SD, 5 seeds)", fontsize=9)
        fig.tight_layout(); fig.savefig(self.p("figures", "hyperparameter_sweeps.png"), dpi=160); plt.close(fig)
        pick = [("bert", df[(df.arch == "bert") & (df.L == L)]), ("concept", df[(df.arch == "concept") & (df.K == K) & (df.L == L)]),
                ("senn", df[(df.arch == "senn") & (df.K == K) & (df.L == L)]), ("bert512", df[(df.arch == "bert") & (df.L == 512)]),
                ("hier", df[df.arch == "hier"])]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for i, (n, g) in enumerate(pick):
            if g.empty:
                continue
            ax.scatter(np.full(len(g), i) + np.linspace(-0.12, 0.12, len(g)), g.test_macro_f1, s=18)
            ax.errorbar(i + 0.25, g.test_macro_f1.mean(), yerr=g.test_macro_f1.std(ddof=1), fmt="k_", capsize=4)
        ax.set_xticks(range(len(pick)))
        ax.set_xticklabels([f"A: Legal-BERT\nL={L}", f"B: Concept\nK={K}", f"C: SENN\nK={K}", "Legal-BERT\nL=512", "Hierarchical\n64×128"], fontsize=7)
        ax.set_ylabel("Test Macro-F1"); ax.set_title("Seed-level test Macro-F1", fontsize=9)
        fig.tight_layout(); fig.savefig(self.p("figures", "ablation_long_context.png"), dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="workdir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ag = Agg(a.root, a.out)
    ag.runs()
    K = ag.sel("K") or 32
    L = ag.sel("L") or 256
    ag.sweep("K", (8, 16, 32, 64), 256, "k_sweep")
    ag.sweep("L", (128, 256, 512), K, "l_sweep")
    ag.ablation(K, L)
    ag.long_context(K)
    ag.masking(K, L)
    ag.explain(K, L)
    ag.summary["chance_jaccard"] = {f"K={k},k=6": expected_jaccard_random(k, 6) for k in (8, 16, 32, 64)}
    rng = np.random.RandomState(2026)
    mc = {}
    for k in (8, 16, 32, 64):
        a_ = np.argsort(rng.rand(200000, k), 1)[:, :6]; b_ = np.argsort(rng.rand(200000, k), 1)[:, :6]
        inter = np.array([len(set(x) & set(y)) for x, y in zip(a_[:50000], b_[:50000])])
        mc[f"K={k}"] = float(np.mean(inter / (12 - inter)))
    ag.summary["chance_jaccard_monte_carlo_check_50000_pairs"] = mc
    ag.save_json("chance_jaccard.json", {"exact": ag.summary["chance_jaccard"], "monte_carlo": mc,
                                         "note": "mathematical check, not a neural-network experiment"})
    try:
        ag.figures(K, L)
    except Exception as e:
        print("figure error", e)
    json.dump(ag.summary, open(ag.p("summaries", "ALL_RESULTS.json"), "w"), indent=1, default=float)
    print("aggregated ->", a.out)


if __name__ == "__main__":
    main()
