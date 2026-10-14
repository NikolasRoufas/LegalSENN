# statistics.py
# Statistical helpers for paired multi-seed comparisons.
# The inferential unit for training-run comparisons is the seed: five paired runs.

from typing import Dict, Sequence

import numpy as np
from scipy import stats


def mean_sd(x: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else float("nan")}


def t_ci(x: Sequence[float], level: float = 0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * se
    return float(m - h), float(m + h)


def paired(a: Sequence[float], b: Sequence[float], level: float = 0.95) -> Dict:
    """
    Paired comparison of b - a over matched units (e.g. seeds).
    Reports the differences, mean, SD, SE, t-based CI, paired t-test,
    exact Wilcoxon signed-rank test and Cohen's d_z.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b - a
    n = len(d)
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    lo, hi = t_ci(d, level)
    t = stats.ttest_rel(b, a)
    try:
        w = stats.wilcoxon(b, a, alternative="two-sided", method="exact")
        w_p = float(w.pvalue)
    except ValueError:
        w_p = float("nan")
    return {
        "n_pairs": int(n), "differences": [float(x) for x in d], "mean_diff": float(d.mean()),
        "sd_diff": float(sd), "se_diff": float(se), "ci_level": level, "ci_low": lo, "ci_high": hi,
        "t": float(t.statistic), "df": int(n - 1), "p_paired_t": float(t.pvalue),
        "p_wilcoxon_exact": w_p, "cohens_dz": float(d.mean() / sd) if sd > 0 else float("nan"),
        "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum()),
    }


def welch(a: Sequence[float], b: Sequence[float]) -> Dict:
    r = stats.ttest_ind(b, a, equal_var=False)
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return {"t": float(r.statistic), "df": float(dof), "p": float(r.pvalue)}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    keys = sorted(pvals, key=lambda k: pvals[k])
    m = len(keys)
    adj, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


def doc_bootstrap_ci(x: Sequence[float], n_boot: int = 10000, level: float = 0.95, seed: int = 0):
    """Percentile bootstrap CI of the mean over documents."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    rng = np.random.RandomState(seed)
    means = x[rng.randint(0, len(x), (n_boot, len(x)))].mean(1)
    return float(np.percentile(means, 50 * (1 - level))), float(np.percentile(means, 100 - 50 * (1 - level)))


def tost_paired(d: Sequence[float], margin: float) -> Dict:
    """Two one-sided paired t-tests for |mean(d)| < margin (only used with a pre-justified margin)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t_lo = (d.mean() + margin) / se
    t_hi = (d.mean() - margin) / se
    p_lo = 1 - stats.t.cdf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    return {"margin": margin, "p_lower": float(p_lo), "p_upper": float(p_hi), "p_tost": float(max(p_lo, p_hi))}
