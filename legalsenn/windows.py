# windows.py
# Fixed-window input construction and perturbation.
#
# The model-visible window of a document is [CLS] + its first (L-2) WordPiece tokens
# + [SEP]. Every perturbation in the experimental pipeline operates on the tokens of
# that window only: tokens are removed (or kept) and the survivors are re-packed
# as [CLS] + kept + [SEP]. No text is re-tokenised and no token that was outside the
# original window can enter the perturbed input.

import re
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

_ABBREV = re.compile(
    r"(?:\b[A-Z]|\b(?:U\.S|S\.Ct|L\.Ed|U\.S\.C|No|Nos|v|vs|Mr|Mrs|Ms|Messrs|Inc|Co|Corp|Ltd|Jr|Sr|Dr|St|"
    r"Art|Sec|Secs|Stat|Cong|Ed|F|Supp|App|Cir|etc|cf|e\.g|i\.e|id|Id|Ann|Rev|Gen|Const|Amdt|ibid|Ct|"
    r"Cal|Fed|Reg|Pa|Wash|Ill|Mass|Tex|Okla|N\.Y|Ga|Ala|Fla|Va|Md|Ohio|Mich|Wis|Minn|Mo|Ky|Tenn|La|Ariz|"
    r"Colo|Conn|Del|Kan|Neb|Nev|Ore|Miss|Ark|Iowa|Me|Vt|Wyo|Mont|Utah|Ida|Okl))\.$"
)
_SENT_BREAK = re.compile(r"(?<=[.?!])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def sentence_spans(text: str) -> List[tuple]:
    """
    Character spans of sentence-like units: lines are split on newlines, then on
    sentence-final punctuation followed by whitespace and an upper-case letter or
    digit, without splitting after common legal abbreviations (U.S., S.Ct., No., v., ...).
    """
    spans = []
    for m in re.finditer(r"[^\n]+", text):
        line_start, line = m.start(), m.group()
        start = 0
        for b in _SENT_BREAK.finditer(line):
            piece = line[start:b.start()].rstrip()
            if _ABBREV.search(piece):
                continue
            if piece.strip():
                spans.append((line_start + start, line_start + b.start()))
            start = b.end()
        if line[start:].strip():
            spans.append((line_start + start, line_start + len(line)))
    return spans


@dataclass
class Window:
    """The exact model-visible token window of one document plus its sentence units."""
    ids: np.ndarray          # (n,) visible token ids, without special tokens
    unit: np.ndarray         # (n,) sentence-unit index for each visible token (0..U-1)
    n_units: int
    unit_sizes: np.ndarray   # (U,) tokens per unit inside the window
    truncated_last: bool     # last unit continues beyond the window
    n_doc_tokens: int        # full document length in tokens

    @property
    def n(self):
        return len(self.ids)


def build_window(text: str, ids: np.ndarray, offsets: np.ndarray, max_length: int) -> Window:
    n_vis = min(len(ids), max_length - 2)
    vis_ids = np.asarray(ids[:n_vis])
    starts = np.asarray(offsets[:n_vis, 0])
    spans = sentence_spans(text)
    span_starts = np.asarray([s for s, _ in spans])
    span_ends = np.asarray([e for _, e in spans])
    raw_unit = np.searchsorted(span_starts, starts, side="right") - 1
    raw_unit = np.clip(raw_unit, 0, len(spans) - 1)
    # re-index units to 0..U-1 in order of appearance inside the window
    uniq, unit = np.unique(raw_unit, return_inverse=True)
    sizes = np.bincount(unit, minlength=len(uniq))
    truncated = False
    if n_vis < len(ids):
        last_span_end = span_ends[uniq[-1]]
        truncated = bool(offsets[n_vis - 1, 1] < last_span_end)
    return Window(vis_ids, unit.astype(np.int64), len(uniq), sizes, truncated, len(ids))


def pack(window_ids: np.ndarray, keep: np.ndarray, cls_id: int, sep_id: int) -> np.ndarray:
    """[CLS] + kept visible tokens (original order) + [SEP]."""
    return np.concatenate([[cls_id], window_ids[keep], [sep_id]]).astype(np.int64)


def keep_mask_from_units(w: Window, dropped_units: Sequence[int]) -> np.ndarray:
    return ~np.isin(w.unit, np.asarray(list(dropped_units), dtype=np.int64))


def drop_units(w: Window, rate: float, rng: np.random.RandomState):
    """Remove round(rate*U) whole visible sentence units (at least one unit is kept)."""
    k = min(int(round(rate * w.n_units)), w.n_units - 1)
    dropped = np.sort(rng.choice(w.n_units, size=k, replace=False)) if k > 0 else np.zeros(0, dtype=np.int64)
    return dropped, keep_mask_from_units(w, dropped)


def budget_tokens(w: Window, unit_order: Sequence[int], budget_frac: float) -> np.ndarray:
    """
    Boolean token mask selecting exactly ceil(budget_frac * n) visible tokens by taking
    whole units in `unit_order` and a prefix of the next unit to hit the budget exactly.
    Identical token budgets make ranked, random and bottom rationales directly comparable.
    """
    target = int(np.ceil(budget_frac * w.n))
    sel = np.zeros(w.n, dtype=bool)
    got = 0
    for u in unit_order:
        if got >= target:
            break
        pos = np.flatnonzero(w.unit == u)
        take = pos[: target - got]
        sel[take] = True
        got += len(take)
    return sel


def pad_batch(seqs: List[np.ndarray], pad_id: int):
    m = max(len(s) for s in seqs)
    ids = np.full((len(seqs), m), pad_id, dtype=np.int64)
    am = np.zeros((len(seqs), m), dtype=np.int64)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
        am[i, : len(s)] = 1
    return ids, am


def jaccard(a, b) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def expected_jaccard_random(n: int, k: int) -> float:
    """
    Exact E[|A∩B| / |A∪B|] for two independent uniformly random k-subsets of n items.
    The overlap X = |A∩B| is hypergeometric(n, k, k); J = X / (2k - X).
    """
    from scipy.stats import hypergeom

    k = min(k, n)
    if k == 0:
        return 1.0
    xs = np.arange(max(0, 2 * k - n), k + 1)
    pmf = hypergeom(n, k, k).pmf(xs)
    return float(np.sum(pmf * xs / (2 * k - xs)))
