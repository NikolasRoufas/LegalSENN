# explanations.py
# Explanation evaluation engine for the experimental pipeline
# (input-level faithfulness, fixed-window stability, post-hoc comparisons).
#
# All explanations are mapped to one shared representation: a score per visible
# sentence unit of the fixed model input window (see windows.py). Faithfulness and
# stability are then measured identically for every model and explanation method.

import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from legalsenn.model import load_from_run
from legalsenn import windows as W

TOPK_CONCEPTS = 6


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class ModelWrap:
    def __init__(self, run_dir: str, device, tok, batch_size: int = 64):
        self.run_dir = run_dir
        with open(os.path.join(run_dir, "config.json")) as f:
            self.cfg = json.load(f)
        self.arch = self.cfg["arch"]
        self.L = self.cfg["max_length"]
        self.device = device
        self.model = load_from_run(run_dir, device)
        self.model.eval()
        self.bs = batch_size
        self.cls_id, self.sep_id, self.pad_id = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
        self.has_concepts = self.arch in ("senn", "concept")
        if self.has_concepts:
            self.Wc = self.model.classifier.weight.detach().double().cpu().numpy()  # (C, K)
            self.bc = self.model.classifier.bias.detach().double().cpu().numpy()    # (C,)
        self.word_emb = self.model.encoder.embeddings.word_embeddings

    @torch.no_grad()
    def run(self, seqs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        logits, contrib = [], []
        for i in range(0, len(seqs), self.bs):
            ids, am = W.pad_batch(seqs[i:i + self.bs], self.pad_id)
            out = self.model(input_ids=torch.from_numpy(ids).to(self.device),
                             attention_mask=torch.from_numpy(am).to(self.device))
            logits.append(out["logits"].double().cpu().numpy())
            if self.has_concepts:
                contrib.append(out["contributions"].double().cpu().numpy())
        res = {"logits": np.concatenate(logits)}
        res["probs"] = softmax(res["logits"])
        if self.has_concepts:
            res["contrib"] = np.concatenate(contrib)
        return res

    # ----- concept effects -----
    def effects(self, contrib: np.ndarray, y: int) -> np.ndarray:
        """effect_i(x, y) = w[y, i] * c_i(x)."""
        return self.Wc[y] * contrib

    def top_concepts(self, contrib: np.ndarray, y: int, k: int = TOPK_CONCEPTS) -> np.ndarray:
        e = self.effects(contrib, y)
        return np.argsort(-np.abs(e), kind="stable")[:k]

    # ----- integrated gradients over input word embeddings -----
    def ig_token_attr(self, seq: np.ndarray, target: Callable, n_steps: int = 32) -> (np.ndarray, float):
        """
        Integrated Gradients (Sundararajan et al., 2017) w.r.t. the word embeddings of the
        visible tokens; baseline replaces every non-special token by [PAD] (position and
        segment embeddings are added by the encoder for both input and baseline).
        Uses a Gauss-Legendre rule with n_steps points. Returns per-token attribution
        (sum over the embedding dimension, excluding [CLS]/[SEP]) and the completeness gap.
        """
        ids = torch.from_numpy(seq).to(self.device).unsqueeze(0)
        base_ids = ids.clone()
        base_ids[0, 1:-1] = self.pad_id
        x = self.word_emb(ids).detach()
        b = self.word_emb(base_ids).detach()
        am = torch.ones_like(ids)
        nodes, weights = np.polynomial.legendre.leggauss(n_steps)
        alphas = torch.tensor((nodes + 1) / 2, dtype=x.dtype, device=self.device)
        wts = torch.tensor(weights / 2, dtype=x.dtype, device=self.device)
        grads = torch.zeros_like(x)
        chunk = 16
        for s in range(0, n_steps, chunk):
            a = alphas[s:s + chunk].view(-1, 1, 1)
            inp = (b + a * (x - b)).requires_grad_(True)
            out = self.model(input_ids=None, attention_mask=am.expand(len(a), -1), inputs_embeds=inp)
            val = target(out)
            g, = torch.autograd.grad(val.sum(), inp)
            grads += (g * wts[s:s + chunk].view(-1, 1, 1)).sum(0, keepdim=True)
        attr = ((x - b) * grads).sum(-1)[0]
        with torch.no_grad():
            fx = target(self.model(input_ids=None, attention_mask=am, inputs_embeds=x)).item()
            fb = target(self.model(input_ids=None, attention_mask=am, inputs_embeds=b)).item()
        gap = float(attr.sum().item() - (fx - fb))
        return attr[1:-1].double().cpu().numpy(), gap


def unit_sum(w: W.Window, token_scores: np.ndarray) -> np.ndarray:
    return np.bincount(w.unit, weights=token_scores, minlength=w.n_units)


def sub_window(w: W.Window, keep: np.ndarray) -> (W.Window, np.ndarray):
    """Window restricted to kept tokens; returns it with the original unit id of each new unit."""
    unit = w.unit[keep]
    orig_units, new_unit = np.unique(unit, return_inverse=True)
    sw = W.Window(w.ids[keep], new_unit.astype(np.int64), len(orig_units),
                  np.bincount(new_unit, minlength=len(orig_units)), w.truncated_last, w.n_doc_tokens)
    return sw, orig_units


# ---------------------------------------------------------------------------
# Sentence-unit attribution methods. Each returns a score per unit of `w`
# (higher = more support for class y) and a small dict of diagnostics.
# ---------------------------------------------------------------------------

def _unit_masks_to_seqs(mw: ModelWrap, w: W.Window, Z: np.ndarray) -> List[np.ndarray]:
    seqs = []
    for z in Z:
        keep = z.astype(bool)[w.unit]
        seqs.append(W.pack(w.ids, keep, mw.cls_id, mw.sep_id))
    return seqs


def attr_ce_occlusion(mw: ModelWrap, w: W.Window, y: int, top: np.ndarray, base_contrib: np.ndarray):
    """
    Concept-effect occlusion: importance of unit u = S(x) - S(x without u), where
    S(x) = sum_{i in top} effect_i(x, y) is the net effect of the explanation's own
    top-|effect| concepts on the class-y logit (top fixed from the unperturbed input).
    """
    Z = np.ones((w.n_units, w.n_units), dtype=np.int8) - np.eye(w.n_units, dtype=np.int8)
    res = mw.run(_unit_masks_to_seqs(mw, w, Z))
    S0 = mw.effects(base_contrib, y)[top].sum()
    S = (mw.Wc[y][top] * res["contrib"][:, top]).sum(1)
    return S0 - S, {}


def attr_ce_ig(mw: ModelWrap, w: W.Window, y: int, top: np.ndarray, n_steps: int = 32):
    """Integrated Gradients of S(x) = sum_{i in top} effect_i(x, y) w.r.t. the input embeddings."""
    wy = torch.tensor(mw.Wc[y][top], dtype=torch.float32, device=mw.device)
    top_t = torch.as_tensor(top, device=mw.device)
    tok, gap = mw.ig_token_attr(W.pack(w.ids, np.ones(w.n, bool), mw.cls_id, mw.sep_id),
                                lambda out: (out["contributions"][:, top_t] * wy).sum(-1), n_steps)
    return unit_sum(w, tok), {"ig_gap": gap}


def attr_logit_ig(mw: ModelWrap, w: W.Window, y: int, n_steps: int = 32):
    """Integrated Gradients of the class-y logit, token attributions summed per unit."""
    tok, gap = mw.ig_token_attr(W.pack(w.ids, np.ones(w.n, bool), mw.cls_id, mw.sep_id),
                                lambda out: out["logits"][:, y], n_steps)
    return unit_sum(w, tok), {"ig_gap": gap}


def attr_kernel_shap(mw: ModelWrap, w: W.Window, y: int, seed: int, nsamples: int = 512):
    """
    KernelSHAP (Lundberg & Lee, 2017; shap.KernelExplainer) over sentence-unit presence.
    Value function: p(y | window with absent units removed); background = all units removed.
    """
    import shap

    def f(Z):
        return mw.run(_unit_masks_to_seqs(mw, w, Z))["probs"][:, y]

    np.random.seed(seed)
    ex = shap.KernelExplainer(f, np.zeros((1, w.n_units)))
    phi = ex.shap_values(np.ones((1, w.n_units)), nsamples=nsamples, silent=True, l1_reg=False)
    return np.asarray(phi).reshape(-1)[: w.n_units], {}


def attr_lime(mw: ModelWrap, w: W.Window, y: int, seed: int, num_samples: int = 512):
    """
    LIME (Ribeiro et al., 2016) over sentence units, following LimeTextExplainer's
    sampling (remove a uniformly drawn number of units), cosine distance * 100,
    exponential kernel width 25, Ridge(alpha=1) surrogate fit on all features.
    """
    from lime.lime_base import LimeBase
    from sklearn.metrics.pairwise import pairwise_distances

    d = w.n_units
    rs = np.random.RandomState(seed)
    Z = np.ones((num_samples, d), dtype=np.int8)
    if d > 1:
        sizes = rs.randint(1, d, num_samples - 1)
        for i, s in enumerate(sizes, start=1):
            Z[i, rs.choice(d, s, replace=False)] = 0
    probs = mw.run(_unit_masks_to_seqs(mw, w, Z))["probs"]
    dist = pairwise_distances(Z, Z[0:1], metric="cosine").ravel() * 100
    kernel = lambda dd: np.sqrt(np.exp(-(dd ** 2) / 25 ** 2))
    lb = LimeBase(kernel, random_state=rs)
    _, exp, score, _ = lb.explain_instance_with_data(Z, probs, dist, y, num_features=d, feature_selection="none")
    phi = np.zeros(d)
    for j, v in exp:
        phi[j] = v
    return phi, {"lime_r2": float(score)}


def explain_units(mw: ModelWrap, w: W.Window, y: int, method: str, seed: int, base: Optional[dict] = None):
    if method in ("ce_occlusion", "ce_ig"):
        if base is None:
            base = mw.run([W.pack(w.ids, np.ones(w.n, bool), mw.cls_id, mw.sep_id)])
        contrib = base["contrib"][0]
        top = mw.top_concepts(contrib, y)
        if method == "ce_occlusion":
            return attr_ce_occlusion(mw, w, y, top, contrib)
        return attr_ce_ig(mw, w, y, top)
    if method == "ig":
        return attr_logit_ig(mw, w, y)
    if method == "shap":
        return attr_kernel_shap(mw, w, y, seed)
    if method == "lime":
        return attr_lime(mw, w, y, seed)
    raise ValueError(method)


def rank_units(scores: np.ndarray) -> np.ndarray:
    """Descending score order; ties broken by position for determinism."""
    return np.lexsort((np.arange(len(scores)), -scores))


# ---------------------------------------------------------------------------
# Input-level faithfulness: comprehensiveness / sufficiency at matched budgets
# ---------------------------------------------------------------------------

BUDGETS = (0.05, 0.10, 0.20, 0.30)
N_RANDOM = 10


def faithfulness_doc(mw: ModelWrap, w: W.Window, y: int, p0: np.ndarray, logit0: np.ndarray,
                     order: np.ndarray, rng: np.random.RandomState) -> List[dict]:
    """
    For each budget b, rationale R_b = top-ranked visible tokens (whole units, exact token
    budget). Comprehensiveness = p(y|x) - p(y|x without R_b); sufficiency difference =
    p(y|x) - p(y|R_b only). Controls: N_RANDOM random unit orders and the reversed (bottom) order,
    using the identical token budget.
    """
    orders = {"ranked": [order], "bottom": [order[::-1]],
              "random": [rng.permutation(w.n_units) for _ in range(N_RANDOM)]}
    seqs, meta = [], []
    for b in BUDGETS:
        for kind, olist in orders.items():
            for r, o in enumerate(olist):
                sel = W.budget_tokens(w, o, b)
                seqs.append(W.pack(w.ids, ~sel, mw.cls_id, mw.sep_id)); meta.append((b, kind, r, "without"))
                seqs.append(W.pack(w.ids, sel, mw.cls_id, mw.sep_id)); meta.append((b, kind, r, "only"))
    res = mw.run(seqs)
    pred0 = int(np.argmax(p0))
    rows = {}
    for (b, kind, r, op), pr, lg in zip(meta, res["probs"], res["logits"]):
        key = (b, kind)
        d = rows.setdefault(key, {"budget": b, "control": kind, "comp": [], "suff": [], "flip_without": [],
                                  "flip_only": [], "logit_drop_without": [], "logit_drop_only": [],
                                  "n_tokens": int(np.ceil(b * w.n))})
        if op == "without":
            d["comp"].append(p0[y] - pr[y]); d["flip_without"].append(int(np.argmax(pr) != pred0))
            d["logit_drop_without"].append(logit0[y] - lg[y])
        else:
            d["suff"].append(p0[y] - pr[y]); d["flip_only"].append(int(np.argmax(pr) != pred0))
            d["logit_drop_only"].append(logit0[y] - lg[y])
    out = []
    for d in rows.values():
        out.append({k: (float(np.mean(v)) if isinstance(v, list) else v) for k, v in d.items()})
    return out
