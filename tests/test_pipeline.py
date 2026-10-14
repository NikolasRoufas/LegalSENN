"""Automated checks for the experimental pipeline."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from legalsenn import windows as W
from legalsenn import statistics as S
from legalsenn.model import build_model, load_from_run, ConceptLayerClassifier, LegalSENN
from legalsenn.explanations import ModelWrap, faithfulness_doc, rank_units, softmax, sub_window
from experiments.masking import analyse, masked
from experiments.select_hparams import select


# ---------------------------------------------------------------- tiny encoder
@pytest.fixture(scope="session")
def tiny_encoder(tmp_path_factory):
    from transformers import BertConfig, BertModel
    d = tmp_path_factory.mktemp("tiny_bert")
    cfg = BertConfig(vocab_size=200, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=128)
    torch.manual_seed(0)
    BertModel(cfg).save_pretrained(d)
    return str(d)


def make_run(tmp_path, enc, arch, K=8):
    torch.manual_seed(1)
    m = build_model(arch, enc, 5, num_concepts=K)
    rd = tmp_path / f"run_{arch}"
    rd.mkdir()
    torch.save(m.state_dict(), rd / "best_model.pt")
    json.dump({"arch": arch, "encoder_name": enc, "num_labels": 5, "num_concepts": K, "max_length": 64},
              open(rd / "config.json", "w"))
    return str(rd), m


class Tok:
    cls_token_id, sep_token_id, pad_token_id = 101, 102, 0


# ---------------------------------------------------------------- model checks
def test_logit_reconstruction_and_effects(tiny_encoder):
    for arch in ("senn", "concept"):
        torch.manual_seed(0)
        m = build_model(arch, tiny_encoder, 5, num_concepts=8).eval()
        ids = torch.randint(5, 199, (3, 20))
        out = m(input_ids=ids, attention_mask=torch.ones_like(ids))
        c = out["contributions"].double().detach().numpy()
        Wc = m.classifier.weight.double().detach().numpy()
        b = m.classifier.bias.double().detach().numpy()
        lg = out["logits"].double().detach().numpy()
        for i in range(3):
            for y in range(5):
                assert abs((Wc[y] * c[i]).sum() + b[y] - lg[i, y]) < 1e-5
        if arch == "senn":
            assert torch.allclose(out["contributions"], out["concepts"] * out["relevances"])
        else:
            assert torch.equal(out["contributions"], out["concepts"])
            assert not hasattr(m, "relevance_mlp")


def test_dimensions_and_gradient_update(tiny_encoder):
    m = build_model("senn", tiny_encoder, 5, num_concepts=8)
    assert m.concept_layer.out_features == 8
    assert m.relevance_mlp[0].in_features == 32 and m.relevance_mlp[-1].out_features == 8
    assert m.classifier.in_features == 8 and m.classifier.out_features == 5
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    ids = torch.randint(5, 199, (4, 16))
    loss = torch.nn.functional.cross_entropy(m(ids, torch.ones_like(ids))["logits"], torch.tensor([0, 1, 2, 3]))
    loss.backward(); opt.step()
    changed = [n for n, p in m.named_parameters() if not torch.equal(p, before[n])]
    assert any(n.startswith("encoder.") for n in changed)
    assert any(n.startswith("relevance_mlp") for n in changed)
    assert "classifier.weight" in changed


def test_dropout_train_vs_eval(tiny_encoder):
    m = build_model("senn", tiny_encoder, 5, num_concepts=8)
    ids = torch.randint(5, 199, (2, 16)); am = torch.ones_like(ids)
    m.eval()
    assert torch.equal(m(ids, am)["logits"], m(ids, am)["logits"])
    m.train(); torch.manual_seed(0)
    assert not torch.equal(m(ids, am)["logits"], m(ids, am)["logits"])


def test_checkpoint_reconstruction_uses_config_K(tmp_path, tiny_encoder):
    rd, m = make_run(tmp_path, tiny_encoder, "senn", K=12)
    m2 = load_from_run(rd)
    assert m2.concept_layer.out_features == 12
    for (n1, p1), (n2, p2) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2)


# ---------------------------------------------------------------- ranking / masking
def test_ranking_top_bottom_and_masking():
    rng = np.random.RandomState(0)
    Wc = rng.randn(4, 10); c = rng.randn(10); b = rng.randn(4)
    lg = Wc @ c + b
    y = int(np.argmax(lg))
    eff = Wc[y] * c
    order = np.argsort(-np.abs(eff), kind="stable")
    assert np.all(np.diff(np.abs(eff[order])) <= 0)
    top6, bottom6 = order[:6], order[-6:]
    assert set(top6).isdisjoint(order[6:]) and len(set(bottom6)) == 6
    S_ = np.array([1, 4, 7])
    mask = np.ones(10); mask[S_] = 0
    np.testing.assert_allclose(masked(lg, Wc, c, S_), Wc @ (c * mask) + b, atol=1e-12)
    r = analyse(lg, Wc, c, y, S_)
    assert abs(r["abs_effect_share"] - np.abs(eff[S_]).sum() / np.abs(eff).sum()) < 1e-12
    assert abs(r["signed_effect_sum"] - eff[S_].sum()) < 1e-12
    assert abs(r["delta_logit_y"] + eff[S_].sum()) < 1e-12
    assert abs(r["conf_drop"] - r["conf_drop_own_class_only"] - r["conf_drop_from_competitors"]) < 1e-12
    full = analyse(lg, Wc, c, y, np.arange(10))
    assert abs(full["abs_effect_share"] - 1) < 1e-12


def test_bottom_mask_can_change_confidence_through_competitors():
    # concept 1 barely affects class 0 (the prediction) but strongly supports class 1
    Wc = np.array([[2.0, 0.01], [0.0, -3.0]]); c = np.array([2.0, 1.0]); b = np.zeros(2)
    lg = Wc @ c + b
    r = analyse(lg, Wc, c, 0, np.array([1]))
    assert r["abs_effect_share"] < 0.01
    assert r["conf_drop"] > 0.01 and r["conf_drop_from_competitors"] > r["conf_drop_own_class_only"]


# ---------------------------------------------------------------- jaccard
def test_jaccard_and_chance():
    assert W.jaccard([1, 2, 3], [1, 2, 3]) == 1.0
    assert W.jaccard([1, 2], [3, 4]) == 0.0
    assert abs(W.jaccard([1, 2, 3], [2, 3, 4]) - 0.5) < 1e-12
    assert abs(W.expected_jaccard_random(32, 6) - 0.1110286362) < 1e-9
    rng = np.random.RandomState(0)
    mc = np.mean([W.jaccard(rng.choice(32, 6, replace=False), rng.choice(32, 6, replace=False)) for _ in range(200000)])
    assert abs(mc - W.expected_jaccard_random(32, 6)) < 2e-3
    assert W.expected_jaccard_random(6, 6) == 1.0


# ---------------------------------------------------------------- fixed window
TEXT = ("329 U.S. 29\nCHAMPLIN REFINING CO. v. UNITED STATES et al.\nNo. 21.\n"
        "Mr. Justice JACKSON delivered the opinion of the Court. The Commission acted. "
        "It found that the pipeline was a common carrier. We agree with the District Court. "
        "The judgment is affirmed. ") * 6


@pytest.fixture(scope="session")
def tok():
    try:
        from legalsenn.scotus import get_tokenizer
        return get_tokenizer()
    except Exception:
        pytest.skip("Legal-BERT tokenizer not available offline")


def test_sentence_units_respect_abbreviations():
    spans = W.sentence_spans("Mr. Smith went to the U.S. Court. It ruled. See 329 U.S. 29, 67 S.Ct. 1.")
    texts = [s for s in spans]
    assert len(texts) == 3


def test_fixed_window_perturbation_never_imports_outside_tokens(tok):
    enc = tok(TEXT, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = np.asarray(enc["input_ids"]), np.asarray(enc["offset_mapping"])
    L = 64
    w = W.build_window(TEXT, ids, off, L)
    assert w.n == L - 2 and np.array_equal(w.ids, ids[: L - 2])
    assert np.all(np.diff(w.unit) >= 0) and w.unit.max() == w.n_units - 1
    rng = np.random.RandomState(0)
    for rate in (0.1, 0.2, 0.3):
        dropped, keep = W.drop_units(w, rate, rng)
        seq = W.pack(w.ids, keep, Tok.cls_token_id, Tok.sep_token_id)
        assert seq[0] == Tok.cls_token_id and seq[-1] == Tok.sep_token_id
        assert len(seq) <= L
        assert np.array_equal(seq[1:-1], w.ids[keep])           # survivors, original order
        assert len(dropped) == min(int(round(rate * w.n_units)), w.n_units - 1)
        assert not np.isin(w.unit[keep], dropped).any()          # whole units removed
        sw, orig = sub_window(w, keep)
        assert sw.n_units == w.n_units - len(dropped) and set(orig).isdisjoint(dropped)
    # padding never enters the window
    assert not np.isin(w.ids, [Tok.pad_token_id]).any()


def test_budget_tokens_exact(tok):
    enc = tok(TEXT, add_special_tokens=False, return_offsets_mapping=True)
    w = W.build_window(TEXT, np.asarray(enc["input_ids"]), np.asarray(enc["offset_mapping"]), 128)
    order = np.arange(w.n_units)[::-1]
    for b in (0.05, 0.1, 0.2, 0.3):
        sel = W.budget_tokens(w, order, b)
        assert sel.sum() == int(np.ceil(b * w.n))
        # first selected unit is the first unit in the order, taken whole when budget allows
        first = order[0]
        if w.unit_sizes[first] <= sel.sum():
            assert sel[w.unit == first].all()


# ---------------------------------------------------------------- comprehensiveness / sufficiency
class FakeWrap:
    """Model whose class-0 logit equals the number of 'evidence' tokens (id 7) in the input."""
    cls_id, sep_id = 101, 102

    def run(self, seqs):
        lg = np.array([[float((s == 7).sum()), 0.0] for s in seqs])
        return {"logits": lg, "probs": softmax(lg)}


def test_comprehensiveness_sufficiency_definitions():
    ids = np.array([7, 7, 7, 1, 1, 1, 1, 1, 1, 1])
    unit = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
    w = W.Window(ids, unit, 4, np.bincount(unit), False, 10)
    fw = FakeWrap()
    full = W.pack(ids, np.ones(10, bool), 101, 102)
    base = fw.run([full]); p0, lg0 = base["probs"][0], base["logits"][0]
    rows = faithfulness_doc(fw, w, 0, p0, lg0, np.array([0, 1, 2, 3]), np.random.RandomState(0))
    r = {(x["budget"], x["control"]): x for x in rows}
    top = r[(0.30, "ranked")]  # 3 tokens = the whole evidence unit
    assert abs(top["comp"] - (p0[0] - 0.5)) < 1e-12                         # removing evidence -> uniform
    assert abs(top["suff"] - (p0[0] - softmax(np.array([3.0, 0.0]))[0])) < 1e-12
    assert top["n_tokens"] == 3 and top["flip_without"] == 0
    assert abs(r[(0.30, "bottom")]["comp"]) < 1e-12                     # bottom units carry no evidence
    assert r[(0.30, "ranked")]["comp"] >= r[(0.30, "random")]["comp"]


def test_rank_units_deterministic_ties():
    assert list(rank_units(np.array([1.0, 3.0, 3.0, 0.0]))) == [1, 2, 0, 3]


# ---------------------------------------------------------------- statistics
def test_paired_statistics():
    a = [0.60, 0.62, 0.61, 0.63, 0.59]
    b = [0.61, 0.64, 0.61, 0.66, 0.60]
    r = S.paired(a, b)
    d = np.array(b) - np.array(a)
    assert abs(r["mean_diff"] - d.mean()) < 1e-12
    assert abs(r["sd_diff"] - np.sqrt(((d - d.mean()) ** 2).sum() / 4)) < 1e-12   # sample SD
    from scipy import stats
    h = stats.t.ppf(0.975, 4) * r["sd_diff"] / np.sqrt(5)
    assert abs(r["ci_low"] - (d.mean() - h)) < 1e-12 and abs(r["ci_high"] - (d.mean() + h)) < 1e-12
    assert abs(r["p_paired_t"] - stats.ttest_rel(b, a).pvalue) < 1e-12
    assert S.mean_sd([1, 2, 3])["sd"] == 1.0
    adj = S.holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert abs(adj["x"] - 0.03) < 1e-12 and abs(adj["z"] - 0.06) < 1e-12 and abs(adj["y"] - 0.06) < 1e-12


# ---------------------------------------------------------------- selection (validation only)
def test_selection_uses_validation_and_tie_break():
    rows = []
    for k, f1, acc in ((8, 0.60, 0.70), (16, 0.65, 0.71), (32, 0.65, 0.72), (64, 0.64, 0.73)):
        for s in (13, 21, 42, 87, 100):
            rows.append({"run_id": f"{k}_{s}", "K": k, "seed": s, "val_macro_f1": f1, "val_accuracy": acc,
                         "selected_epoch": 3})
    chosen, stats_ = select(pd.DataFrame(rows), "K")
    assert chosen == 32                                         # tie on F1 -> higher val accuracy
    df = pd.DataFrame(rows)
    with pytest.raises(SystemExit):
        select(df.iloc[:-1], "K")                                  # a missing seed is refused


def test_seed_matching_in_paired_inputs():
    a = pd.DataFrame({"seed": [13, 21, 42, 87, 100], "f": [1, 2, 3, 4, 5]})
    b = pd.DataFrame({"seed": [100, 87, 42, 21, 13], "f": [5, 4, 3, 2, 1]})
    m = a.merge(b, on="seed", suffixes=("_a", "_b"))
    assert (m.f_a == m.f_b).all()
