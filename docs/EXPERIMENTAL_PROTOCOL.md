# Experimental protocol

This document describes the protocol implemented by the experimental pipeline. It contains
no results.

## Data
- LexGLUE-SCOTUS (`coastalcph/lex_glue`, configuration `scotus`), pinned to a Hub revision in
  `legalsenn/scotus.py`; official chronological train / validation / test splits; 13 issue-area
  classes (names in `config/scotus_issue_areas.json`).
- SHA-256 fingerprints of each split are stored in every run's `config.json`.
- Tokeniser: `nlpaueb/legal-bert-base-uncased` fast WordPiece (uncased), pinned revision. Every
  document is tokenised once without special tokens. A flat model with maximum length L reads
  `[CLS] + first L−2 tokens + [SEP]`, which is identical to Hugging Face truncation.

## Models (`legalsenn/model.py`)
| Id | `--arch` | Head on the Legal-BERT [CLS] vector (dropout 0.1) |
|---|---|---|
| A | `bert` | `logits = W z + b` |
| B | `concept` | `h = W_c z + b_c`, `logits = V h + d` |
| C | `senn` | `h = W_c z + b_c`, `a = W_2 ReLU(W_1 z + b_1) + b_2`, `c = h ⊙ a`, `logits = W c + b` |
| H | `hier` | up to 64 consecutive chunks of 126 tokens, each `[CLS] chunk [SEP]`, encoded by Legal-BERT; 2-layer transformer over chunk vectors with learned chunk positions; max-pool; linear layer |

## Training (`legalsenn/run_train.py`), identical for all models by default
- AdamW (PyTorch defaults, weight decay 0.01), learning rate 2e-5, constant schedule,
  batch size 16 (model H: micro-batch 2 with gradient accumulation and gradient checkpointing).
- At most 10 epochs, early stopping with patience 3 on validation Macro-F1, and the
  checkpoint with the best validation Macro-F1 retained.
- bf16 autocast with fp32 master weights; cross-entropy loss.
- Determinism: seeded Python/NumPy/PyTorch, seeded shuffling generator, deterministic cuDNN,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Seeds 13, 21, 42, 87, 100.
- The test split is evaluated once, after training, with the selected checkpoint.
  Validation results are written to `val_summary.json` before the test split is read; test
  results go to a separate `test_summary.json`.

## Hyperparameter selection (`experiments/select_hparams.py`)
- Concept dimension K ∈ {8, 16, 32, 64} at L = 256, then maximum length L ∈ {128, 256, 512}
  at the selected K.
- Criterion: highest mean validation Macro-F1 over the five seeds. Tie-break: within 1e-9,
  higher mean validation accuracy, then the smaller value.
- The selection code path opens only validation files.

## Architecture comparison
Models A, B and C at the selected (K, L), with five seeds each and the same protocol.
Comparisons are paired by seed (`legalsenn/statistics.py`), with Holm adjustment within a
family of comparisons.

## Input-level faithfulness (`experiments/explain_suite.py`, `legalsenn/explanations.py`)
- **Sample.** A fixed sample of 200 test documents (`RandomState(42)` permutation), shared
  by every model and seed.
- **Rationale space.** Visible sentence units of the fixed input window (`legalsenn/windows.py`).
- **Explanation methods.** Concept-effect occlusion and concept-effect Integrated Gradients
  (concept models only); Integrated Gradients of the logit (32-point Gauss–Legendre, [PAD]
  baseline); KernelSHAP (512 samples) and LIME (512 samples) over sentence presence.
- **Budgets.** 5 %, 10 %, 20 % and 30 % of visible tokens, filled exactly. Controls use the
  same token budget: 10 random unit orders per document and the reversed (bottom) order.
- **Metrics.** Comprehensiveness `p(ŷ|x) − p(ŷ|x∖R)`, sufficiency difference `p(ŷ|x) − p(ŷ|R)`,
  logit drops and prediction flips, with the target fixed to the original prediction.

## Concept masking (`experiments/masking.py`)
Top-1/3/6, bottom-6 and random-6 (100 draws) masks by `|W[ŷ,i] c_i|`, applied before the
linear classifier. For each mask the script records all-class logits before and after,
confidence, margin, flips, absolute and signed effect shares, the own-class / competitor
decomposition of the confidence change, and a softmax-gauge invariance check.
`experiments/masking_hypotheses.py` repeats the masking under alternative ranking rules.

## Fixed-window stability
The visible window is built first. 10 / 20 / 30 % of its sentence units are removed
(3 repeats, deterministic seeds) and the survivors are re-packed. The explanation target stays
fixed to the original prediction. Outputs are concept top-6 Jaccard and top-3 sentence
Jaccard for every explanation method, with exact chance levels (hypergeometric), reported
for all perturbations and separately for unchanged and changed predictions. A legacy mode
(`--parts legacy`) re-runs full-text sentence removal with re-truncation and measures how
much of the perturbed input comes from outside the original window.

## Long documents
`experiments/length_stats.py` computes length statistics. The pipeline also trains Legal-BERT
at 256 and 512 tokens and the hierarchical model H. Published LexGLUE reference values, with
their protocol, are in `experiments/long_context/lexglue_reference.json`.

## Statistics (`legalsenn/statistics.py`)
Seed-paired differences with mean, sample SD, SE, t-based 95 % CI, paired t-test, exact
Wilcoxon signed-rank test and Cohen's d_z; Holm adjustment; percentile bootstrap over
documents; TOST (only with an independently justified margin).
