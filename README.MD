# LegalSENN

**Self-explaining (SENN-inspired) transformer models for legal document classification**

This repository contains the implementation for the paper:

> Nikolaos Roufas, Matteo Marinacci, Andreas Kanavos, and Massimo Mecella.
> *Self-Explaining Transformer Architectures for Legal Document Classification.*

It provides the LegalSENN model, baseline and ablation architectures, a deterministic
training runner, and a reproducible experimental pipeline. The pipeline covers
validation-based hyperparameter selection, multi-seed architecture comparisons, input-level
explanation faithfulness, concept masking, fixed-window explanation stability, post-hoc
explanation baselines (Integrated Gradients, KernelSHAP, LIME), long-document references and
paired statistical analysis.

## Architecture

```text
z(x)   = Legal-BERT [CLS] representation (dropout 0.1)
h(x)   = W_c z(x) + b_c                        K concept activations
a(x)   = W_2 ReLU(W_1 z(x) + b_1) + b_2        K relevance scores
c(x)   = h(x) ⊙ a(x)                           concept contributions
logits = W c(x) + b
effect_i(x, y) = W[y, i] c_i(x)                additive decomposition of logit_y
```

The model is trained with cross-entropy. It uses the additive concept × relevance form
of Self-Explaining Neural Networks (Alvarez-Melis & Jaakkola, 2018) without SENN's
robustness or concept-reconstruction regularisers, i.e. it is a SENN-inspired architecture.

`legalsenn/model.py` provides:

| Class | `--arch` | Description |
|---|---|---|
| `LegalBertClassifier` | `bert` | Legal-BERT with a linear head |
| `ConceptLayerClassifier` | `concept` | Legal-BERT + concept layer, no relevance network |
| `LegalSENN` | `senn` | full SENN-inspired model |
| `HierarchicalLegalBert` | `hier` | long-document reference (up to 64 × 128 tokens) |

`load_from_run(run_dir)` rebuilds the exact architecture, including the concept dimension,
from a run's `config.json`.

## Dataset

LexGLUE-SCOTUS (`coastalcph/lex_glue`, `scotus`): 13 US Supreme Court issue-area classes with
the official chronological train / validation / test splits. Dataset and encoder revisions are
pinned in `legalsenn/scotus.py`. Issue-area names are in `config/scotus_issue_areas.json`.

## Installation

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130   # choose the build for your CUDA
pip install -r requirements-experiments.txt
python -m pytest -q tests/
```

`requirements.txt` keeps the minimal dependency list of the original release.

## Training

```bash
export HF_HOME=/path/to/hf_cache PYTHONPATH=$PWD
python -m legalsenn.run_train --arch senn --num_concepts 32 --max_length 256 --seed 13 \
    --run_dir runs/senn_K32_L256_s13 --cache_dir cache
```

- **Defaults:** AdamW at lr 2e-5, batch 16, at most 10 epochs, early stopping (patience 3) on
  validation Macro-F1, and the best validation checkpoint retained.
- **Test evaluation:** runs once, after training, and is written separately from the
  validation results.
- **Determinism:** seeded runs are deterministic on a given GPU and software stack.

The original CSV-based entry points are still available:

```bash
python scripts/prepare_lexglue_csv.py --task scotus --output_dir data/lexglue_csv --primary_label_only
python -m legalsenn.train --train_csv data/lexglue_csv/scotus/train.csv --val_csv data/lexglue_csv/scotus/val.csv \
    --encoder_name nlpaueb/legal-bert-base-uncased --num_epochs 10 --batch_size 16 --lr 2e-5 \
    --max_length 256 --num_concepts 32 --output_dir outputs --run_id legal_senn_scotus
```

## Inference

```bash
python -m legalsenn.inference_cli --model_dir outputs/legal_senn_scotus \
    --text_file examples/terry_v_ohio.txt --topk_concepts 6 \
    --label_semantics_path config/scotus_issue_areas.json --concept_names_path config/concept_names.json
```

The original inference tools infer K from the concept-name file. For other concept
dimensions, load models with `legalsenn.model.load_from_run`.

## Experimental pipeline

```bash
export HF_HOME=/path/to/hf_cache PYTHONPATH=$PWD LEGALSENN_COMMIT=$(git rev-parse HEAD)
python -m experiments.length_stats --cache_dir ROOT/cache --out ROOT/results/long_context
python -m experiments.pipeline --root ROOT --parallel 2      # resumable
python -m experiments.aggregate --root ROOT --out ROOT/summary
```

| Stage | Code |
|---|---|
| Hyperparameter selection (K, then L) on validation Macro-F1 only | `experiments/select_hparams.py` |
| Five-seed comparison of `bert`, `concept`, `senn` | `experiments/pipeline.py`, `experiments/make_jobs.py` |
| Concept masking (top/bottom/random sets, all-class logits, gauge check) | `experiments/masking.py`, `experiments/masking_hypotheses.py` |
| Input-level faithfulness (comprehensiveness, sufficiency, random/bottom controls, flips) | `experiments/explain_suite.py`, `legalsenn/explanations.py` |
| Fixed-window sentence-removal stability; legacy protocol for comparison | `legalsenn/windows.py`, `experiments/explain_suite.py` |
| Integrated Gradients, KernelSHAP, LIME in a shared sentence-level space | `legalsenn/explanations.py` |
| Concept concentration (Top-6 absolute effect share) | `experiments/concentration.py` |
| Document-length analysis, Legal-BERT 512, hierarchical Legal-BERT | `experiments/length_stats.py`, `legalsenn/model.py` |
| Paired statistics (t-CI, paired t, exact Wilcoxon, d_z, Holm, bootstrap) | `legalsenn/statistics.py` |
| Result tables and figures from stored outputs | `experiments/aggregate.py` |

See [docs/EXPERIMENTAL_PROTOCOL.md](docs/EXPERIMENTAL_PROTOCOL.md) and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Tests

`tests/test_pipeline.py` checks:
- exact logit reconstruction and effect computation;
- model dimensions, gradient updates and dropout behaviour;
- checkpoint reconstruction;
- ranking and masking logic;
- Jaccard and its exact chance level;
- fixed-window perturbation invariants and exact rationale budgets;
- comprehensiveness and sufficiency;
- paired statistics, seed matching and validation-only selection.

```bash
python -m pytest -q tests/
```

## Repository structure

```text
legalsenn/      models, data, training, fixed-window utilities, explanation engine, statistics
experiments/    pipeline, selection, masking, explanation suite, long-document analysis, aggregation
tests/          automated checks
config/         label and concept-name files
scripts/        LexGLUE CSV preparation
docs/           experimental protocol and reproduction guide
examples/       example texts for inference
```

Checkpoints and datasets are not stored in the repository.

## Citation

If you use this repository, please cite the associated manuscript:

```bibtex
@article{roufas2026selfexplaining,
  title  = {Self-Explaining Transformer Models for Interpretable Legal Document Classification},
  author = {Roufas, Nikolaos and Marinacci, Matteo and Kanavos, Andreas and Mecella, Massimo},
  year   = {2026},
  note   = {Manuscript}
}
```

## Ownership And License

This codebase is the property of Nikolaos Roufas. Redistribution, modification, or commercial use is not permitted without prior written permission.
