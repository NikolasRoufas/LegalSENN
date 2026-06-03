# LegalSENN

**Self-Explaining Transformer Models for Interpretable Legal Document Classification**

This repository contains the implementation for the paper:

> Nikolaos Roufas, Matteo Marinacci, Andreas Kanavos, and Massimo Mecella.
> *Self-Explaining Transformer Models for Interpretable Legal Document Classification.*

LegalSENN is a transformer-based legal document classifier designed to produce both predictions and concept-level explanations during the model's own forward computation. The model follows the Self-Explaining Neural Network (SENN) idea: instead of applying a post-hoc explanation method after prediction, it exposes a compact set of learned concepts and uses an additive decision mechanism to show how those concepts contribute to the final class score.

The current implementation targets legal text classification with LexGLUE-style CSV files and includes utilities for training, batch inference, single-document inference, and explanation evaluation.

## Overview

LegalSENN combines:

- a pretrained transformer encoder, such as `nlpaueb/legal-bert-base-uncased`;
- a concept bottleneck layer that maps the document representation into learned concepts;
- a relevance network that estimates concept relevance for the input document;
- an additive classifier that produces logits from concept contributions.

For an input document `x`, the model computes:

```text
z(x)  = transformer document representation
h(x)  = concept activations
a(x)  = concept relevances
c(x)  = h(x) * a(x)
logits = W c(x) + b
```

The vector `c(x)` is used directly by the classifier, so concept-level explanations are tied to the same computation that produces the prediction.

## Paper Motivation

Legal NLP systems often operate on long, dense, high-stakes documents where transparency matters. Post-hoc explainability methods such as attribution maps or surrogate models can be useful, but they explain a trained model from the outside. LegalSENN instead makes explanation part of the architecture.

The paper evaluates this idea on the LexGLUE-SCOTUS benchmark and studies:

- predictive performance on legal issue classification;
- the effect of the concept bottleneck and additive SENN head;
- explanation stability under sentence-level perturbations;
- qualitative concept explanations for Fourth Amendment-style legal reasoning.

In the paper experiments, the proposed SENN model achieved performance close to a Legal-BERT classifier while providing intrinsic concept-level explanations.

## Repository Structure

```text
LegalSENN/
├─ legalsenn/
│  ├─ model.py              # LegalSENN architecture
│  ├─ data.py               # CSV dataset and label map utilities
│  ├─ train.py              # Training loop and evaluation artifacts
│  ├─ inference.py          # Batch inference on CSV files
│  ├─ inference_cli.py      # Single-document inference
│  ├─ explain_eval.py       # Explanation deletion/stability evaluation
│  └─ utils_reporting.py    # Reporting helpers
│
├─ scripts/
│  └─ prepare_lexglue_csv.py # Convert LexGLUE datasets to CSV
│
├─ config/
│  ├─ concept_names.json     # Human-readable concept names
│  ├─ label_names.json       # Alternative label names
│  └─ label_semantics.json   # Human-readable legal labels
│
├─ examples/                 # Example legal texts for inference
├─ data/                     # Generated datasets, ignored by git
├─ outputs/                  # Model checkpoints and reports, ignored by git
├─ requirements.txt
└─ README.MD
```

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:

- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- pandas
- tqdm

## Preparing LexGLUE Data

The project expects CSV files with two columns:

```text
text,label
```

To prepare the SCOTUS subset from LexGLUE:

```bash
python scripts/prepare_lexglue_csv.py \
  --task scotus \
  --output_dir data/lexglue_csv \
  --primary_label_only
```

This creates:

```text
data/lexglue_csv/scotus/train.csv
data/lexglue_csv/scotus/val.csv
data/lexglue_csv/scotus/test.csv
```

For multi-label LexGLUE tasks, the current script keeps the first label as the primary class so the data remains compatible with single-label cross-entropy training.

## Training

Example training command:

```bash
python -m legalsenn.train \
  --train_csv data/lexglue_csv/scotus/train.csv \
  --val_csv data/lexglue_csv/scotus/val.csv \
  --encoder_name nlpaueb/legal-bert-base-uncased \
  --output_dir outputs \
  --run_id legal_senn_scotus \
  --num_epochs 10 \
  --batch_size 16 \
  --lr 2e-5 \
  --max_length 256 \
  --num_concepts 16 \
  --concept_names_path config/concept_names.json \
  --label_semantics_path config/label_semantics.json \
  --save_eval_artifacts
```

Training saves a run directory such as:

```text
outputs/legal_senn_scotus/
├─ best_model.pt
├─ config.json
├─ label_map.json
├─ metrics.jsonl
└─ epoch_*/
```

The best checkpoint is selected by validation macro-F1.

The paper reports experiments with a compact concept space; this implementation exposes the number of concepts through `--num_concepts`, so it can be set to match a specific experiment.

## Single-Document Inference

Run prediction and faithful concept explanation for one text file:

```bash
python -m legalsenn.inference_cli \
  --model_dir outputs/legal_senn_scotus \
  --text_file examples/terry_v_ohio.txt \
  --topk_concepts 6 \
  --label_semantics_path config/label_semantics.json \
  --concept_names_path config/concept_names.json
```

The output includes:

- the predicted legal label;
- prediction confidence;
- top supporting concepts;
- top opposing concepts;
- each concept's additive effect on the predicted class.

## Batch Inference

Run inference on a CSV split:

```bash
python -m legalsenn.inference \
  --csv data/lexglue_csv/scotus/test.csv \
  --encoder_name nlpaueb/legal-bert-base-uncased \
  --ckpt outputs/legal_senn_scotus/best_model.pt \
  --label_map outputs/legal_senn_scotus/label_map.json \
  --out_jsonl outputs/inference_predictions.jsonl \
  --label_semantics_path config/label_semantics.json \
  --concept_names_path config/concept_names.json
```

The output JSONL contains gold labels, predicted labels, confidence scores, top-3 class probabilities, and top concept contributions when concept names are available.

## Explanation Evaluation

The repository includes explanation evaluation utilities for:

- concept deletion tests;
- explanation stability under sentence-level perturbations;
- summary JSON/CSV artifacts for later analysis.

Example:

```bash
python -m legalsenn.explain_eval \
  --model_dir outputs/legal_senn_scotus \
  --split_csv data/lexglue_csv/scotus/test.csv \
  --encoder_name nlpaueb/legal-bert-base-uncased \
  --label_map outputs/legal_senn_scotus/label_map.json \
  --run_stability \
  --run_deletion \
  --max_points 200 \
  --topk 6
```

Artifacts are written by default to:

```text
outputs/legal_senn_scotus/explain_eval/
```

## Notes On Generated Files

The repository is configured so generated data, trained checkpoints, and experiment outputs are not committed by default:

```text
data/lexglue_csv/
outputs/
*.pt
*.pth
*.ckpt
```

This keeps the GitHub repository focused on the implementation. Recreate datasets and checkpoints locally using the commands above.

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
