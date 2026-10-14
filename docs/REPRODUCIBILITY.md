# Reproducing the experiments

## Environment
- Python 3.12, PyTorch 2.12.1 (CUDA 13.0 build), transformers 5.12.1, datasets 5.1.0,
  captum 0.9.0, shap 0.53.0, lime 0.2.0.1, scikit-learn, scipy, pandas. Exact versions are
  in `requirements-experiments.txt`.
- One CUDA GPU with ≥ 16 GB is sufficient for flat models. The hierarchical model uses
  about 8 GB with micro-batch 2 and gradient checkpointing.
- Hugging Face dataset and encoder revisions are pinned in `legalsenn/scotus.py`.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-experiments.txt
python -m pytest -q tests/
```

## Full pipeline (resumable)
```bash
export HF_HOME=/path/to/hf_cache PYTHONPATH=$PWD
export LEGALSENN_COMMIT=$(git rev-parse HEAD)
# optional: refuse to run unless CUDA device 0 is on the given PCI bus
# export LEGALSENN_GPU_BUS=2
python -m experiments.length_stats --cache_dir ROOT/cache --out ROOT/results/long_context
python -m experiments.pipeline --root ROOT --parallel 2
python -m experiments.aggregate --root ROOT --out ROOT/summary
```

`experiments/pipeline.py` runs the phases in order:
1. the K sweep, followed by a validation-only choice of K;
2. the L sweep, followed by a validation-only choice of L;
3. the A/B/C comparison;
4. Legal-BERT at 256 and 512 tokens;
5. the hierarchical model;
6. concept masking, input-level faithfulness, fixed-window stability and the post-hoc comparisons.

Each job appends its start and end time, exit status, commit, arguments and log path to
`ROOT/logs/jobs.jsonl`. Runs with a `DONE` marker are skipped, so an interrupted pipeline can
be restarted.

## Single components
```bash
# one training run (arch: bert | concept | senn | hier; hier also: --micro_batch 2 --grad_ckpt --eval_batch 4)
python -m legalsenn.run_train --arch senn --num_concepts 32 --max_length 256 --seed 13 \
    --run_dir ROOT/runs/senn_K32_L256_s13 --cache_dir ROOT/cache

# concept-masking audit
python -m experiments.masking --run_dir ROOT/runs/senn_K32_L256_s13 --out ROOT/results/masking/senn_K32_L256_s13 --cache_dir ROOT/cache

# faithfulness + fixed-window stability + IG / KernelSHAP / LIME on the fixed 200-document sample
python -m experiments.explain_suite --run_dir ROOT/runs/senn_K32_L256_s13 --out ROOT/results/explain/senn_K32_L256_s13 --cache_dir ROOT/cache

# concept-only stability, or the legacy (re-truncating) stability protocol
python -m experiments.explain_suite --run_dir RUN --out OUT --cache_dir ROOT/cache --methods none --parts stab
python -m experiments.explain_suite --run_dir RUN --out OUT --cache_dir ROOT/cache --sample_seed 123 --parts legacy
```

## Run directory contents
- `config.json`: architecture, hyperparameters, dataset and encoder revisions, split
  fingerprints, commit, library versions, GPU.
- `metrics.jsonl`: per-epoch metrics.
- `val_summary.json`: selected epoch and validation metrics, written before test is read.
- `test_summary.json`, `val_logits.npz`, `test_logits.npz`, `best_model.pt`, `DONE`.

Checkpoints and datasets are not stored in this repository.
