"""
End-to-end driver for the experimental pipeline, in its scientific order:

  1.1 K sweep (SENN, L=256)          -> validation-only selection of K
  1.2 L sweep (SENN, K*)             -> validation-only selection of L
  2   ablation (BERT / Concept / SENN at K*, L*)
  7.2 Legal-BERT at 256 and 512 tokens
  then, concurrently:
  7.3 hierarchical Legal-BERT (5 seeds)
  3-6 concentration, masking and explanation evaluations

Selections are made by experiments/select_hparams.py from validation metrics only.
Every stage is resumable (completed runs are skipped).
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

from experiments.make_jobs import SEEDS, job, run_id
from experiments.run_queue import run_job

PY = sys.executable


def log(msg, root):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}"
    print(line, flush=True)
    with open(os.path.join(root, "logs", "pipeline.log"), "a") as f:
        f.write(line + "\n")


def run_jobs(jobs, parallel, root, name):
    from concurrent.futures import ThreadPoolExecutor
    log_dir = os.path.join(root, "logs")
    json.dump(jobs, open(os.path.join(root, "jobs", f"{name}.json"), "w"), indent=1)
    log(f"stage {name}: {len(jobs)} jobs, parallel={parallel}", root)
    with ThreadPoolExecutor(parallel) as ex:
        res = list(ex.map(lambda j: run_job(j, log_dir, 1), jobs))
    failed = [r for r in res if r[1] == "failed"]
    log(f"stage {name}: done; failed={[f[0] for f in failed]}", root)
    return failed


def select(root, factor, fixed, values):
    out = subprocess.run([PY, "experiments/select_hparams.py", "select", "--root", root,
                          "--out_dir", os.path.join(root, "results", "selection"), "--factor", factor,
                          "--fixed", str(fixed), "--values", ",".join(map(str, values))],
                         capture_output=True, text=True)
    log(out.stdout.strip().replace("\n", " | "), root)
    if out.returncode != 0:
        raise SystemExit(out.stderr)
    return json.load(open(os.path.join(root, "results", "selection", f"selection_{factor}.json")))["selected"]


def eval_job(module, args, rid, done_path):
    return {"id": rid, "module": module, "args": args, "run_dir": done_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="workdir")
    ap.add_argument("--parallel", type=int, default=2)
    a = ap.parse_args()
    root, cache = a.root, os.path.join(a.root, "cache")
    runs = os.path.join(root, "runs")
    res = os.path.join(root, "results")
    os.makedirs(os.path.join(root, "jobs"), exist_ok=True)

    # wait for any separately launched queue (the K sweep was started before this driver existed)
    while subprocess.run(["pgrep", "-f", "experiments/run_queue.py"], capture_output=True).returncode == 0:
        time.sleep(60)

    Ks, Ls = (8, 16, 32, 64), (128, 256, 512)
    run_jobs([job("senn", k, 256, s, root, cache, "1.1_k_sweep") for k in Ks for s in SEEDS], a.parallel, root, "k_sweep")
    K = select(root, "K", 256, Ks)
    log(f"validation-selected K = {K}", root)

    run_jobs([job("senn", K, L, s, root, cache, "1.2_l_sweep") for L in Ls for s in SEEDS], a.parallel, root, "l_sweep")
    L = select(root, "L", K, Ls)
    log(f"validation-selected L = {L}", root)

    run_jobs([job(arch, K, L, s, root, cache, "2_ablation") for arch in ("bert", "concept", "senn") for s in SEEDS],
             a.parallel, root, "ablation")
    run_jobs([job("bert", None, Lb, s, root, cache, "7.2_bert_length") for Lb in (256, 512) for s in SEEDS],
             a.parallel, root, "bert_length")

    hier = [job("hier", None, 8192, s, root, cache, "7.3_hierarchical",
                extra=["--micro_batch", "2", "--grad_ckpt", "--eval_batch", "4"]) for s in SEEDS]

    concept_runs = sorted({os.path.join(runs, d) for d in os.listdir(runs)
                           if d.startswith(("senn_", "concept_")) and os.path.exists(os.path.join(runs, d, "DONE"))})
    ev = [eval_job("experiments.concentration", ["--run_dirs", *concept_runs, "--cache_dir", cache],
                   "concentration", None)]
    mask_cfgs = [("senn", K, L), ("concept", K, L)]
    if (K, L) != (32, 256):
        mask_cfgs.append(("senn", 32, 256))  # reference configuration K=32, L=256
    for arch, k, l in mask_cfgs:
        for s in SEEDS:
            rid = run_id(arch, k, l, s)
            out = os.path.join(res, "masking", rid)
            ev.append(eval_job("experiments.masking", ["--run_dir", os.path.join(runs, rid), "--out", out,
                                                      "--cache_dir", cache], f"mask_{rid}", out))
    for s in SEEDS:  # interleave so every seed of every model progresses together
        for arch in ("senn", "concept", "bert"):
            rid = run_id(arch, K, L, s)
            out = os.path.join(res, "explain", rid)
            ev.append(eval_job("experiments.explain_suite", ["--run_dir", os.path.join(runs, rid), "--out", out,
                                                            "--cache_dir", cache], f"explain_{rid}", out))
    for s in SEEDS:  # released stability protocol on its original 200-document sample (seed 123)
        rid = run_id("senn", K, L, s)
        out = os.path.join(res, "legacy123", rid)
        ev.append(eval_job("experiments.explain_suite", ["--run_dir", os.path.join(runs, rid), "--out", out,
                                                        "--cache_dir", cache, "--sample_seed", "123",
                                                        "--parts", "legacy"], f"legacy123_{rid}", out))

    t = threading.Thread(target=run_jobs, args=(hier, 1, root, "hierarchical"))
    t.start()
    run_jobs(ev, 1, root, "evaluations")
    t.join()
    log("PIPELINE COMPLETE", root)


if __name__ == "__main__":
    main()
