"""
Generate job lists for each training phase.

Run directories are keyed by configuration only ({arch}_K{K}_L{L}_s{seed}), so a run
needed by several phases (e.g. SENN K=32, L=256 in the K sweep, the L sweep and the
ablation) is trained once and reused with identical provenance.
"""
import argparse
import json
import os

SEEDS = [13, 21, 42, 87, 100]


def run_id(arch, k, L, seed):
    if arch in ("senn", "concept"):
        return f"{arch}_K{k}_L{L}_s{seed}"
    return f"{arch}_L{L}_s{seed}"


def job(arch, k, L, seed, root, cache, experiment, extra=()):
    rid = run_id(arch, k, L, seed)
    rd = os.path.join(root, "runs", rid)
    args = ["--arch", arch, "--seed", str(seed), "--run_dir", rd, "--cache_dir", cache, "--experiment", experiment]
    if arch in ("senn", "concept"):
        args += ["--num_concepts", str(k)]
    if arch != "hier":
        args += ["--max_length", str(L)]
    args += list(extra)
    return {"id": rid, "module": "legalsenn.run_train", "args": args, "run_dir": rd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["k_sweep", "l_sweep", "ablation", "bert512", "hier"])
    ap.add_argument("--root", default="workdir")
    ap.add_argument("--K", type=int, help="validation-selected K")
    ap.add_argument("--L", type=int, help="validation-selected L")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cache = os.path.join(a.root, "cache")
    jobs = []
    if a.phase == "k_sweep":
        jobs = [job("senn", k, 256, s, a.root, cache, "1.1_k_sweep") for k in (8, 16, 32, 64) for s in SEEDS]
    elif a.phase == "l_sweep":
        jobs = [job("senn", a.K, L, s, a.root, cache, "1.2_l_sweep") for L in (128, 256, 512) for s in SEEDS]
    elif a.phase == "ablation":
        jobs = [job(arch, a.K, a.L, s, a.root, cache, "2_ablation") for arch in ("bert", "concept", "senn") for s in SEEDS]
    elif a.phase == "bert512":
        jobs = [job("bert", None, L, s, a.root, cache, "7.2_bert_length") for L in (256, 512) for s in SEEDS]
    elif a.phase == "hier":
        jobs = [job("hier", None, 8192, s, a.root, cache, "7.3_hierarchical",
                    extra=["--micro_batch", "2", "--grad_ckpt", "--eval_batch", "4"]) for s in SEEDS]
    json.dump(jobs, open(a.out, "w"), indent=1)
    print(f"{len(jobs)} jobs -> {a.out}")


if __name__ == "__main__":
    main()
