"""
Resumable job queue for the experimental pipeline.

Each job is one `python -m legalsenn.run_train ...` (or any module) invocation with a
unique run id. Jobs whose run directory contains DONE are skipped. Every attempt is
appended to <log_dir>/jobs.jsonl with start/end time, exit status, commit and log path.

Usage:
  python experiments/run_queue.py --jobs jobs.json --parallel 2 --log_dir LOGS
jobs.json: [{"id": "senn_K32_L256_s13", "module": "legalsenn.run_train",
             "args": [...], "run_dir": "..."}]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def run_job(job, log_dir, retries):
    if job.get("run_dir") and os.path.exists(os.path.join(job["run_dir"], "DONE")):
        return job["id"], "skipped", 0
    log_path = os.path.join(log_dir, f"{job['id']}.log")
    for attempt in range(1, retries + 2):
        start = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        t0 = time.time()
        with open(log_path, "a") as lf:
            lf.write(f"\n===== attempt {attempt} start {start} =====\n")
            lf.flush()
            p = subprocess.run([sys.executable, "-m", job["module"], *job["args"]], stdout=lf, stderr=subprocess.STDOUT)
        rec = {"id": job["id"], "attempt": attempt, "start": start,
               "end": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "wall_s": round(time.time() - t0, 1),
               "exit_status": p.returncode, "commit": os.environ.get("LEGALSENN_COMMIT", "unknown"),
               "module": job["module"], "args": job["args"], "log": log_path}
        with open(os.path.join(log_dir, "jobs.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        if p.returncode == 0:
            return job["id"], "ok", attempt
    return job["id"], "failed", attempt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--log_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.log_dir, exist_ok=True)
    jobs = json.load(open(a.jobs))
    with ThreadPoolExecutor(a.parallel) as ex:
        for jid, status, attempts in ex.map(lambda j: run_job(j, a.log_dir, a.retries), jobs):
            print(f"{time.strftime('%H:%M:%S')} {jid}: {status} (attempts={attempts})", flush=True)


if __name__ == "__main__":
    main()
