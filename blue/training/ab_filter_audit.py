"""A/B the filter under a controlled initialisation.

Runs the exact 4-iteration pilot recipe for filtered (min_proba=0.5) and
unfiltered (0.0) training across several training seeds, then evaluates each
checkpoint on the 8 dev seeds. Writes one JSON line per run so partial
progress survives an interrupt.

Dev seeds only. The reserved block 7809-8200 is never used.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "blue", "results", "audit_ab")
RESULTS = os.path.join(OUT, "ab_results.jsonl")
PY = os.path.join(ROOT, ".venv-train", "bin", "python")
EVAL_SEEDS = ["7629", "7630", "7640", "7701", "7702", "7703", "7704", "7705"]


def run_one(min_proba, seed):
    tag = "mp%s_s%d" % (min_proba, seed)
    d = os.path.join(OUT, tag)
    ev_json = os.path.join(d, "eval.json")
    os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "actor.th")) and \
            os.path.exists(ev_json):
        print("skip (done)", tag, flush=True)
        return
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    tr = subprocess.run(
        [PY, "-m", "blue.training.risk_actor", "--mode", "train",
         "--out", d, "--iters", "4", "--eps-per-iter", "4",
         "--steps", "400", "--temp", "0.5",
         "--train-min-proba", str(min_proba), "--seed", str(seed)],
        cwd=ROOT, env=env, capture_output=True, text=True)
    if tr.returncode != 0:
        print("TRAIN FAIL", tag, tr.stderr[-800:], flush=True)
        return
    ev = subprocess.run(
        [PY, "-m", "blue.training.risk_actor", "--mode", "eval",
         "--model-dir", d, "--out", ev_json, "--steps", "400",
         "--gate-thresholds", "0.5"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    if ev.returncode != 0:
        print("EVAL FAIL", tag, ev.stderr[-800:], flush=True)
        return
    with open(ev_json) as f:
        cells = json.load(f)
    man_path = os.path.join(d, "pilot_manifest.json")
    hist = []
    if os.path.exists(man_path):
        with open(man_path) as f:
            hist = json.load(f).get("hist", [])
    row = {"tag": tag, "train_min_proba": min_proba, "train_seed": seed,
           "cells": cells["cells"], "summary": cells["summary"],
           "train_hist": hist, "elapsed_s": round(time.time() - t0, 1)}
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")
    g = cells["summary"]["greedy_minus_lancer"]
    c = cells["summary"]["conf0.5_minus_lancer"]
    print("%s greedy %+.2f CI[%+.1f,%+.1f] conf %.2f (%.0f min)"
          % (tag, g["mean"], g["ci95"][0], g["ci95"][1], c["mean"],
             (time.time() - t0) / 60.0), flush=True)


if __name__ == "__main__":
    jobs = []
    for mp in (0.5, 0.0):
        for sd in (0, 1, 2):
            jobs.append((mp, sd))
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        run_one(float(sys.argv[2]), int(sys.argv[3]))
    else:
        shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        for i, (mp, sd) in enumerate(jobs):
            if i % n == shard:
                run_one(mp, sd)