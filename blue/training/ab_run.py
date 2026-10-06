"""Run the risk-input ablation and analyse it correctly.

Two questions, kept separate:

  Q1  RL+risk vs Lancer        does this RL agent improve defense at all?
  Q2  RL+risk vs RL+norisk     does the risk input contribute?

Statistician's trap this file exists to avoid: 3 training seeds x 32 eval
seeds is NOT 96 independent observations. The evaluation seeds are the SAME
32 for every training seed, so they are a blocked nuisance factor, not extra
replication. Treating them as independent would shrink the interval by
roughly sqrt(32) and manufacture significance out of nothing.

So:
  - the independent unit is the TRAINING seed (n = 3);
  - each training seed contributes ONE number: the mean paired difference
    over the 32 eval seeds;
  - the headline CI is across those 3 numbers -> t(df=2) = 4.303;
  - per-training-seed eval-seed CIs are reported as a within-unit spread,
    clearly labelled, never as the headline.

Every run is reported. Nothing is selected.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from blue.training import ab_freeze_config as fz  # noqa: E402

OUT = fz.OUT
PY = os.path.join(REPO, ".venv-train", "bin", "python")
RESULTS = os.path.join(OUT, "runs.jsonl")
T_DF2 = 4.303          # t(0.975, df=2) for the across-training-seed CI


def tag(arm, seed):
    return "%s_s%d" % (arm, seed)


def train_and_eval(arm, seed, steps):
    d = os.path.join(OUT, tag(arm, seed))
    ev = os.path.join(d, "eval.json")
    os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "actor.th")) and os.path.exists(ev):
        print("skip (done)", tag(arm, seed), flush=True)
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    common = ["--iters", str(fz.BUDGET["iters"]),
              "--eps-per-iter", str(fz.BUDGET["eps_per_iter"]),
              "--steps", str(steps),
              "--hidden", str(fz.BUDGET["hidden"]),
              "--lr", str(fz.BUDGET["lr"]),
              "--bonus", str(fz.BUDGET["bonus"]),
              "--temp", str(fz.BUDGET["temp"]),
              "--train-min-proba", str(fz.BUDGET["train_min_proba"]),
              "--seed", str(seed),
              "--risk-ablation", arm,
              "--train-seeds", *[str(s) for s in fz.TRAIN_EPISODE_SEEDS]]
    t0 = time.time()
    tr = subprocess.run([PY, "-m", "blue.training.risk_actor",
                         "--mode", "train", "--out", d] + common,
                        cwd=REPO, env=env, capture_output=True, text=True)
    if tr.returncode != 0:
        print("TRAIN FAIL", tag(arm, seed), tr.stderr[-600:], flush=True)
        return
    # Evaluation uses the SAME ablation as training. Only the Lancer arm is
    # passed, since the confidence-gate arm is meaningless at zero risk.
    evc = subprocess.run(
        [PY, "-m", "blue.training.risk_actor", "--mode", "eval",
         "--model-dir", d, "--out", ev, "--steps", str(steps),
         "--risk-ablation", arm,
         "--eval-seeds", *[str(s) for s in fz.EVAL_SEEDS]]
        + (["--gate-thresholds", "0.5"] if arm == "real" else []),
        cwd=REPO, env=env, capture_output=True, text=True)
    if evc.returncode != 0:
        print("EVAL FAIL", tag(arm, seed), evc.stderr[-600:], flush=True)
        return
    with open(ev) as f:
        res = json.load(f)
    with open(os.path.join(d, "pilot_manifest.json")) as f:
        man = json.load(f)
    row = {"arm": arm, "train_seed": seed, "steps": steps,
           "cells": res["cells"], "summary": res["summary"],
           "risk_ablation": res.get("risk_ablation"),
           "train_hist": man.get("hist", []),
           "elapsed_s": round(time.time() - t0, 1)}
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")
    g = res["summary"]["greedy_minus_lancer"]
    print("%s  %+7.2f  (%.0f min)" % (tag(arm, seed), g["mean"],
                                      (time.time() - t0) / 60.0), flush=True)


def paired_diffs(row, keys):
    """Per-eval-seed paired differences (greedy - lancer)."""
    c = row["cells"]
    g, l = c["greedy"], c["lancer"]
    return [g[k] - l[k] for k in sorted(g, key=int)]


def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def analyse():
    rows = []
    with open(RESULTS) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    by = {(r["arm"], r["train_seed"]): r for r in rows}

    # one number per training seed: the mean paired difference over eval seeds
    per_seed = {}
    for r in rows:
        d = paired_diffs(r, None)
        per_seed.setdefault(r["arm"], {})[r["train_seed"]] = {
            "mean_diff": mean(d),
            "sd_across_eval_seeds": stdev(d),
            "n_eval": len(d),
            "wins": sum(1 for x in d if x > 0),
        }

    def across(arm):
        vals = [per_seed[arm][s]["mean_diff"] for s in sorted(per_seed[arm])]
        if len(vals) < 2:
            return None
        m, sd = mean(vals), stdev(vals)
        se = sd / (len(vals) ** 0.5)
        return {"per_train_seed": vals, "n_training_seeds": len(vals),
                "mean": m, "sd": sd,
                "ci95": [m - T_DF2 * se, m + T_DF2 * se],
                "t_crit": T_DF2, "df": len(vals) - 1}

    q1 = across("real")       # RL+risk vs Lancer
    q2 = None
    if "real" in per_seed and "zero" in per_seed:
        common = sorted(set(per_seed["real"]) & set(per_seed["zero"]))
        if len(common) >= 2:
            diffs = [per_seed["real"][s]["mean_diff"]
                     - per_seed["zero"][s]["mean_diff"] for s in common]
            m, sd = mean(diffs), stdev(diffs)
            se = sd / (len(diffs) ** 0.5)
            q2 = {"matched_pairs": {str(s): round(
                per_seed["real"][s]["mean_diff"]
                - per_seed["zero"][s]["mean_diff"], 3) for s in common},
                "per_pair_diff": diffs, "n_pairs": len(diffs),
                "mean": m, "sd": sd,
                "ci95": [m - T_DF2 * se, m + T_DF2 * se],
                "t_crit": T_DF2, "df": len(diffs) - 1}

    return {"q1_rl_risk_vs_lancer": q1,
            "q2_rl_risk_vs_rl_norisk": q2,
            "per_training_seed": per_seed,
            "every_run_reported": True,
            "n_runs": len(rows),
            "independence_note": "unit of replication is the TRAINING seed "
                                 "(n=%d). The %d eval seeds are shared "
                                 "across training seeds and are a blocked "
                                 "factor, not independent observations."
                                 % (len(fz.TRAIN_SEEDS), len(fz.EVAL_SEEDS))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=3)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()

    if args.analyse:
        print(json.dumps(analyse(), indent=1))
        return 0

    steps = args.steps if args.steps is not None else fz.BUDGET["steps"]
    jobs = [(arm, s) for s in fz.TRAIN_SEEDS for arm in fz.ARMS]
    for i, (arm, s) in enumerate(jobs):
        if i % args.nshards == args.shard:
            train_and_eval(arm, s, steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())