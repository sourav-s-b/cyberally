"""G1 frozen-guide re-measure on a fresh dev window (OP2 prereg gate).

Evaluates scripted arms (guard / lancer / sleep / random) over an explicit
seed list under the frozen wrapper, with native rewards only — no shaping,
no learner, hooks stay None, no privileged reads beyond the native reward
channel. The guard arm is the SAME construction the OP2 JSRL guide uses
(OrderedPolicy over a LancerValues scorer with a max-age coverage guard),
so guide and control are literally one frozen policy family.

Usage (train venv, repo root):
  .venv-train/bin/python -m blue.training.harness_guide_remeasure \
      --seeds 8445 8446 ... --steps 400 --out blue/results/op2_jsrl_guide_v1/guard_remeasure

Writes report.json: config, git provenance, per-seed per-arm native returns,
mean and paired contrasts vs guard.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode()
    except Exception:
        return "unknown"


def env_info_hash(env):
    return hashlib.sha256(json.dumps(
        env.get_env_info(), sort_keys=True).encode()).hexdigest()


def build_arm(name, max_age):
    from blue.core.baselines import MaskedRandomBaseline, SleepBaseline
    from blue.policies.ordered import LancerValues, OrderedPolicy
    if name == "guard":
        return OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                             max_age=max_age)
    if name == "lancer":
        return OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5))
    if name == "sleep":
        return SleepBaseline()
    if name == "random":
        return MaskedRandomBaseline(seed=0)
    raise ValueError(f"unknown arm {name}")


def main():
    from blue.training.mappo_guide import run_team_episode, ENV_KW
    ap = argparse.ArgumentParser(description="guide re-measure (G1)")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--arms", nargs="+",
                    default=["guard", "lancer", "sleep", "random"])
    ap.add_argument("--max-age", type=float, default=80.0)
    ap.add_argument("--red-agent", default="finite",
                    choices=("discovery", "finite"))
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()

    from blue.core.wrapper import CC4MARLEnv
    env_kw = dict(ENV_KW)
    env_kw["red_agent"] = cli.red_agent
    probe = CC4MARLEnv(seed=cli.seeds[0], steps=cli.steps, **env_kw)
    probe.reset(seed=cli.seeds[0])
    info_sha = env_info_hash(probe)
    probe.close()

    # Random arm seeded per episode for order-independence, same as
    # mappo_guide.cmd_eval.
    from blue.core.baselines import MaskedRandomBaseline
    results = {}
    for arm in cli.arms:
        results[arm] = {}
        for seed in cli.seeds:
            policy = MaskedRandomBaseline(seed=int(seed)) \
                if arm == "random" else build_arm(arm, cli.max_age)
            r = run_team_episode(policy, int(seed), cli.steps, **env_kw)
            results[arm][str(seed)] = {"return": r["return"],
                                       "coverage": r["coverage"],
                                       "ticks": r["ticks"],
                                       "n_guide": r["n_guide"]}
            print(f"{arm:8s} seed={seed} return={r['return']:+7.1f} "
                  f"cov={r['coverage']:.2f}", flush=True)

    guard = [results["guard"][str(s)]["return"] for s in cli.seeds]
    contrasts = {}
    for other in cli.arms:
        if other == "guard":
            continue
        d = [results[other][str(s)]["return"] - results["guard"][str(s)]["return"]
             for s in cli.seeds]
        contrasts[f"{other}_minus_guard"] = {
            "paired_mean": sum(d) / len(d), "n": len(d)}
    report = {
        "purpose": "OP2 prereg G1: frozen-guide re-measure on fresh dev "
                   "window; native returns only; no performance claim",
        "git_commit": git_commit(),
        "env_kw": {k: list(v) if isinstance(v, tuple) else v
                   for k, v in env_kw.items()},
        "env_info_sha256": info_sha,
        "config": {"seeds": cli.seeds, "steps": cli.steps,
                   "arms": cli.arms, "max_age": cli.max_age,
                   "red_agent": cli.red_agent},
        "cells": results,
        "means": {a: sum(v["return"] for v in results[a].values())
                  / len(results[a]) for a in cli.arms},
        "contrasts": contrasts,
    }
    os.makedirs(cli.out, exist_ok=True)
    path = os.path.join(cli.out, "report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    for k, v in report["means"].items():
        print(f"mean {k}: {v:.2f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
