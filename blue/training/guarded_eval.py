"""Paired evaluation of guarded actors against heuristic baselines.

Why this exists. The LLM-Red evaluation found the coverage guard is the
load-bearing component, and that every checkpoint the project owns was
trained while the sweep was unguarded (`risk_actor.cmd_train` hardcoded
`guard=False`) and against `red_agent="discovery"`, an attacker that cannot
punish a broken investigation policy. This harness scores the retrained
guarded actors under finite Red on one fresh seed block, paired against the
baselines on identical simulator seeds.

Statistical structure, which is the whole point of this file:

  - A *training seed* produces one model. Three training seeds give three
    independently initialised models. That is the unit of inference.
  - The evaluation seeds are SHARED across those models. Averaging 3 models
    x 24 seeds into 72 observations and running a t-test on them would treat
    correlated episodes as independent evidence and manufacture significance.
  - So: per training seed, compute the mean paired difference against Lancer
    over the 24 eval seeds. That gives 3 numbers. The headline interval is a
    t interval on those 3 (df=2, t=4.303). The within-training-seed spread is
    reported separately and labelled as a nuisance, not as evidence.

Every episode of every arm is written out. No seed, model or arm is dropped
after the fact.

Usage:
    .venv-train/bin/python -m blue.training.guarded_eval \
        --models blue/results/guarded_retrain/guard_s0 ... \
        --eval-seeds 6016 ... 6039 --red-agent finite \
        --out blue/results/guarded_retrain/eval.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from blue.training import risk_actor as ra  # noqa: E402
from blue.training import mappo_guide as mg  # noqa: E402

# Two-sided 95% t critical values. df=2 is the headline case (3 training
# seeds). Hardcoded rather than pulled from scipy so the reported interval
# cannot silently change with a library upgrade.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
       15: 2.131, 20: 2.086, 23: 2.069, 30: 2.042}


def t95(df):
    if df in T95:
        return T95[df]
    keys = sorted(T95)
    if df > keys[-1]:
        return 1.96
    return T95[min(k for k in keys if k >= df)]


def summarise(values):
    """Mean, sd and t interval for a list of differences."""
    n = len(values)
    mean = statistics.mean(values)
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "ci95": None, "t": None}
    sd = statistics.stdev(values)
    se = sd / (n ** 0.5)
    crit = t95(n - 1)
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t_crit95": crit,
            "t": (mean / se if se else None),
            "ci95": [mean - crit * se, mean + crit * se]}


def run_ordered(hook_factory, seed, steps, red_agent, guard):
    """One episode of the OrderedPolicy sweep, optionally guarded."""
    from blue.policies.ordered import LancerValues, OrderedPolicy
    pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5), guard=guard)
    pol.reset()
    pol.hook = hook_factory()
    r = mg.run_team_episode(pol, seed, steps, hook=pol.hook,
                            **ra.env_kwargs(red_agent))
    return float(r["return"])


def run_baseline(name, seed, steps, red_agent):
    """Reference policies that are not the ordered sweep."""
    from blue.core import baselines
    if name == "sleep":
        pol = baselines.SleepBaseline()
    elif name == "masked_random":
        pol = baselines.MaskedRandomBaseline(seed=11)
    else:
        raise ValueError("unknown baseline %r" % name)
    r = baselines.run_episode(pol, seed=seed, steps=steps,
                              **ra.env_kwargs(red_agent))
    return float(r["cumulative_return"])


def merge(shard_paths, out_path):
    """Combine per-seed shards into one report.

    Sharding is by evaluation seed, so a shard holds every arm for its own
    seeds and nothing else. The merge is therefore exact: no arm crosses a
    shard boundary, and the paired differences are recomputed from the union
    of per-seed returns rather than averaged across shards.
    """
    seeds, arms, steps, red, abl, models = [], None, None, None, None, None
    for p in shard_paths:
        with open(p) as f:
            r = json.load(f)
        if arms is None:
            arms, steps = set(), r["steps"]
            red, abl, models = r["red_agent"], r["risk_ablation"], r["models"]
        arms |= set(r["arms"])
        for arm, blk in r["arms"].items():
            for s, v in blk["per_seed"].items():
                seeds.append(int(s))
    seeds = sorted(set(seeds))
    cells = {arm: {s: v for s, v in
                   ((int(k), val)
                    for arm_blk in [json.load(open(p))["arms"][arm]
                                    for p in shard_paths
                                    if arm in json.load(open(p))["arms"]]
                    for k, val in arm_blk["per_seed"].items())}
             for arm in sorted(arms)}
    for arm in cells:
        missing = [s for s in seeds if s not in cells[arm]]
        if missing:
            raise SystemExit("arm %s missing seeds %s -- shards disagree"
                             % (arm, missing))
    args = argparse.Namespace(eval_seeds=seeds, red_agent=red,
                              steps=steps, risk_ablation=abl,
                              models=models)
    report = build_report(args, cells, {})
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1, allow_nan=False)
    print("merged %d shards -> %s (%d seeds x %d arms)"
          % (len(shard_paths), out_path, len(seeds), len(arms)))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merge", nargs="+", metavar="SHARD",
                    help="merge shard reports instead of running episodes")
    ap.add_argument("--merge-out", default=None)
    ap.add_argument("--models", nargs="+", help="actor directories, one per training seed")
    ap.add_argument("--eval-seeds", type=int, nargs="+",
                    default=None)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--red-agent", default="finite")
    ap.add_argument("--bonus", type=float, default=1.0)
    ap.add_argument("--risk-ablation", default="real",
                    choices=("real", "zero"))
    ap.add_argument("--baselines", nargs="*",
                    default=["lancer", "masked_random", "sleep"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.merge:
        if not args.merge_out:
            raise SystemExit("--merge requires --merge-out")
        merge(args.merge, args.merge_out)
        return 0
    if not args.models or not args.eval_seeds:
        raise SystemExit("--models and --eval-seeds are required unless "
                         "--merge is used")

    reserved = [s for s in args.eval_seeds if 7809 <= s <= 8200]
    if reserved:
        raise SystemExit("refusing: reserved final seeds %s" % reserved)

    import torch as th
    actors = []
    for d in args.models:
        a = ra.build_actor(hidden=64)
        a.load_state_dict(th.load(os.path.join(d, "actor.th"),
                                  map_location="cpu", weights_only=False))
        a.eval()
        actors.append((d, a))
    risk = ra._risk(args.risk_ablation)

    cells = collections.defaultdict(dict)   # arm -> seed -> return
    detail = collections.defaultdict(dict)  # arm -> seed -> extra counters

    def record(arm, seed, value, extra=None):
        cells[arm][seed] = value
        if extra:
            detail[arm][seed] = extra

    for seed in args.eval_seeds:
        for b in args.baselines:
            t0 = time.time()
            if b == "lancer":
                # The Lancer arm is its own reference and its own cursor
                # supplies the coverage discipline, so it is never guarded.
                v = run_ordered(lambda: None, seed, args.steps,
                                args.red_agent, guard=False)
            else:
                v = run_baseline(b, seed, args.steps, args.red_agent)
            record(b, seed, v)
            print("%-22s %6d %+9.0f  (%.0fs)"
                  % (b, seed, v, time.time() - t0), flush=True)

    for d, actor in actors:
        arm = "actor_" + os.path.basename(d.rstrip("/"))
        for seed in args.eval_seeds:
            t0 = time.time()
            hook = ra.RiskGreedyHook(actor, args.bonus, risk)
            v = run_ordered(lambda: hook, seed, args.steps,
                            args.red_agent, guard=True)
            record(arm, seed, v)
            print("%-22s %6d %+9.0f  (%.0fs)"
                  % (arm, seed, v, time.time() - t0), flush=True)

    report = build_report(args, cells, detail)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1, allow_nan=False)
        print("\nwrote", args.out)
    print(json.dumps(report["headline"], indent=1))


def build_report(args, cells, detail):
    seeds = list(args.eval_seeds)
    arms = list(cells)
    baseline = "lancer" if "lancer" in cells else arms[0]

    per_arm = {}
    for arm in arms:
        vals = [cells[arm][s] for s in seeds]
        per_arm[arm] = {
            "mean": statistics.mean(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else None,
            "n_episodes": len(vals),
            "per_seed": {str(s): cells[arm][s] for s in seeds},
        }

    # Per-training-seed mean paired difference against the baseline.
    actor_arms = [a for a in arms if a.startswith("actor_")]
    per_model = {}
    for arm in actor_arms:
        diffs = [cells[arm][s] - cells[baseline][s] for s in seeds]
        per_model[arm] = {
            "vs": baseline,
            "per_seed_diff": {str(s): cells[arm][s] - cells[baseline][s]
                              for s in seeds},
            "within": summarise(diffs),
            "wins": sum(1 for d in diffs if d > 0),
            "n_episodes": len(diffs),
        }

    # Headline: the training seed is the unit of inference, so the interval
    # is over the per-model means with df = n_models - 1.
    model_means = [per_model[a]["within"]["mean"] for a in actor_arms]
    headline = {
        "unit_of_inference": "training seed (one independently initialised "
                             "model each)",
        "n_training_seeds": len(actor_arms),
        "n_eval_seeds_shared": len(seeds),
        "per_model_mean_diff": {a: per_model[a]["within"]["mean"]
                               for a in actor_arms},
        "across_training_seeds": summarise(model_means),
        "warning": "the 24 evaluation seeds are SHARED across models; "
                   "pooling model x seed cells as independent observations "
                   "would overstate the evidence and is not done here",
    }
    return {
        "red_agent": args.red_agent,
        "steps": args.steps,
        "risk_ablation": args.risk_ablation,
        "eval_seeds": seeds,
        "models": list(args.models),
        "reserved_block_untouched": [7809, 8200],
        "arms": per_arm,
        "paired_vs_baseline": per_model,
        "headline": headline,
        "detail": {a: detail[a] for a in detail},
    }


if __name__ == "__main__":
    sys.exit(main())
