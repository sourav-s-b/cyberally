"""Instrument one episode to find the catastrophic-return mechanism.

The ablation's per-seed paired diffs are bimodal: median around +16, but
individual seeds reach -1209 while Lancer scores -64 on the same seed. A
mean that looks like "the agent is bad" can instead be one or two episodes
destroying an otherwise-fine policy. This module replays a named seed and
records, per tick, what the actor did that the Lancer baseline did not.

Read-only with respect to the simulator: it uses the same
`run_team_episode` path and only observes.

Usage:
    .venv-train/bin/python -m blue.training.collapse_probe \
        --model-dir blue/results/ablation_risk/real_s0 --seed 8241
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from blue.training import mappo_guide as mg  # noqa: E402
from blue.training import risk_actor as ra  # noqa: E402


def _decode(env, agent, act):
    from blue.core.baselines import decode_index
    return decode_index(env, agent, act)


def trace(model_dir, seed, steps, ablation, bonus, limit_ticks=None):
    import torch as th
    from blue.policies.ordered import LancerValues, OrderedPolicy
    from blue.core.wrapper import BLUE_AGENTS

    actor = ra.build_actor(hidden=64)
    actor.load_state_dict(th.load(os.path.join(model_dir, "actor.th"),
                                  map_location="cpu", weights_only=False))
    actor.eval()
    risk = ra._risk(ablation)

    env_kw = dict(mg.ENV_KW)
    out = {}
    for arm in ("lancer", "actor"):
        from blue.core.wrapper import CC4MARLEnv
        env = CC4MARLEnv(seed=seed, steps=steps, **env_kw)
        env.reset(seed=seed)
        pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                            guard=False)
        hook = None
        if arm == "actor":
            hook = ra.RiskGreedyHook(actor, bonus, risk)
        pol.reset()
        pol.hook = hook
        rewards, diffs = [], collections.Counter()
        actions_by_name = collections.Counter()
        per_tick = []
        prev = None
        for tick in range(1, steps + 1):
            acts = {}
            for a in BLUE_AGENTS:
                acts[a] = int(pol.select(env, a))
            _, rew, term, trunc, _ = env.step(acts)
            rewards.append(float(rew[0]))
            names = {a: _decode(env, a, acts[a])[0] for a in BLUE_AGENTS}
            for a in BLUE_AGENTS:
                actions_by_name[names[a]] += 1
            per_tick.append({"tick": tick, "reward": float(rew[0]),
                             "actions": {a: list(names[a]) for a in BLUE_AGENTS}})
            if prev is not None:
                for a in BLUE_AGENTS:
                    if names[a] != prev[a]:
                        diffs["%s->%s" % (prev[a], names[a])] += 1
            prev = names
            if term or trunc:
                break
        out[arm] = {
            "return": float(sum(rewards)),
            "ticks": len(rewards),
            "action_counts": dict(actions_by_name),
            "arm_switches": dict(diffs.most_common(12)),
            "rewards": rewards,
            "per_tick": per_tick[:limit_ticks] if limit_ticks else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--ablation", choices=("real", "zero"), default="real")
    ap.add_argument("--bonus", type=float, default=1.0)
    ap.add_argument("--limit-ticks", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = trace(args.model_dir, args.seed, args.steps, args.ablation,
                args.bonus, args.limit_ticks)
    L, A = res["lancer"], res["actor"]
    print("seed %d  model %s" % (args.seed, args.model_dir))
    print("  lancer return %8.1f  ticks %d" % (L["return"], L["ticks"]))
    print("  actor  return %8.1f  ticks %d" % (A["return"], A["ticks"]))
    print("\n  action mix (lancer): %s" % L["action_counts"])
    print("  action mix (actor) : %s" % A["action_counts"])
    print("\n  most common arm switches: %s" % A["arm_switches"])
    lr, ar = L["rewards"], A["rewards"]
    n = min(len(lr), len(ar))
    print("\n  reward by decile (lancer -> actor):")
    for d in range(10):
        i0, i1 = d * n // 10, (d + 1) * n // 10
        if i1 <= i0:
            continue
        print("    ticks %3d-%3d  %8.1f -> %8.1f   (delta %+8.1f)"
              % (i0, i1 - 1, sum(lr[i0:i1]) / (i1 - i0),
                 sum(ar[i0:i1]) / (i1 - i0),
                 sum(ar[i0:i1]) / (i1 - i0) - sum(lr[i0:i1]) / (i1 - i0)))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print("\nwrote", args.out)


if __name__ == "__main__":
    sys.exit(main())