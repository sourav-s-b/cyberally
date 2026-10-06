"""Mechanism check: does the coverage guard remove the agent-trap failure?

Diagnosis (blue/training/trace_episode.py, dev seed 8241): the actor's
agent 3 spent 162 of 200 ticks re-analysing a single router, confirmed no
hosts, and issued zero Remove/Restore against Lancer's 14. The team pays
because all five agents share one reward. Cause: the learner sweep ranks the
FULL candidate list every tick (guard off), the residual has no memory, and
nothing about a clean result lowers a host's score -- so the argmax repeats.

Hypothesis H: turning on the strict coverage guard (rank only among hosts
unvisited this sweep, the mechanism Lancer's cursor uses) removes the trap.

This is a MECHANISM CHECK, not a performance claim. It runs 3 fixed seeds
spanning the known failure mode (8241) and two known wins (8237, 8235), and
reports per-agent remediation, not just return. Falsification condition:
if the guard also removes the wins, then the actor's value came from
unrestricted prioritisation and the guard is too blunt.
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

from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv  # noqa: E402
from blue.training import mappo_guide as mg  # noqa: E402
from blue.training import risk_actor as ra  # noqa: E402

AGENTS = ["blue_agent_%d" % i for i in range(5)]


def run_arm(model_dir, seed, steps, guard, ablation="real", bonus=1.0):
    import torch as th
    from blue.policies.ordered import LancerValues, OrderedPolicy

    actor = ra.build_actor(hidden=64)
    actor.load_state_dict(th.load(os.path.join(model_dir, "actor.th"),
                                  map_location="cpu", weights_only=False))
    actor.eval()
    risk = ra._risk(ablation)

    out = {}
    for arm in ("lancer", "actor"):
        env = CC4MARLEnv(seed=seed, steps=steps, **mg.ENV_KW)
        env.reset(seed=seed)
        pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                            guard=(arm == "actor" and guard == "strict"))
        pol.reset()
        pol.hook = ra.RiskGreedyHook(actor, bonus, risk) \
            if arm == "actor" else None
        rewards = []
        rem = collections.Counter()
        analyse = collections.Counter()
        picks = collections.Counter()
        conf = set()
        for _ in range(steps):
            acts = {}
            for a in AGENTS:
                if int(env.get_avail_agent_actions(
                        AGENTS.index(a)).sum()) <= 1:
                    continue
                acts[a] = int(pol.select(env, a))
                from blue.core.baselines import decode_index
                name, host = decode_index(env, a, acts[a])
                if arm == "actor" and name == "Analyse" and host:
                    picks[host] += 1
                if arm != "actor":
                    continue
                if name == "Analyse":
                    analyse[a] += 1
                elif name in ("Remove", "Restore"):
                    rem[a] += 1
            _, rew, term, trunc, _ = env.step(acts)
            rewards.append(float(rew[0]))
            if term or trunc:
                break
        for a in AGENTS:
            for h, st in env.trackers[a].state.items():
                if st in ("CONFIRMED", "VERIFY"):
                    conf.add(h)
        out[arm] = {"return": float(sum(rewards)),
                    "remediation": dict(rem),
                    "analyse": dict(analyse),
                    "total_remediation": sum(rem.values()),
                    "confirmed_hosts": len(conf),
                    "top_pick": picks.most_common(3) if picks else [],
                    "top_pick_share": (round(100.0 * picks.most_common(1)[0][1]
                                             / max(sum(picks.values()), 1), 1)
                                       if picks else None)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--ablation", default="real")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = {}
    print("model %s  seeds %s" % (args.model_dir, args.seeds))
    print("%-6s %-8s | %9s %9s %8s | %9s %9s %8s | %s"
          % ("seed", "arm", "L_ret", "A_ret", "A-L",
             "L_rem", "A_rem", "A_conf", "A_top1%"))
    for seed in args.seeds:
        row = run_arm(args.model_dir, seed, args.steps, "strict",
                      args.ablation)
        res[str(seed)] = row
        for arm in ("lancer", "actor"):
            d = row[arm]
            tag = "Lancer" if arm == "lancer" else "actor"
            delta = "" if arm == "lancer" else \
                "%+8.0f" % (d["return"] - row["lancer"]["return"])
            print("%-6d %-8s | %9.0f %9s %8s | %9s %9s %8s | %s"
                  % (seed, tag,
                     d["return"] if arm == "lancer" else
                     row["lancer"]["return"],
                     "%.0f" % d["return"] if arm == "actor" else "",
                     delta if arm == "actor" else "",
                     sum(row["lancer"]["remediation"].values())
                     if arm == "lancer" else "",
                     d["total_remediation"] if arm == "actor" else "",
                     d["confirmed_hosts"] if arm == "actor" else "",
                     d["top_pick_share"] if arm == "actor" else ""))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    sys.exit(main())