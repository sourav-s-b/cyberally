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


def run_arm(model_dir, seed, steps, guard="strict", ablation="real",
            bonus=1.0):
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
                if name == "Analyse" and host:
                    picks[host] += 1
                # Count behaviour for BOTH arms. An earlier version gated
                # this on the actor, which reported Lancer remediation as 0
                # and made the mechanism table unreadable.
                if name == "Analyse":
                    analyse[a] += 1
                elif name in ("Remove", "Restore"):
                    rem[a] += 1
            _, rew, term, trunc, _ = env.step(acts)
            rewards.append(float(rew[0]))
            # Count hosts CONFIRMED at any point during the episode. Reading
            # FINAL state instead reports ~0, because a remediated host
            # moves CONFIRMED -> VERIFY -> UNKNOWN once its re-analysis is
            # clean, so the end-of-episode snapshot misses every discovery.
            for a in AGENTS:
                tr = env.trackers[a]
                for h, st in tr.state.items():
                    if st == "CONFIRMED":
                        conf.add(h)
            if term or trunc:
                break
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
    ap.add_argument("--guard", choices=("off", "strict"), default="strict",
                    help="which learner regime to measure; the Lancer arm "
                         "is always unguarded (reference behaviour)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--ablation", default="real")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = {}
    print("model %s  seeds %s  guard %s"
          % (args.model_dir, args.seeds, args.guard))
    hdr = ("seed", "arm", "return", "L_ret", "delta", "L_rem", "A_rem",
           "L_conf", "A_conf", "A_top1%")
    print("%-6s %-8s | %9s %9s %8s | %6s %6s | %6s %6s | %s" % hdr)
    for seed in args.seeds:
        row = run_arm(args.model_dir, seed, args.steps, args.guard,
                      args.ablation)
        res[str(seed)] = row
        L, A = row["lancer"], row["actor"]
        for arm in ("lancer", "actor"):
            d = row[arm]
            other = A if arm == "lancer" else L
            print("%-6d %-8s | %9.0f %9.0f %8s | %6d %6d | %6d %6d | %s"
                  % (seed, arm, d["return"], other["return"],
                     ("%+8.0f" % (d["return"] - L["return"]))
                     if arm == "actor" else "",
                     sum(L["remediation"].values()),
                     A["total_remediation"],
                     L["confirmed_hosts"], A["confirmed_hosts"],
                     A["top_pick_share"] if arm == "actor" else ""))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    sys.exit(main())