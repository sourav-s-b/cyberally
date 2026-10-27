"""Recency penalty: the middle ground between the hard guard and free ranking.

Why. Measured on dev seeds 8241/8237/8235 with the real_s0 checkpoint:

  guard off    -> top-1 pick share 8.8-18.1%, agent 3 spent 162/200 ticks on
                  one router on 8241 and issued zero remediation; delta -1209.
  guard strict -> top-1 share 2.3-4.0%, remediation rises on every seed,
                  8241 delta +19; but the actor still loses to Lancer on two
                  of three seeds because coverage discipline leaves no room
                  for prioritisation.

The hard guard buys coverage by forbidding revisits outright. This decays a
host's score by how recently it was last analysed, so revisits stay legal
but must earn their place. Read-out is Blue-visible only: `last_analysis`
comes from the tracker, which records completed analyses and is already the
input to Lancer's sweep and the MaxAgeGuard.

Decay: score' = score - penalty * exp(-(tick - last_an) / tau)
  penalty=0 reproduces guard=off exactly (so the flag is testable as a no-op)
  tau -> 0 approaches the hard guard

This is a MECHANISM CHECK, not a performance claim: 3 fixed seeds, and the
falsification condition is that it must beat BOTH guard=off and guard=strict
on the collapse seed without giving up the wins.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from blue.core.baselines import decode_index  # noqa: E402
from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv  # noqa: E402
from blue.training import mappo_guide as mg  # noqa: E402
from blue.training import risk_actor as ra  # noqa: E402

AGENTS = ["blue_agent_%d" % i for i in range(5)]


class RecencyPenaltyHook(object):
    """Wraps the greedy actor pick with a recency decay.

    The decay is applied to the actor's residual contribution, so the hook
    still only chooses which host to INVESTIGATE. It cannot issue Remove or
    Restore: those come from rules 1+3, which return before the hook is
    consulted (blue/policies/ordered.py).
    """

    def __init__(self, inner, penalty, tau):
        self.inner = inner
        self.penalty = float(penalty)
        self.tau = float(tau)

    def __call__(self, env, agent, cands, scored):
        if self.penalty <= 0.0:
            return self.inner(env, agent, cands, scored)
        import torch as th
        from blue.policies.ordered import argmax_pick
        tracker = env.trackers[agent]
        tick = getattr(env, "_tick", 0) or 0
        comb = []
        for base, host in scored:
            last = tracker.last_analysis.get(host)
            if last is None:
                decay = 0.0
            else:
                age = max(0, tick - last)
                decay = self.penalty * math.exp(-age / max(self.tau, 1e-6))
            comb.append((base - decay, host))
        return argmax_pick(comb)


def run_arm(model_dir, seed, steps, penalty, tau, ablation="real", bonus=1.0):
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
                            guard=False)
        pol.reset()
        if arm == "actor":
            inner = ra.RiskGreedyHook(actor, bonus, risk)
            pol.hook = (RecencyPenaltyHook(inner, penalty, tau)
                        if penalty > 0 else inner)
        rem = collections.Counter()
        picks = collections.Counter()
        conf = set()
        rewards = []
        for _ in range(steps):
            acts = {}
            for a in AGENTS:
                if int(env.get_avail_agent_actions(AGENTS.index(a)).sum()) <= 1:
                    continue
                acts[a] = int(pol.select(env, a))
                name, host = decode_index(env, a, acts[a])
                if name == "Analyse" and host:
                    picks[host] += 1
                if name in ("Remove", "Restore"):
                    rem[a] += 1
            _, rew, term, trunc, _ = env.step(acts)
            rewards.append(float(rew[0]))
            for a in AGENTS:
                for h, st in env.trackers[a].state.items():
                    if st == "CONFIRMED":
                        conf.add(h)
            if term or trunc:
                break
        total_picks = max(sum(picks.values()), 1)
        out[arm] = {"return": float(sum(rewards)),
                    "total_remediation": sum(rem.values()),
                    "confirmed_hosts": len(conf),
                    "distinct_analysed": len(picks),
                    "top_pick_share": round(
                        100.0 * picks.most_common(1)[0][1] / total_picks, 1)
                    if picks else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--penalty", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("penalty %.2f  tau %.1f" % (args.penalty, args.tau))
    print("%-6s | %8s %8s %7s | %6s %6s %6s %6s %7s"
          % ("seed", "L_ret", "A_ret", "delta", "L_rem", "A_rem",
             "L_conf", "A_conf", "top1%"))
    res = {}
    for seed in args.seeds:
        row = run_arm(args.model_dir, seed, args.steps, args.penalty, args.tau)
        res[str(seed)] = row
        L, A = row["lancer"], row["actor"]
        print("%-6d | %8.0f %8.0f %+7.0f | %6d %6d %6d %6d %7s"
              % (seed, L["return"], A["return"],
                 A["return"] - L["return"],
                 L["total_remediation"], A["total_remediation"],
                 L["confirmed_hosts"], A["confirmed_hosts"],
                 A["top_pick_share"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"penalty": args.penalty, "tau": args.tau, "seeds": res},
                  f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())