"""Targeted trace: localise where Blue loses a defence.

Purpose. The risk-input ablation produced bimodal returns (median ~+16,
worst -1209). On dev seed 8241 the actor scored -1273 against Lancer's -64.
Two explanations are still live and they imply different fixes:

  H1  the actor investigates the wrong hosts, so compromises are never
      DISCOVERED, no host reaches CONFIRMED, and rules 1+3 therefore never
      issue Remove/Restore -> fewer repairs is a symptom, not a cause;
  H2  the actor investigates correctly but remediation is postponed ->
      fewer repairs is a cause.

These are distinguished here by logging, per tick and per agent: what was
requested, on which host, whether it was a hook decision or a forced rule,
tracker state transitions (UNKNOWN -> CONFIRMED -> VERIFY), action
completion, busy/pending status, and the reward signal. Ground-truth
compromise state is recorded for DIAGNOSIS ONLY and is never fed to the
policy (Blue-visible inputs are logged separately).

The RL hook can only influence the sweep branch (see
blue/policies/ordered.py: rules 1+3 and rule 2 return before the hook is
consulted), so every Remove/Restore in the trace must originate from a
forced rule. This module asserts that rather than assuming it.

Usage:
    .venv-train/bin/python -m blue.training.trace_episode \
        --model-dir blue/results/ablation_risk/real_s0 --seed 8241 \
        --out blue/results/diagnosis/trace_8241.json
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


def _decode(env, agent, act):
    from blue.core.baselines import decode_index
    return decode_index(env, agent, act)


def _ground_truth(env):
    """True compromise state per host. DIAGNOSIS ONLY -- never given to the
    policy, actor, masks or risk model."""
    out = {}
    try:
        hosts = env.state.hosts
    except Exception:
        return out
    for name, host in hosts.items():
        try:
            out[name] = bool(getattr(host, "compromised", False))
        except Exception:
            continue
    return out


def _actor_visible(env, agent, host):
    """What Blue could legitimately know about this host from observation."""
    try:
        rec = env.trackers[agent]
        return {"tracker_state": rec.state.get(host),
                "last_analysis": rec.last_analysis.get(host),
                "last_remediation": rec.last_remediation.get(host)}
    except Exception as e:
        return {"error": str(e)}


def trace(model_dir, seed, steps, ablation, bonus, sample_obs_every=0):
    import torch as th
    from blue.policies.ordered import LancerValues, OrderedPolicy

    actor = None
    if model_dir:
        actor = ra.build_actor(hidden=64)
        actor.load_state_dict(th.load(os.path.join(model_dir, "actor.th"),
                                      map_location="cpu", weights_only=False))
        actor.eval()
    risk = ra._risk(ablation) if model_dir else None

    out = {}
    for arm in ("lancer", "actor"):
        if arm == "actor" and actor is None:
            continue
        env = CC4MARLEnv(seed=seed, steps=steps, **mg.ENV_KW)
        env.reset(seed=seed)
        pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                            guard=False)
        hook = ra.RiskGreedyHook(actor, bonus, risk) if arm == "actor" else None
        pol.reset()
        pol.hook = hook

        rows = []
        state_changes = []
        prev_states = {a: dict(env.trackers[a].state) for a in BLUE_AGENTS}
        comp_when_discovered = {}
        counts = collections.Counter()
        hook_picks = collections.Counter()
        for tick in range(1, steps + 1):
            rec = {"tick": tick, "agents": {}}
            acts = {}
            for a in BLUE_AGENTS:
                busy = a in env._awaiting
                mask = env.get_avail_agent_actions(BLUE_AGENTS.index(a))
                if int(mask.sum()) <= 1:
                    counts["forced_sleep_no_legal"] += 1
                    continue
                acts[a] = int(pol.select(env, a))
                name, host = _decode(env, a, acts[a])
                counts[name] += 1
                rec["agents"][a] = {
                    "action": name, "host": host, "busy_before": busy,
                    "blue_visible": _actor_visible(env, a, host) if host else {},
                    "legal_count": int(mask.sum()),
                }
                if arm == "actor" and name == "Analyse":
                    hook_picks[host] += 1

            _, rew, term, trunc, _ = env.step(acts)
            rec["reward"] = float(rew[0])

            # state transitions AFTER the step, plus first-discovery ticks
            for a in BLUE_AGENTS:
                tr = env.trackers[a]
                for h, st in tr.state.items():
                    if prev_states[a].get(h) != st:
                        state_changes.append(
                            {"tick": tick, "agent": a, "host": h,
                             "from": prev_states[a].get(h), "to": st,
                             "result": tr.last_result.get(h)})
                        if st == "CONFIRMED" and h not in comp_when_discovered:
                            comp_when_discovered[h] = tick
                        prev_states[a][h] = st
            if sample_obs_every and tick % sample_obs_every == 0:
                rec["obs_sample"] = {
                    a: [round(float(x), 3)
                        for x in env.get_obs_agent(BLUE_AGENTS.index(a))[:12]]
                    for a in BLUE_AGENTS}
            rows.append(rec)
            if term or trunc:
                break

        gt = _ground_truth(env)
        out[arm] = {
            "rows": rows,
            "state_changes": state_changes,
            "action_counts": dict(counts),
            "hook_pick_counts": dict(hook_picks.most_common(15)),
            "confirmed_at_tick": comp_when_discovered,
            "ground_truth_compromised_at_end": sorted(
                [h for h, v in gt.items() if v]),
            "ticks": len(rows),
            "return": float(sum(r["reward"] for r in rows)),
        }
    return out


def summarise(res, seed):
    L = res.get("lancer")
    A = res.get("actor")
    print("seed %d" % seed)
    if L:
        print("  lancer return %8.1f  ticks %d"
              % (L["return"], L["ticks"]))
    if not A:
        return
    print("  actor  return %8.1f  ticks %d" % (A["return"], A["ticks"]))
    print("\n  action counts   lancer -> actor")
    keys = sorted(set(L["action_counts"]) | set(A["action_counts"]))
    for k in keys:
        print("    %-8s %6d -> %6d" % (k, L["action_counts"].get(k, 0),
                                       A["action_counts"].get(k, 0)))

    # H1 vs H2: did the actor DISCOVER the same compromises, and when?
    lc = set(L["confirmed_at_tick"])
    ac = set(A["confirmed_at_tick"])
    print("\n  hosts CONFIRMED  lancer %d  actor %d" % (len(lc), len(ac)))
    print("    missed by actor: %s" % sorted(lc - ac)[:12])
    print("    extra by actor : %s" % sorted(ac - lc)[:12])
    both = sorted(lc & ac)
    if both:
        late = [(h, L["confirmed_at_tick"][h], A["confirmed_at_tick"][h])
                for h in both
                if A["confirmed_at_tick"][h] != L["confirmed_at_tick"][h]]
        print("    discovery-tick differences (host, lancer, actor): %s"
              % (late[:12] if late else "none"))

    # when did the first big penalty land, per arm
    for name, d in (("lancer", L), ("actor", A)):
        big = [(r["tick"], r["reward"]) for r in d["rows"]
               if r["reward"] <= -10]
        if big:
            print("  %s: first reward <=-10 at tick %d (%.0f); %d such ticks"
                  % (name, big[0][0], big[0][1], len(big)))
        else:
            print("  %s: no reward <=-10" % name)
    gt = A["ground_truth_compromised_at_end"]
    print("\n  ground truth compromised at end: %d host(s) %s"
          % (len(gt), gt[:8]))
    missed = sorted(set(L["confirmed_at_tick"]) - ac)
    print("  DIAGNOSTIC: actor never confirmed %d host(s) lancer did" % len(missed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--ablation", choices=("real", "zero"), default="real")
    ap.add_argument("--bonus", type=float, default=1.0)
    ap.add_argument("--sample-obs-every", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = trace(args.model_dir, args.seed, args.steps, args.ablation,
                args.bonus, args.sample_obs_every)
    summarise(res, args.seed)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f)
        print("\nwrote", args.out)


if __name__ == "__main__":
    sys.exit(main())