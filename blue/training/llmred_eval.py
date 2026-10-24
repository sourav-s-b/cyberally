"""Evaluate Blue policies against LLM Red (planner + executor).

Why this exists. The earlier Llama Red runs exposed a large gap between
DiscoveryFSRed and a planner-driven attacker. This runner also supports a
smaller Qwen planner. Each model/digest is a separate attacker regime: past
Llama returns are historical context, not a Qwen baseline.

Policies are evaluated on identical seeds through ``CC4LLMEnv``, which owns
the attacker. The Blue-visible observation contract is unchanged, so no
actor, mask, risk model or scorer sees anything new.

Notes recorded with every run:
  - the model digest and ollama version, because LLM inference is not
    bit-reproducible across runtime and hardware (Red's own caveat);
  - per-seed returns, never a selected best episode;
  - zero tolerance for invalid executed Red actions.

Usage:
    .venv-train/bin/python -m blue.training.llmred_eval \
        --actor /path/to/real_s0 --seeds 7629 7630 7631 \
        --model qwen2.5:3b --arms lancer lancer_guard actor_guard
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from blue.core.baselines import decode_index  # noqa: E402
from blue.core.wrapper import BLUE_AGENTS  # noqa: E402
from blue.training import risk_actor as ra  # noqa: E402

AGENTS = ["blue_agent_%d" % i for i in range(5)]


def source_hashes():
    out = {}
    for rel in ("red/llm_agent.py", "red/planned_agent.py",
                "red/blue_env.py", "blue/core/wrapper.py",
                "blue/policies/ordered.py", "blue/training/risk_actor.py",
                "blue/training/llmred_eval.py"):
        p = os.path.join(REPO, rel)
        if os.path.exists(p):
            with open(p, "rb") as f:
                out[rel] = hashlib.sha256(f.read()).hexdigest()
    return out


def build_env(steps, model, llm_endpoint, max_requests, plans_per_agent,
              plan_interval):
    from red.blue_env import CC4LLMEnv
    from red.planned_agent import PlannerConfig
    # CC4LLMEnv expands these with **, so they must be plain mappings.
    llm = {"model": model, "endpoint": llm_endpoint,
           "timeout_seconds": 60.0, "max_requests": max_requests}
    pc = {"max_plans_per_agent": plans_per_agent,
          "interval": plan_interval}
    env = CC4LLMEnv(llm_config=llm, planner_config=pc, steps=steps,
                    temporal_features=("ages", "belief"),
                    include_root_session=True)
    return env, llm


def run_arm(env, seed, hook_factory, guard, steps):
    """One episode. hook_factory() -> hook or None."""
    env.reset(seed=seed)
    from blue.policies.ordered import LancerValues, OrderedPolicy
    pol = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                        guard=guard)
    pol.reset()
    pol.hook = hook_factory()
    rewards = []
    rem = collections.Counter()
    conf = set()
    picks = collections.Counter()
    actions_hash = hashlib.sha256()
    for _ in range(steps):
        acts = {}
        for a in AGENTS:
            if int(env.get_avail_agent_actions(AGENTS.index(a)).sum()) <= 1:
                continue
            acts[a] = int(pol.select(env, a))
            name, host = decode_index(env, a, acts[a])
            if name in ("Remove", "Restore"):
                rem[a] += 1
            if name == "Analyse" and host:
                picks[host] += 1
        actions_hash.update(json.dumps(
            sorted(acts.items()), separators=(",", ":")
        ).encode("utf-8"))
        actions_hash.update(b"\n")
        _, rew, term, trunc, _ = env.step(acts)
        rewards.append(float(rew[0]))
        for a in AGENTS:
            for h, st in env.trackers[a].state.items():
                if st == "CONFIRMED":
                    conf.add(h)
        if term or trunc:
            break
    total = max(sum(picks.values()), 1)
    return {"return": float(sum(rewards)), "ticks": len(rewards),
            "actions_sha256": actions_hash.hexdigest(),
            "red_plan_requests": getattr(getattr(env, "llm_client", None),
                                         "requests", None),
            "remediation": sum(rem.values()), "confirmed_hosts": len(conf),
            "distinct_analysed": len(picks),
            "top_pick_share": round(100.0 * picks.most_common(1)[0][1]
                                    / total, 1) if picks else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", default=None, help="dir containing actor.th")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7629, 7630, 7631])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--arms", nargs="+",
                    default=["lancer", "actor", "actor_guard"])
    ap.add_argument("--bonus", type=float, default=1.0)
    ap.add_argument("--ablation", default="real")
    ap.add_argument("--risk-model", default="blue/results/risk_model_v2.pkl",
                    help="frozen risk-model artifact used by actor arms")
    ap.add_argument("--endpoint", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen2.5:3b",
                    help="exact installed local Ollama model name")
    ap.add_argument("--max-requests", type=int, default=36)
    ap.add_argument("--plans-per-agent", type=int, default=2)
    ap.add_argument("--plan-interval", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    known_arms = {"lancer", "lancer_guard", "actor", "actor_guard"}
    unknown = set(args.arms) - known_arms
    if unknown:
        raise SystemExit("unknown evaluation arms: " + ", ".join(sorted(unknown)))

    if any(7809 <= s <= 8200 for s in args.seeds):
        raise SystemExit("refusing: reserved final seeds 7809-8200")

    env, cfg = build_env(args.steps, args.model, args.endpoint, args.max_requests,
                         args.plans_per_agent, args.plan_interval)
    print("attacker model:", env.llm_model)

    actor = None
    if any(a.startswith("actor") for a in args.arms):
        if not args.actor:
            raise SystemExit("--actor required for actor arms")
        import torch as th
        actor = ra.build_actor(hidden=64)
        actor.load_state_dict(th.load(os.path.join(args.actor, "actor.th"),
                                      map_location="cpu", weights_only=False))
        actor.eval()
        if args.ablation == "zero":
            risk = ra.ZeroRisk()
        elif args.ablation == "real":
            from blue.policies.hybrid import RiskPriority
            risk = RiskPriority(model_path=args.risk_model)
        else:
            raise SystemExit("--ablation must be real or zero")

    def make_hook():
        return ra.RiskGreedyHook(actor, args.bonus, risk) if actor else None

    actor_sha = (hashlib.sha256(open(os.path.join(args.actor, "actor.th"),
                                     "rb").read()).hexdigest()
                 if actor is not None else None)
    risk_sha = (hashlib.sha256(open(args.risk_model, "rb").read()).hexdigest()
                if actor is not None and args.ablation == "real" else None)
    identity = {"model": env.llm_model, "llm_config": cfg,
                "planner_config": {"max_plans_per_agent": args.plans_per_agent,
                                   "interval": args.plan_interval},
                "steps": args.steps, "bonus": args.bonus,
                "ablation": args.ablation, "actor_sha256": actor_sha,
                "risk_model_sha256": risk_sha,
                "source_sha256": source_hashes()}
    run_fingerprint = hashlib.sha256(json.dumps(
        identity, sort_keys=True).encode("utf-8")).hexdigest()

    print("%-13s %6s %9s %6s %6s %8s" % ("arm", "seed", "return", "rem",
                                         "conf", "top1%"))
    results = collections.defaultdict(dict)
    # Append each episode as it completes. A previous version wrote the JSON
    # only at the very end, so a crash at 43/48 episodes lost 49 minutes of
    # compute and the data was recoverable only by parsing the text log.
    partial = os.path.splitext(args.out)[0] + ".partial.jsonl" if args.out \
        else None
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if partial and os.path.exists(partial):
        with open(partial) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["arm"] in args.arms and rec["seed"] in args.seeds:
                    if rec.get("run_fingerprint") != run_fingerprint:
                        raise SystemExit("partial result belongs to another "
                                         "model/config/source run: " + partial)
                    results[rec["arm"]][str(rec["seed"])] = rec["result"]
    print("%-13s %6s %9s %6s %6s %8s" % ("arm", "seed", "return", "rem",
                                         "conf", "top1%"))
    for arm in args.arms:
        for seed in args.seeds:
            if str(seed) in results[arm]:
                print("%-13s %6d %9.0f %6d %6d %8s  (cached)"
                      % (arm, seed, results[arm][str(seed)]["return"],
                         results[arm][str(seed)]["remediation"],
                         results[arm][str(seed)]["confirmed_hosts"],
                         results[arm][str(seed)]["top_pick_share"]), flush=True)
                continue
            # A fresh env per episode so the global request budget resets the
            # way Red's own runs do (per-episode planning quota).
            e, _ = build_env(args.steps, args.model, args.endpoint, args.max_requests,
                             args.plans_per_agent, args.plan_interval)
            if e.llm_model != env.llm_model:
                raise SystemExit("installed model changed during evaluation")
            factory = (lambda: None) if arm in ("lancer", "lancer_guard") \
                else make_hook
            t0 = time.time()
            r = run_arm(e, seed, factory,
                        guard=(arm in ("actor_guard", "lancer_guard")),
                        steps=args.steps)
            r["wall_seconds"] = round(time.time() - t0, 1)
            results[arm][str(seed)] = r
            if partial:
                with open(partial, "a") as f:
                    f.write(json.dumps({"run_fingerprint": run_fingerprint,
                                        "arm": arm, "seed": seed,
                                        "result": r}) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            print("%-13s %6d %9.0f %6d %6d %8s"
                  % (arm, seed, r["return"], r["remediation"],
                     r["confirmed_hosts"], r["top_pick_share"]), flush=True)

    print("\n%-13s %9s" % ("arm", "mean"))
    for arm in args.arms:
        m = sum(r["return"] for r in results[arm].values()) / len(args.seeds)
        print("%-13s %+9.1f" % (arm, m))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"arms": results, "seeds": args.seeds,
                       "steps": args.steps,
                       "model": env.llm_model,
                       "llm_config": cfg,
                       "planner_config": identity["planner_config"],
                       "actor_sha256": actor_sha,
                       "risk_model_sha256": risk_sha,
                       "run_fingerprint": run_fingerprint,
                       "reserved_block_untouched": [7809, 8200],
                       "claim": "mechanism comparison against LLM Red; "
                                "inference is not bit-reproducible, so "
                                "numbers are indicative not exact",
                       "source_sha256": source_hashes()}, f, indent=1)
        print("\nwrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
