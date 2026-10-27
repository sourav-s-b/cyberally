"""Bounded mechanism evaluation, with separate privileged diagnostic events.

Default episodes are already-consumed diagnostic seeds, never a fresh claim.
All arms use native reward. No simulator changes and no label-fed policy.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

import numpy as np

from blue.core.baselines import decode_index
from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv
from blue.harness.features import BlueFeatures, NAMES, VERSION
from blue.harness.policy import HarnessPolicy, MLRankingHook
from blue.harness.scoring import MLScorer, sha256
from blue.policies.ordered import LancerValues, OrderedPolicy, completed_investigation_age_summary
from blue.training import mappo_guide as mg
from blue.training.harness_rl import check_seeds, load_hook


def penalty_events(env):
    """Mirror upstream BlueRewardMachine's event accounting, diagnosis only.

    Every tick's contribution sum is checked against the native reward by
    the evaluator. A mismatch fails visibly instead of inventing a cause.
    """
    from CybORG.Shared.BlueRewardMachine import BlueRewardMachine
    from CybORG.Simulator.Actions.GreenActions import GreenAccessService, GreenLocalWork
    from CybORG.Simulator.Actions.AbstractActions.Impact import Impact
    controller = env.env
    state = controller.state
    weights = BlueRewardMachine("Blue").get_phase_rewards(state.mission_phase)
    events = []
    for agent, actions in controller.action.items():
        if not actions:
            continue
        action = actions[0]
        if isinstance(action, Impact):
            host, kind = action.hostname, "RIA"
        elif isinstance(action, (GreenAccessService, GreenLocalWork)):
            host = state.ip_addresses[action.ip_address]
            kind = "LWF" if isinstance(action, GreenLocalWork) else "ASF"
        else:
            continue
        if not any(s.active for s in state.sessions[agent].values()):
            continue
        success = controller.observation[agent].observations[0].data["success"]
        counted = ("green" in agent and success == False) or (
            "red" in agent and success and isinstance(action, Impact))
        if counted:
            zone = state.hostname_subnet_map[host].value
            events.append({"agent": agent, "host": host, "event": kind,
                           "phase": state.mission_phase, "penalty": weights[zone][kind]})
    # Preserve the simulator's separately reported action cost, if nonzero.
    cost = float(controller.get_reward(BLUE_AGENTS[0]).get("action_cost", 0))
    if cost:
        events.append({"event": "action_cost", "penalty": cost})
    return events


def make_policy(arm, args):
    if arm == "lancer":
        return OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5)), None
    if arm == "guard":
        return HarnessPolicy(max_age=args.max_age), None
    if arm == "ml":
        scorer = MLScorer(args.scorer)
        return HarnessPolicy(MLRankingHook(scorer), args.max_age), scorer.sha256
    if arm == "actor":
        hook, manifest = load_hook(args.model_dir, args.scorer)
        if args.max_age != manifest["config"]["max_age"] or args.steps != manifest["config"]["steps"]:
            raise ValueError("actor evaluation must use its frozen harness/horizon")
        return HarnessPolicy(hook, args.max_age), manifest["actor_sha256"]
    if arm in ("legacy", "legacy_guard"):
        import torch as th
        from blue.training import risk_actor as ra
        actor = ra.build_actor()
        actor.load_state_dict(th.load(Path(args.legacy_model) / "actor.th", map_location="cpu", weights_only=True))
        actor.eval()
        risk = ra._risk("real")  # retained ML model; coverage is the ONLY intervention
        hook = ra.RiskGreedyHook(actor, 1.0, risk)
        policy = (HarnessPolicy(hook, args.max_age) if arm == "legacy_guard"
                  else OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5)))
        policy.hook = hook
        return policy, sha256(Path(args.legacy_model) / "actor.th")
    raise ValueError("unknown policy arm")


def episode(policy, seed, steps, trace_path):
    from blue.training.risk_data import true_compromised
    env = CC4MARLEnv(seed=seed, steps=steps, **mg.ENV_KW)
    env.reset(seed=seed)
    policy.reset()
    pipe = BlueFeatures()
    total, repairs, analyses = 0.0, 0, {}
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    privileged_path = trace_path.with_suffix(".privileged.jsonl")
    with trace_path.open("w") as blue_log, privileged_path.open("w") as truth_log:
        for _ in range(steps):
            records, actions = {}, {}
            previous = dict(env._awaiting)
            truth_before = sorted(true_compromised(env))
            for agent in BLUE_AGENTS:
                X = pipe.rows(env, agent)
                requested = int(policy.select(env, agent))
                actions[agent] = requested
                name, host = decode_index(env, agent, requested)
                if name in ("Remove", "Restore"):
                    repairs += 1
                if name == "Analyse":
                    key = (agent, host)
                    analyses[key] = analyses.get(key, 0) + 1
                request = copy.deepcopy(getattr(policy, "last_request", None))
                row = X[env.hostnames[agent].index(host)] if host else None
                records[agent] = {"action": name, "host": host,
                                  "busy_before": agent in previous,
                                  "pending_before": previous.get(agent),
                                  "request": request,
                                  "features": ([float(x) if np.isfinite(x) else None for x in row]
                                               if row is not None else None)}
            _, reward, terminated, truncated, _ = env.step(actions)
            reward = float(reward[0])
            total += reward
            events = penalty_events(env)
            if not np.isclose(sum(e["penalty"] for e in events), reward):
                raise RuntimeError("penalty event accounting differs from native reward")
            for agent in BLUE_AGENTS:
                rec = records[agent]
                completed = previous.get(agent) if agent not in env._awaiting else None
                rec["pending_after"] = env._awaiting.get(agent)
                rec["completion"] = None
                if completed:
                    host, name = completed
                    rec["completion"] = {"host": host, "action": name,
                                         "result": env.trackers[agent].last_result.get(host),
                                         "belief_after": env.trackers[agent].state.get(host),
                                         "last_analysis": env.trackers[agent].last_analysis.get(host),
                                         "last_remediation": env.trackers[agent].last_remediation.get(host)}
            blue_log.write(json.dumps({"tick": env._tick, "agents": records,
                                       "reward": reward}, allow_nan=False) + "\n")
            truth_log.write(json.dumps({"tick": env._tick,
                                        "compromised_before": truth_before,
                                        "compromised_after": sorted(true_compromised(env)),
                                        "penalty_events": events}) + "\n")
            if terminated or truncated:
                break
    result = {"return": total, "ticks": env._tick, "repair_requests": repairs,
              "max_analyses_one_host": max(analyses.values(), default=0),
              "completed_ages": {a: completed_investigation_age_summary(env, a) for a in BLUE_AGENTS},
              "guard": policy.guard_stats(), "blue_trace": str(trace_path),
              "privileged_diagnostic": str(privileged_path)}
    env.close()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[8241, 8229])
    ap.add_argument("--arms", nargs="+", choices=("lancer", "guard", "ml", "legacy", "legacy_guard", "actor"),
                    default=["lancer", "legacy", "legacy_guard", "ml"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--max-age", type=float, default=80)
    ap.add_argument("--scorer", default="blue/results/gpt_harness/scorer.pkl")
    ap.add_argument("--legacy-model", default="blue/results/ablation_risk/real_s0")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--out", default="blue/results/gpt_harness/diagnostic")
    args = ap.parse_args()
    check_seeds(args.seeds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").exists() or (out / "frozen_config.json").exists():
        raise FileExistsError("use a new evaluation directory")
    report = {"version": VERSION, "config": vars(args), "env_kw": mg.ENV_KW,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "claim": "diagnostic mechanism check; no held-out performance claim", "cells": {}}
    report["source_files"] = {str(p): sha256(p) for p in [
        Path(__file__), *sorted(Path("blue/harness").glob("*.py")),
        Path("blue/training/harness_rl.py"), Path("blue/core/wrapper.py"),
        Path("blue/policies/ordered.py")]}
    report["artifacts_before_run"] = {}
    if any(a in args.arms for a in ("legacy", "legacy_guard")):
        report["artifacts_before_run"][args.legacy_model + "/actor.th"] = sha256(Path(args.legacy_model) / "actor.th")
        report["artifacts_before_run"]["blue/results/risk_model_v2.pkl"] = sha256("blue/results/risk_model_v2.pkl")
    if any(a in args.arms for a in ("ml", "actor")):
        report["artifacts_before_run"][args.scorer] = sha256(args.scorer)
    if "actor" in args.arms:
        report["artifacts_before_run"][args.model_dir + "/actor.th"] = sha256(Path(args.model_dir) / "actor.th")
    (out / "frozen_config.json").write_text(json.dumps(report, indent=2) + "\n")
    for arm in args.arms:
        for seed in args.seeds:
            policy, artifact = make_policy(arm, args)
            result = episode(policy, seed, args.steps, out / f"{arm}_{seed}.jsonl")
            result["artifact_sha256"] = artifact
            report["cells"].setdefault(arm, {})[str(seed)] = result
            (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"arm={arm} seed={seed} return={result['return']:+.0f}", flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
