"""Collect Blue-visible decision context for the finite-Red specialist C0 screen.

This bounded diagnostic runs the frozen Lancer+age-80 policy. It does not
train, compare policies, or write privileged simulator labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from blue.core.baselines import decode_index
from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv
from blue.harness.features import BlueFeatures, NAMES, VERSION
from blue.harness.policy import HarnessPolicy
from blue.policies.ordered import completed_investigation_age_summary
from blue.training.harness_rl import check_seeds, env_kwargs


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "blue/configs/specialist-c0-audit-20261018.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _feature_row(row):
    return [float(value) if math.isfinite(float(value)) else None
            for value in row]


def _decision_record(policy, agent):
    decision = getattr(policy, "last_decision", None) or {}
    if decision.get("agent") not in (None, agent):
        raise RuntimeError("decision record is attributed to another agent")
    eligible = decision.get("eligible_choices")
    if eligible is None:
        raise RuntimeError("choice audit is enabled but eligible choices are absent")
    return {
        "branch": decision.get("branch"),
        "selection_rule": decision.get("selection_rule"),
        "selected_action": decision.get("selected_action"),
        "selected_host": decision.get("selected_host"),
        "eligible_choices": [dict(choice) for choice in eligible],
        "guard_record": decision.get("guard_record"),
    }


def _upstream_executed_action(env, agent):
    """Read the action object accepted by the wrapper's simulator call."""
    actions = env.env.action.get(agent) or []
    if not actions:
        return {"action": None, "host": None}
    action = actions[0]
    return {"action": type(action).__name__,
            "host": getattr(action, "hostname", None)}


def collect_episode(seed: int, steps: int, red_agent: str, trace_path: Path):
    env = CC4MARLEnv(seed=seed, steps=steps, **env_kwargs(red_agent))
    env.reset(seed=seed)
    policy = HarnessPolicy(max_age=80, record_choices=True)
    policy.reset()
    pipe = BlueFeatures()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    pending_ids: dict[str, str] = {}
    request_meta: dict[str, dict] = {}
    branch_counts = Counter()
    recovery_false = 0
    recovery_false_with_options = 0
    unresolved_verify = 0
    unresolved_verify_with_options = 0
    total_return = 0.0
    ticks = 0
    terminal = truncated = False

    with trace_path.open("x", encoding="utf-8") as log:
        for _ in range(steps):
            rows, actions = {}, {}
            previous_pending = dict(env._awaiting)
            pending_ids_before = dict(pending_ids)
            for agent in BLUE_AGENTS:
                feature_matrix = pipe.rows(env, agent)
                valid_mask = env.get_avail_agent_actions(
                    BLUE_AGENTS.index(agent))
                action_index = int(policy.select(env, agent))
                if action_index < 0 or action_index >= len(valid_mask) \
                        or not valid_mask[action_index]:
                    raise RuntimeError("policy proposed an invalid action")
                action_name, host = decode_index(env, agent, action_index)
                decision = _decision_record(policy, agent)
                request_id = f"{seed}:{env._tick}:{agent}"
                feature_hosts = {choice.get("host")
                                 for choice in decision["eligible_choices"]}
                if host is not None:
                    feature_hosts.add(host)
                visible_features = {}
                for feature_host in sorted(h for h in feature_hosts if h):
                    if feature_host not in env.hostnames[agent]:
                        raise RuntimeError("choice refers to a host outside the agent view")
                    pos = env.hostnames[agent].index(feature_host)
                    visible_features[feature_host] = _feature_row(
                        feature_matrix[pos])
                choices = []
                for choice in decision["eligible_choices"]:
                    choice = dict(choice)
                    choice_host = choice.get("host")
                    choice["features"] = (visible_features.get(choice_host)
                                           if choice_host else None)
                    choices.append(choice)
                request = getattr(policy, "last_request", None) or {}
                proposal = request.get("proposal")
                if proposal != {"action": action_name, "host": host}:
                    raise RuntimeError("proposal differs from selected policy action")
                branch = decision["branch"] or request.get("branch", "unknown")
                branch_counts[branch] += 1
                request_meta[request_id] = {
                    "branch": branch,
                    "tick": int(env._tick),
                    "eligible_choices": choices,
                }
                rows[agent] = {
                    "request_id": request_id,
                    "observation_tick": int(env._tick),
                    "branch": branch,
                    "selection_rule": decision["selection_rule"],
                    "blue_feature_version": VERSION,
                    "blue_visible_features": visible_features,
                    "simulator_valid_action_indices": np.flatnonzero(
                        valid_mask).astype(int).tolist(),
                    "selected_action_index": action_index,
                    "eligible_choices": choices,
                    "manager_proposal": proposal,
                    "selected_action": {"action": action_name, "host": host},
                    "pending_before": previous_pending.get(agent),
                    "pending_request_id_before": pending_ids_before.get(agent),
                    "guard_record": decision["guard_record"],
                    "override": None,
                }
                actions[agent] = action_index

            _, reward, terminal, truncated, _ = env.step(actions)
            if isinstance(reward, (list, tuple, np.ndarray)):
                # The wrapper returns the same native team reward per agent
                # in non-common-reward mode; count it once.
                reward = reward[0]
            reward = float(reward)
            total_return += reward
            ticks += 1
            for agent in BLUE_AGENTS:
                record = rows[agent]
                selected = record["selected_action"]
                name, host = selected["action"], selected["host"]
                executed = _upstream_executed_action(env, agent)
                record["executed_action"] = executed
                if executed != selected:
                    record["override"] = {
                        "reason": "wrapper_or_simulator_action_differs_from_policy_selection",
                        "executed": executed,
                    }
                record["pending_after"] = env._awaiting.get(agent)
                record["completion"] = None
                before = previous_pending.get(agent)
                if before and agent not in env._awaiting:
                    completed_host, completed_action = before
                    request_id = pending_ids.pop(agent, None)
                    meta = request_meta.get(request_id, {})
                    next_matrix = pipe.rows(env, agent)
                    next_feature = _feature_row(
                        next_matrix[env.hostnames[agent].index(completed_host)])
                    last_result = env.trackers[agent].last_result.get(
                        completed_host)
                    result_value = (last_result[-1]
                                    if isinstance(last_result, (tuple, list))
                                    else last_result)
                    belief_after = env.trackers[agent].state.get(completed_host)
                    completion = {
                        "request_id": request_id,
                        "request_branch": meta.get("branch"),
                        "host": completed_host,
                        "action": completed_action,
                        "duration_ticks": int(env._tick - int(
                            request_meta[request_id]["tick"]
                            if request_id in request_meta else env._tick)),
                        "simulator_result": result_value,
                        "blue_belief_after": belief_after,
                        "next_blue_features": next_feature,
                    }
                    # The source request tick is linked by request_id, not by
                    # simulator-private state.
                    request_record = meta
                    completion["eligible_choice_count"] = len(
                        request_record.get("eligible_choices", []))
                    record["completion"] = completion
                    if completed_action in ("Remove", "Restore") \
                            and result_value == "FALSE":
                        recovery_false += 1
                        if len(request_record.get("eligible_choices", [])) >= 2:
                            recovery_false_with_options += 1
                    if meta.get("branch") == "verification" \
                            and belief_after in ("VERIFY", "UNKNOWN"):
                        unresolved_verify += 1
                        if len(request_record.get("eligible_choices", [])) >= 2:
                            unresolved_verify_with_options += 1
                # The wrapper accepts valid requests; track the request id for
                # an action still pending after this simulator step.
                if agent in env._awaiting and selected["action"] not in ("Sleep", "Monitor"):
                    if not previous_pending.get(agent):
                        pending_ids[agent] = record["request_id"]
                if agent not in env._awaiting and not record["completion"]:
                    pending_ids.pop(agent, None)

            log.write(json.dumps({
                "tick": int(env._tick),
                "agents": rows,
                "native_reward": reward,
                "terminated": bool(terminal),
                "truncated": bool(truncated),
            }, allow_nan=False) + "\n")
            if terminal or truncated:
                break

    coverage = {agent: completed_investigation_age_summary(env, agent)
                for agent in BLUE_AGENTS}
    has_coverage_miss = any(
        item["n_never_investigated"] > 0 for item in coverage.values())
    pending_at_end = {
        agent: {
            "pending": env._awaiting.get(agent),
            "request_id": pending_ids.get(agent),
            "request_branch": request_meta.get(
                pending_ids.get(agent), {}).get("branch"),
            "request_tick": request_meta.get(
                pending_ids.get(agent), {}).get("tick"),
        }
        for agent in BLUE_AGENTS if agent in env._awaiting
    }
    result = {
        "seed": seed,
        "red_agent": red_agent,
        "steps": ticks,
        "native_return": total_return,
        "terminated": bool(terminal),
        "truncated": bool(truncated),
        "branch_counts": dict(branch_counts),
        "coverage": coverage,
        "coverage_miss": has_coverage_miss,
        "recovery_false_completions": recovery_false,
        "recovery_false_completions_with_options": recovery_false_with_options,
        "unresolved_verification_completions": unresolved_verify,
        "unresolved_verification_with_multiple_choices": unresolved_verify_with_options,
        "pending_at_horizon": pending_at_end,
        "trace": str(trace_path),
        "trace_sha256": _sha256(trace_path),
    }
    env.close()
    return result


def screen_results(episodes: list[dict], criteria: dict) -> dict:
    n = len(episodes)
    coverage_n = sum(bool(e["coverage_miss"]) for e in episodes)
    recovery_n = sum(
        e["recovery_false_completions_with_options"] > 0 for e in episodes)
    verification_events = sum(
        e["unresolved_verification_with_multiple_choices"] for e in episodes)
    verification_episodes = sum(
        e["unresolved_verification_with_multiple_choices"] > 0
        for e in episodes)
    gates = {
        "coverage_miss": {
            "episodes": coverage_n,
            "required": criteria["coverage_miss"]["episodes_required"],
            "qualifies_for_followup": coverage_n >= criteria[
                "coverage_miss"]["episodes_required"],
        },
        "recovery_failure": {
            "episodes": recovery_n,
            "required": criteria["recovery_failure"]["episodes_required"],
            "qualifies_for_followup": recovery_n >= criteria[
                "recovery_failure"]["episodes_required"],
        },
        "verification_opportunity": {
            "episodes": verification_episodes,
            "events": verification_events,
            "episodes_required": criteria["verification_opportunity"][
                "episodes_required"],
            "events_required": criteria["verification_opportunity"][
                "minimum_events"],
            "qualifies_for_followup": (
                verification_episodes >= criteria[
                    "verification_opportunity"]["episodes_required"]
                and verification_events >= criteria[
                    "verification_opportunity"]["minimum_events"]),
        },
    }
    qualifying = [name for name, gate in gates.items()
                  if gate["qualifies_for_followup"]]
    return {
        "episodes": n,
        "gates": gates,
        "qualifying_followup_targets": qualifying,
        "decision": ("candidate signal only; freeze a separate objective before training"
                     if qualifying else
                     "no repeated actionable failure met the frozen screen; stop before specialist training"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--plan", action="store_true",
                        help="validate the frozen seed/config and print the plan without running episodes")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    seeds = [int(seed) for seed in config["seeds"]]
    check_seeds(seeds)
    from blue.training.seed_audit import reserved_conflicts
    conflicts = reserved_conflicts(min(seeds), max(seeds))
    if conflicts:
        raise SystemExit(f"seed block overlaps protected reservations: {conflicts}")
    if len(seeds) != len(set(seeds)) or seeds != list(range(seeds[0], seeds[-1] + 1)):
        raise SystemExit("C0 seed block must be unique and contiguous")
    out = args.out or (ROOT / "blue/results" / config["study_id"])
    if args.plan:
        print(json.dumps({"config": str(args.config), "seeds": seeds,
                          "episodes": len(seeds), "out": str(out),
                          "reserved_conflicts": conflicts}, indent=2))
        return
    if out.exists():
        raise FileExistsError(f"refusing to overwrite C0 artifacts: {out}")
    out.mkdir(parents=True)
    source_files = [
        args.config,
        Path(__file__),
        ROOT / "blue/harness/features.py",
        ROOT / "blue/harness/policy.py",
        ROOT / "blue/policies/ordered.py",
        ROOT / "blue/core/wrapper.py",
        ROOT / "blue/core/baselines.py",
        ROOT / "blue/training/harness_rl.py",
        ROOT / "blue/training/seed_audit.py",
    ]
    report = {
        "study_id": config["study_id"],
        "status": "running",
        "config": config,
        "config_sha256": _sha256(args.config),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "feature_names": list(NAMES),
        "feature_version": VERSION,
        "environment_kwargs": env_kwargs(config["red_agent"]),
        "action_mask_semantics": "simulator-valid action indices; distinct from branch candidate choices",
        "source_sha256": {str(path.relative_to(ROOT)): _sha256(path)
                           for path in source_files},
        "privileged_state_used": False,
        "episodes": {},
    }
    _write_json(out / "manifest.json", report)
    for seed in seeds:
        result = collect_episode(
            seed, int(config["steps"]), config["red_agent"],
            out / f"episode_{seed}.jsonl")
        report["episodes"][str(seed)] = result
        report["progress"] = {"completed": len(report["episodes"]),
                              "total": len(seeds), "last_seed": seed}
        _write_json(out / "manifest.json", report)
        print(f"C0 seed={seed} return={result['native_return']:+.0f} "
              f"branches={result['branch_counts']}", flush=True)
    report["screen"] = screen_results(
        list(report["episodes"].values()), config["screening_criteria"])
    report["status"] = "complete"
    _write_json(out / "report.json", report)
    _write_json(out / "manifest.json", report)
    print(json.dumps(report["screen"], indent=2), flush=True)


if __name__ == "__main__":
    main()
