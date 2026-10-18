"""Describe where finite-Red native-return differences come from.

This is a trace audit, not a causal estimator. Privileged labels are read here
only for post-episode evaluation summaries and never passed to policy code.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import t


def read_json(path):
    return json.loads(Path(path).read_text())


def read_episode(trace_path, expected_return):
    trace_path = Path(trace_path)
    truth_path = trace_path.with_suffix(".privileged.jsonl")
    actions, branches, completions = Counter(), Counter(), Counter()
    event_phase_subnet, lwf_context = Counter(), Counter()
    blue_by_tick, host_history = {}, {}
    confirmed = 0
    trace_return = 0.0
    ticks = 0
    for line in trace_path.open():
        row = json.loads(line)
        ticks += 1
        trace_return += float(row["reward"])
        blue_by_tick[int(row["tick"])] = row["agents"]
        for record in row["agents"].values():
            actions[record["action"]] += 1
            request = record.get("request") or {}
            branches[request.get("branch", "unrecorded")] += 1
            if request.get("host"):
                host_history.setdefault(request["host"], []).append(
                    (int(row["tick"]), "request", request.get("action"),
                     request.get("branch"), None))
            done = record.get("completion")
            if done:
                result = done.get("result")
                outcome = result[-1] if isinstance(result, list) and result else str(result)
                completions[f"{done['action']}:{outcome}"] += 1
                confirmed += done.get("belief_after") == "CONFIRMED"
                host_history.setdefault(done["host"], []).append(
                    (int(row["tick"]), "completion", done["action"], None,
                     done.get("belief_after")))

    event_penalties, event_counts = Counter(), Counter()
    compromised_host_ticks = 0
    new_compromised_hosts = set()
    max_compromised = 0
    owner_by_subnet = {
        "restricted_zone_a_subnet": "blue_agent_0",
        "operational_zone_a_subnet": "blue_agent_1",
        "restricted_zone_b_subnet": "blue_agent_2",
        "operational_zone_b_subnet": "blue_agent_3",
        "public_access_zone_subnet": "blue_agent_4",
        "admin_network_subnet": "blue_agent_4",
        "office_network_subnet": "blue_agent_4",
    }
    for line in truth_path.open():
        row = json.loads(line)
        before = set(row.get("compromised_before", []))
        after = set(row.get("compromised_after", []))
        compromised_host_ticks += len(after)
        max_compromised = max(max_compromised, len(after))
        new_compromised_hosts.update(after - before)
        for event in row.get("penalty_events", []):
            name = event.get("event", "unknown")
            event_penalties[name] += float(event.get("penalty", 0.0))
            event_counts[name] += 1
            host = event.get("host", "")
            subnet = (host.split("_subnet_")[0] + "_subnet"
                      if "_subnet_" in host else "unknown")
            event_phase_subnet[(name, event.get("phase"), subnet)] += float(
                event.get("penalty", 0.0))
            if name == "LWF":
                tick = int(row["tick"])
                history = host_history.get(host, [])
                previous = [item for item in history
                            if item[0] < tick and item[1] == "completion"]
                if previous:
                    last = previous[-1]
                    lag = tick - last[0]
                    lag_bucket = "1-5" if lag <= 5 else "6-20" if lag <= 20 else "21+"
                    previous_action = f"{last[2]}:{last[4] or 'none'}:{lag_bucket}"
                else:
                    previous_action = "none"
                owner = owner_by_subnet.get(subnet)
                same_tick = (blue_by_tick.get(tick, {}).get(owner, {})
                             if owner else {})
                same_request = same_tick.get("request") or {}
                same_completion = same_tick.get("completion") or {}
                if same_request.get("host") == host:
                    same_action = (f"request:{same_request.get('action')}"
                                   f":{same_request.get('branch')}")
                elif same_completion.get("host") == host:
                    same_action = (f"completion:{same_completion.get('action')}"
                                   f":{same_completion.get('belief_after')}")
                else:
                    same_action = "none"
                pending = same_tick.get("pending_after")
                pending_action = (pending[1] if pending and pending[0] == host
                                  else "none")
                compromised = str(host in after).lower()
                lwf_context[(event.get("phase"), subnet, previous_action,
                             same_action, pending_action, compromised)] += 1

    if ticks != 399:
        raise ValueError(f"unexpected episode length {ticks}: {trace_path}")
    if not np.isclose(trace_return, expected_return):
        raise ValueError(f"trace/report return mismatch: {trace_path}")
    if not np.isclose(sum(event_penalties.values()), trace_return):
        raise ValueError(f"penalty/event accounting mismatch: {trace_path}")
    return {
        "return": trace_return,
        "penalties": dict(event_penalties),
        "penalty_event_counts": dict(event_counts),
        "event_penalties_by_phase_subnet": {
            f"{event}|phase={phase}|{subnet}": float(value)
            for (event, phase, subnet), value in event_phase_subnet.items()},
        "lwf_event_context_counts": {
            "|".join(map(str, key)): int(value)
            for key, value in lwf_context.items()},
        "actions": dict(actions),
        "branches": dict(branches),
        "completions": dict(completions),
        "confirmed_completions": int(confirmed),
        "compromised_host_ticks": compromised_host_ticks,
        "new_compromised_hosts": len(new_compromised_hosts),
        "max_compromised_hosts": max_compromised,
    }


def load_arm(report_path, arm, trace_prefix):
    report = read_json(report_path)
    rows = report.get("cells", {}).get(arm, {})
    if not rows:
        raise ValueError(f"missing {arm} cells in {report_path}")
    result = {}
    for seed, cell in rows.items():
        trace = cell.get("blue_trace")
        if not trace:
            raise ValueError(f"missing trace for {arm} seed {seed}")
        trace_path = Path(trace)
        # Reports retain original absolute paths. Prefer the co-located trace
        # when the artifact tree has been copied to a new workspace.
        candidate = Path(report_path).parent / f"{trace_prefix}_{seed}.jsonl"
        if candidate.is_file():
            trace_path = candidate
        result[str(seed)] = read_episode(trace_path, float(cell["return"]))
    return result


def average(rows):
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row.get(key, 0) for row in rows])) for key in keys}


def mean_ci(values):
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    radius = float(t.ppf(.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    return {"n_training_seeds": len(x), "per_training_seed": x.tolist(),
            "mean": mean, "ci95": [mean - radius, mean + radius],
            "df": len(x) - 1}


def diagnose(artifact_root, guard_reports):
    root = Path(artifact_root)
    lancer = load_arm(root / "eval/lancer/report.json", "lancer", "lancer")
    guard = {}
    for report_path in guard_reports:
        guard.update(load_arm(report_path, "guard", "guard"))
    expected = set(lancer)
    if set(guard) != expected:
        raise ValueError("guard and Lancer evaluation seed sets differ")

    actor_reports, actors = {}, {}
    for seed in range(6):
        key = f"s{seed}"
        report_path = root / f"eval/actor_{key}/report.json"
        actor_reports[key] = read_json(report_path)
        actors[key] = load_arm(report_path, "actor", "actor")
        if set(actors[key]) != expected:
            raise ValueError(f"actor {key} seed set differs")

    def system_summary(cells):
        ordered = [cells[s] for s in sorted(cells, key=int)]
        return {
            "n_episodes": len(ordered),
            "mean_return": float(np.mean([r["return"] for r in ordered])),
            "worst_return": float(min(r["return"] for r in ordered)),
            "mean_penalties_per_episode": average([r["penalties"] for r in ordered]),
            "mean_penalty_event_counts_per_episode": average(
                [r["penalty_event_counts"] for r in ordered]),
            "mean_event_penalties_by_phase_subnet_per_episode": average(
                [r["event_penalties_by_phase_subnet"] for r in ordered]),
            "mean_lwf_event_context_counts_per_episode": average(
                [r["lwf_event_context_counts"] for r in ordered]),
            "mean_actions_per_episode": average([r["actions"] for r in ordered]),
            "mean_completions_per_episode": average([r["completions"] for r in ordered]),
            "mean_confirmed_completions": float(np.mean(
                [r["confirmed_completions"] for r in ordered])),
            "mean_compromised_host_ticks": float(np.mean(
                [r["compromised_host_ticks"] for r in ordered])),
            "mean_new_compromised_hosts": float(np.mean(
                [r["new_compromised_hosts"] for r in ordered])),
        }

    seed_differences = {"actor_minus_lancer": [], "actor_minus_guard": []}
    model_summaries = {}
    for model_seed, rows in actors.items():
        diffs = {}
        for comparison, baseline in (("lancer", lancer), ("guard", guard)):
            per_episode = {
                s: rows[s]["return"] - baseline[s]["return"] for s in sorted(expected, key=int)
            }
            diffs[comparison] = {
                "mean_paired_return_difference": float(np.mean(list(per_episode.values()))),
                "per_evaluation_seed": per_episode,
            }
            seed_differences[f"actor_minus_{comparison}"].append(
                diffs[comparison]["mean_paired_return_difference"])
        model_summaries[model_seed] = {
            "system": system_summary(rows), "paired_return_differences": diffs}

    lancer_guard = {
        s: guard[s]["return"] - lancer[s]["return"] for s in sorted(expected, key=int)
    }
    src_commits = sorted({
        read_json(p)["source_commit"] for p in guard_reports
    })
    return {
        "claim": "descriptive trace audit; paired differences are conditional on this fixed evaluation block",
        "artifact_root": str(root),
        "evaluation_seeds": sorted(map(int, expected)),
        "lancer": system_summary(lancer),
        "guard_only": system_summary(guard),
        "guard_minus_lancer_mean_paired_return": float(np.mean(list(lancer_guard.values()))),
        "actor_models": model_summaries,
        "actor_minus_lancer_training_seed_interval": mean_ci(seed_differences["actor_minus_lancer"]),
        "actor_minus_guard_training_seed_interval": mean_ci(seed_differences["actor_minus_guard"]),
        "guard_report_source_commits": src_commits,
        "actor_report_source_commits": sorted({
            report["source_commit"] for report in actor_reports.values()
        }),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True,
                        help="finite_red_native result directory")
    parser.add_argument("--guard-reports", nargs="+", required=True,
                        help="guard-only report.json files covering the same seeds")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = diagnose(args.artifact_root, args.guard_reports)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "lancer_mean": result["lancer"]["mean_return"],
        "guard_mean": result["guard_only"]["mean_return"],
        "guard_minus_lancer": result["guard_minus_lancer_mean_paired_return"],
        "actor_minus_lancer_ci95": result["actor_minus_lancer_training_seed_interval"]["ci95"],
        "actor_minus_guard_ci95": result["actor_minus_guard_training_seed_interval"]["ci95"],
    }, indent=2))


if __name__ == "__main__":
    main()
