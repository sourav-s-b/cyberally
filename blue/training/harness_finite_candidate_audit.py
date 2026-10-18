"""Summarize Blue-visible sweep ranks around finite-Red phase-1 LWF events.

This is a post-hoc diagnostic. Privileged event labels are read from the
separate evaluation trace and never enter policy selection.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def episode_audit(blue_path, truth_path, expected_return):
    blue_rows = read_jsonl(blue_path)
    truth_rows = read_jsonl(truth_path)
    if len(blue_rows) != len(truth_rows):
        raise ValueError(f"trace length mismatch: {blue_path}")
    observed_return = sum(float(row["reward"]) for row in blue_rows)
    if observed_return != float(expected_return):
        raise ValueError(f"return mismatch: {blue_path}")

    decisions = []
    totals = Counter()
    picked_lancer_ranks = []
    picked_rank_loss = []
    for row in blue_rows:
        agent = row["agents"]["blue_agent_1"]
        audit = agent.get("candidate_audit")
        if not audit:
            continue
        totals["audited_decisions"] += 1
        totals["age_guard_interventions"] += int(audit["guard_intervened"])
        if audit.get("actor_pick") is None:
            continue
        totals["actor_ranked_decisions"] += 1
        totals["actor_differs_lancer"] += int(not audit["actor_equals_lancer"])
        totals["actor_differs_guard"] += int(not audit["actor_equals_guard"])
        rank_by_host = {c["host"]: c["lancer_rank"] for c in audit["candidates"]}
        selected_rank = rank_by_host.get(audit["actor_pick"])
        if selected_rank is not None:
            picked_lancer_ranks.append(selected_rank)
            picked_rank_loss.append(selected_rank - 1)
        decisions.append(audit)

    event_rows = []
    for truth in truth_rows:
        tick = truth["tick"]
        for event in truth.get("penalty_events", []):
            host = event.get("host", "")
            if event.get("event") != "LWF" or event.get("phase") != 1 \
                    or not host.startswith("operational_zone_a_"):
                continue
            preceding = [a for a in decisions if a["tick"] <= tick and
                         any(c["host"] == host for c in a["candidates"])]
            nearest = preceding[-1] if preceding else None
            relation = {"tick": tick, "host": host, "penalty": event["penalty"],
                        "candidate_seen_before": nearest is not None}
            if nearest is not None:
                candidate = next(c for c in nearest["candidates"] if c["host"] == host)
                relation.update({
                    "candidate_tick": nearest["tick"],
                    "ticks_since_candidate": tick - nearest["tick"],
                    "actor_pick": nearest["actor_pick"],
                    "lancer_pick": nearest["lancer_pick"],
                    "guard_pick": nearest["guard_pick"],
                    "actor_picked_event_host": nearest["actor_pick"] == host,
                    "lancer_picked_event_host": nearest["lancer_pick"] == host,
                    "lancer_rank_of_event_host": candidate["lancer_rank"],
                    "actor_rank_of_event_host": candidate.get("actor_rank"),
                    "actor_diverted_from_lancer": not nearest["actor_equals_lancer"],
                })
            event_rows.append(relation)

    known = [e for e in event_rows if e["candidate_seen_before"]]
    return {
        "return": observed_return,
        "candidate_decisions": dict(totals),
        "mean_lancer_rank_of_actor_pick": (
            sum(picked_lancer_ranks) / len(picked_lancer_ranks)
            if picked_lancer_ranks else None),
        "mean_rank_loss_vs_lancer_top": (
            sum(picked_rank_loss) / len(picked_rank_loss)
            if picked_rank_loss else None),
        "phase1_op_zone_a_lwf_events": len(event_rows),
        "events_with_prior_candidate_decision": len(known),
        "events_actor_picked_event_host": sum(e.get("actor_picked_event_host", False) for e in known),
        "events_lancer_picked_event_host": sum(e.get("lancer_picked_event_host", False) for e in known),
        "events_actor_diverted_from_lancer": sum(e.get("actor_diverted_from_lancer", False) for e in known),
        "mean_lancer_rank_of_event_host": (
            sum(e["lancer_rank_of_event_host"] for e in known) / len(known)
            if known else None),
        "mean_actor_rank_of_event_host": (
            sum(e["actor_rank_of_event_host"] for e in known
                if e["actor_rank_of_event_host"] is not None) /
            sum(e["actor_rank_of_event_host"] is not None for e in known)
            if any(e["actor_rank_of_event_host"] is not None for e in known) else None),
        "event_context": event_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True,
                        help="directory containing s0..s5 reports and traces")
    parser.add_argument("--expected-seeds", nargs="+", type=int,
                        default=list(range(7665, 7689)))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.eval_root)
    all_models = {}
    for model_seed in range(6):
        model_dir = root / f"s{model_seed}"
        report_path = model_dir / "report.json"
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        report = json.loads(report_path.read_text())
        cells = report.get("cells", {}).get("actor", {})
        if set(cells) != set(map(str, args.expected_seeds)):
            raise ValueError(f"incomplete or mismatched evaluation seed list: {report_path}")
        episodes = {}
        for seed in args.expected_seeds:
            key = str(seed)
            episodes[key] = episode_audit(
                model_dir / f"actor_{seed}.jsonl",
                model_dir / f"actor_{seed}.privileged.jsonl",
                cells[key]["return"])
        all_models[f"s{model_seed}"] = {
            "n_episodes": len(episodes),
            "mean_return": sum(e["return"] for e in episodes.values()) / len(episodes),
            "mean_candidate_decisions": sum(
                e["candidate_decisions"].get("audited_decisions", 0)
                for e in episodes.values()) / len(episodes),
            "mean_actor_ranked_decisions": sum(
                e["candidate_decisions"].get("actor_ranked_decisions", 0)
                for e in episodes.values()) / len(episodes),
            "mean_actor_differs_lancer": sum(
                e["candidate_decisions"].get("actor_differs_lancer", 0)
                for e in episodes.values()) / len(episodes),
            "mean_actor_differs_guard": sum(
                e["candidate_decisions"].get("actor_differs_guard", 0)
                for e in episodes.values()) / len(episodes),
            "mean_guard_interventions": sum(
                e["candidate_decisions"].get("age_guard_interventions", 0)
                for e in episodes.values()) / len(episodes),
            "mean_lancer_rank_of_actor_pick": sum(
                (e["mean_lancer_rank_of_actor_pick"] or 0) *
                e["candidate_decisions"].get("actor_ranked_decisions", 0)
                for e in episodes.values()) / sum(
                    e["candidate_decisions"].get("actor_ranked_decisions", 0)
                    for e in episodes.values()),
            "mean_rank_loss_vs_lancer_top": sum(
                (e["mean_rank_loss_vs_lancer_top"] or 0) *
                e["candidate_decisions"].get("actor_ranked_decisions", 0)
                for e in episodes.values()) / sum(
                    e["candidate_decisions"].get("actor_ranked_decisions", 0)
                    for e in episodes.values()),
            "phase1_op_zone_a_lwf_events": sum(
                e["phase1_op_zone_a_lwf_events"] for e in episodes.values()),
            "events_with_prior_candidate_decision": sum(
                e["events_with_prior_candidate_decision"] for e in episodes.values()),
            "events_actor_picked_event_host": sum(
                e["events_actor_picked_event_host"] for e in episodes.values()),
            "events_lancer_picked_event_host": sum(
                e["events_lancer_picked_event_host"] for e in episodes.values()),
            "events_actor_diverted_from_lancer": sum(
                e["events_actor_diverted_from_lancer"] for e in episodes.values()),
            "mean_lancer_rank_of_event_host": sum(
                (e["mean_lancer_rank_of_event_host"] or 0) *
                e["events_with_prior_candidate_decision"]
                for e in episodes.values()) / sum(
                    e["events_with_prior_candidate_decision"]
                    for e in episodes.values()),
            "mean_actor_rank_of_event_host": sum(
                (e["mean_actor_rank_of_event_host"] or 0) *
                e["events_with_prior_candidate_decision"]
                for e in episodes.values()) / sum(
                    e["events_with_prior_candidate_decision"]
                    for e in episodes.values()),
            "episodes": episodes,
        }
    result = {
        "claim": "descriptive candidate-choice audit; trajectories diverge and privileged LWF labels are post-hoc only",
        "eval_seeds": args.expected_seeds,
        "models": all_models,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {k: v for k, v in row.items()
                             if k not in ("episodes", "n_episodes")}
                      for name, row in all_models.items()}, indent=2))


if __name__ == "__main__":
    main()
