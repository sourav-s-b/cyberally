"""Validate inspection-yield records and freeze whole-episode partitions.

This checker does not fit a model. An ``audit`` partition means a separate
episode group within this *reused development cohort*, not untouched final
evaluation data. Host rows from one episode can never cross partitions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from blue.harness.scoring import sha256


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_episode(root, episode, feature_width, memory_width):
    seed = int(episode["seed"])
    rp = root / f"episode_{seed}_requests.jsonl"
    tp = root / f"episode_{seed}_truth_audit.jsonl"
    for path in (rp, tp):
        if sha256(path) != episode[path.name]:
            raise ValueError(f"artifact hash mismatch: {path}")
    requests, truth = read_jsonl(rp), read_jsonl(tp)
    if len(requests) != episode["analyse_requests"] or len(truth) != len(requests):
        raise ValueError("request/outcome/truth row count mismatch")
    rids = [r["request_id"] for r in requests]
    tids = [r["request_id"] for r in truth]
    if len(set(rids)) != len(rids) or len(set(tids)) != len(tids) or set(rids) != set(tids):
        raise ValueError("request and truth IDs do not join one-to-one")
    for row in requests:
        if "red_session_at_request" in row or "compromised_label" in row:
            raise ValueError("privileged truth leaked into training file")
        pre, outcome = row["pre_input"], row["outcome"]
        if (len(pre["feature_values"]) != feature_width or
                len(pre["event_memory_values"]) != memory_width):
            raise ValueError("feature geometry mismatch")
        for v in (*pre["feature_values"], *pre["event_memory_values"]):
            if v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v)):
                raise ValueError("invalid feature value")
        if outcome["status"] == "censored_at_episode_end":
            if outcome["new_confirmation"] is not None or outcome["completion_tick"] is not None:
                raise ValueError("censored result treated as observed label")
        elif outcome["status"] in ("success", "failure"):
            if outcome["completion_tick"] <= row["request_tick"]:
                raise ValueError("completion precedes request")
            if not isinstance(outcome["new_confirmation"], bool):
                raise ValueError("missing observed result label")
        else:
            raise ValueError("unknown action outcome")
    for row in truth:
        if (set(row) != {"request_id", "red_session_at_request"} or
                not isinstance(row["red_session_at_request"], bool)):
            raise ValueError("unexpected privileged audit shape")
    observed_counts = {status: sum(r["outcome"]["status"] == status
                                   for r in requests)
                       for status in ("success", "failure", "censored_at_episode_end")}
    if "outcomes" in episode and observed_counts != episode["outcomes"]:
        raise ValueError("episode outcome tallies mismatch")
    rates = [r["outcome"]["new_confirmation"] for r in requests
             if r["outcome"]["status"] == "success"]
    if "new_confirmations" in episode and sum(rates) != episode["new_confirmations"]:
        raise ValueError("episode confirmation tally mismatch")
    return {
        "seed": seed, "request_count": len(requests),
        "completed_success": len(rates),
        "new_confirmations": sum(rates),
        "confirmation_yield": sum(rates) / len(rates) if rates else None,
        "censored": sum(r["outcome"]["status"] == "censored_at_episode_end"
                        for r in requests),
        "return": episode["return"],
    }


def audit(root, partitions, plan_path=None):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    manifest = summary["manifest"]
    available = {int(e["seed"]) for e in summary["episodes"]}
    if len(available) != len(summary["episodes"]):
        raise ValueError("duplicate episode")
    seen = set()
    for name, seeds in partitions.items():
        if not seeds:
            raise ValueError(f"empty {name} partition")
        for seed in seeds:
            if seed in seen:
                raise ValueError("episode appears in multiple partitions")
            seen.add(seed)
    if seen != available:
        raise ValueError("partitions must assign every collected episode exactly once")
    if plan_path is not None:
        if sha256(plan_path) != manifest["plan_sha256"]:
            raise ValueError("frozen plan hash mismatch")
        plan = json.loads(Path(plan_path).read_text())["config"]
        training = set(plan["train_seeds"])
        development = set(plan["eval_seeds"])
        if not (set(partitions.get("train", ())) |
                set(partitions.get("validation", ()))) <= training:
            raise ValueError("training/validation episodes outside frozen training cohort")
        if not set(partitions.get("audit", ())) <= development:
            raise ValueError("audit episodes outside frozen development cohort")
    reports = [validate_episode(root, e, len(manifest["feature_names"]),
                                len(manifest["history_names"]))
               for e in summary["episodes"]]
    by_seed = {r["seed"]: r for r in reports}
    return {
        "source_summary_sha256": sha256(root / "summary.json"),
        "schema": manifest["schema"],
        "caveat": "selected actions only; reused development episodes; no policy-value claim",
        "partitions": {name: {
            "episode_seeds": seeds,
            "episodes": [by_seed[s] for s in seeds],
            "total_requests": sum(by_seed[s]["request_count"] for s in seeds),
            "total_new_confirmations": sum(by_seed[s]["new_confirmations"] for s in seeds),
        } for name, seeds in partitions.items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--train", type=int, nargs="+", required=True)
    p.add_argument("--validation", type=int, nargs="+", required=True)
    p.add_argument("--audit", type=int, nargs="+", required=True)
    args = p.parse_args()
    report = audit(args.input, {"train": args.train,
                                "validation": args.validation,
                                "audit": args.audit}, args.plan)
    dest = Path(args.input) / "episode_partitions.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"validated {sum(len(p['episodes']) for p in report['partitions'].values())} "
          f"episodes; wrote {dest}")


if __name__ == "__main__":
    main()
