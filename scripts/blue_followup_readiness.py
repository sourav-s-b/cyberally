#!/usr/bin/env python3
"""Read-only preparation check; stdlib only, no simulator imports or launches."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_report(report, task, seeds, plan_hash, model=None):
    errors = []
    expected = {str(s) for s in seeds}
    if report.get("task") != task or report.get("seeds") != seeds:
        errors.append("task or seed-list mismatch")
    if report.get("frozen_plan_sha256") != plan_hash:
        errors.append("frozen-plan identity mismatch")
    if model:
        for key, value in model.items():
            report_key = "model_manifest_sha256" if key == "manifest_sha256" else key
            if report.get(report_key) != value:
                errors.append(f"{report_key} mismatch")
    cells = report.get("cells", {})
    if not isinstance(cells, dict):
        return ["cells must be an object"], 0
    if set(cells) - expected:
        errors.append("unexpected episode seeds")
    for seed, row in cells.items():
        if not isinstance(row, dict):
            errors.append(f"invalid row {seed}")
            continue
        value = row.get("return")
        if (str(row.get("seed")) != seed or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value)):
            errors.append(f"invalid seed or return in row {seed}")
    return errors, len(expected & set(cells))


def inspect(root, config, scorer):
    out = root / config["confirmation"]
    plan_path = out / "frozen_plan.json"
    plan = json.loads(plan_path.read_text())
    plan_hash = digest(plan_path)
    errors, pending, counts = [], [], {}

    def verify(path, expected):
        if not path.is_file() or digest(path) != expected:
            errors.append(f"missing or changed artifact: {path}")

    for rel, sha in plan["training_source_files"].items():
        verify(root / rel, sha)
    verify(root / "blue/training/harness_dagger_confirm.py",
           plan["confirmation_source_sha256"])
    base = root / plan["training_artifacts"]
    verify(base / "frozen_plan.json", plan["training_frozen_plan_sha256"])
    if scorer is None:
        pending.append("scorer hash not checked; supply --scorer")
    else:
        verify(scorer, plan["scorer_sha256"])
    for task, model in plan["models"].items():
        arm, model_id = task.split(":")
        folder = base / "models" / arm / model_id
        verify(folder / "manifest.json", model["manifest_sha256"])
        verify(folder / "actor.pt", model["actor_sha256"])
        if (folder / "manifest.json").is_file():
            manifest = json.loads((folder / "manifest.json").read_text())
            if manifest.get("status") != "complete":
                errors.append(f"incomplete model: {task}")
    seeds = plan["evaluation_seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        errors.append("empty or duplicate evaluation seed list")
    tasks = [(t, "controls", t, None) for t in plan["controls"]]
    tasks += [(t, *t.split(":"), m) for t, m in plan["models"].items()]
    for task, arm, name, model in tasks:
        path = out / "evaluation" / arm / f"{name}.json"
        if not path.exists():
            counts[task] = 0
            pending.append(f"{task}: not started")
            continue
        try:
            report = json.loads(path.read_text())
        except json.JSONDecodeError:
            pending.append(f"{task}: unreadable snapshot; retry after writer finishes")
            counts[task] = 0
            continue
        problems, n = check_report(report, task, seeds, plan_hash, model)
        errors.extend(f"{task}: {p}" for p in problems)
        counts[task] = n
        if n != len(seeds):
            pending.append(f"{task}: {n}/{len(seeds)}")
    return {
        "confirmation_ready_for_analysis": not errors and not pending,
        "completed_episode_rows": sum(counts.values()),
        "expected_episode_rows": len(tasks) * len(seeds),
        "per_task": counts, "errors": errors, "pending": pending,
        "tracks": {name: {"launchable": False, "remaining": track["blockers"]}
                   for name, track in config["tracks"].items()},
        "note": "Read-only snapshot; readiness does not imply a positive result or authorize a launch. Track blockers are manually maintained draft requirements.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scorer", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / "blue/configs/followup-preparation.json").read_text())
    try:
        result = inspect(root, config, args.scorer)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc), "confirmation_ready_for_analysis": False}))
        return 2
    print(json.dumps(result, indent=2))
    return 2 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
