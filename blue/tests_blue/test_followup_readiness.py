"""Preparation must never accept missing episodes or mismatched provenance."""
import copy
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "readiness", Path(__file__).resolve().parents[2] / "scripts/blue_followup_readiness.py")
readiness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(readiness)


def report():
    return {"task": "bc:s0", "seeds": [1, 2], "frozen_plan_sha256": "plan",
            "actor_sha256": "actor", "model_manifest_sha256": "manifest",
            "cells": {"1": {"seed": 1, "return": -3.0},
                      "2": {"seed": 2, "return": -5.0}}}


def check(value):
    return readiness.check_report(value, "bc:s0", [1, 2], "plan",
                                  {"actor_sha256": "actor", "manifest_sha256": "manifest"})


def test_complete_and_partial_progress():
    assert check(report()) == ([], 2)
    partial = report()
    del partial["cells"]["2"]
    assert check(partial) == ([], 1)


def test_reject_identity_drift():
    for field in ("task", "seeds", "frozen_plan_sha256", "actor_sha256",
                  "model_manifest_sha256"):
        value = report()
        value[field] = "changed"
        assert check(value)[0], field


def test_reject_extra_or_misidentified_episode():
    value = report()
    value["cells"]["3"] = {"seed": 3, "return": -1}
    assert check(value)[0]
    value = report()
    value["cells"]["1"]["seed"] = 2
    assert check(value)[0]


def test_reject_invalid_returns_and_cells():
    for invalid in (None, float("nan"), float("inf"), True, "-3"):
        value = copy.deepcopy(report())
        value["cells"]["1"]["return"] = invalid
        assert check(value)[0]
    value = report()
    value["cells"] = []
    assert check(value)[0]


def test_disk_artifacts_and_incomplete_run(tmp_path):
    def write(rel, value):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return readiness.digest(path)

    source_hash = write("blue/training/harness_dagger_confirm.py", "source")
    training_hash = write("training/frozen_plan.json", {})
    manifest_hash = write("training/models/bc/s0/manifest.json", {"status": "complete"})
    actor_hash = write("training/models/bc/s0/actor.pt", "weights")
    scorer_hash = write("scorer.pkl", "scorer")
    plan = {"training_source_files": {}, "confirmation_source_sha256": source_hash,
            "training_artifacts": "training", "training_frozen_plan_sha256": training_hash,
            "scorer_sha256": scorer_hash, "evaluation_seeds": [1, 2], "controls": [],
            "models": {"bc:s0": {"manifest_sha256": manifest_hash, "actor_sha256": actor_hash}}}
    plan_hash = write("confirm/frozen_plan.json", plan)
    config = {"confirmation": "confirm", "tracks": {}}
    result = readiness.inspect(tmp_path, config, tmp_path / "scorer.pkl")
    assert not result["errors"] and not result["confirmation_ready_for_analysis"]
    value = report()
    value.update(frozen_plan_sha256=plan_hash, actor_sha256=actor_hash,
                 model_manifest_sha256=manifest_hash)
    write("confirm/evaluation/bc/s0.json", value)
    assert readiness.inspect(tmp_path, config, tmp_path / "scorer.pkl")["confirmation_ready_for_analysis"]
    write("training/models/bc/s0/actor.pt", "changed weights")
    result = readiness.inspect(tmp_path, config, tmp_path / "scorer.pkl")
    assert result["errors"] and not result["confirmation_ready_for_analysis"]
