"""Episode-level splitting and strict separation of training/audit files."""

import json

import pytest

from blue.harness.scoring import sha256
from blue.training.inspection_yield_audit import audit


def fixture(tmp_path):
    row = {
        "request_id": "1:a:h", "request_tick": 1,
        "pre_input": {"feature_values": [None, 1.],
                      "event_memory_values": [0.]},
        "outcome": {"status": "success", "completion_tick": 3,
                    "new_confirmation": True},
    }
    request = tmp_path / "episode_10_requests.jsonl"
    truth = tmp_path / "episode_10_truth_audit.jsonl"
    request.write_text(json.dumps(row) + "\n")
    truth.write_text(json.dumps({"request_id": "1:a:h",
                                 "red_session_at_request": False}) + "\n")
    summary = {"manifest": {"schema": "test", "feature_names": ["x", "y"],
                            "history_names": ["m"]},
               "episodes": [{"seed": 10, "return": -3, "analyse_requests": 1,
                             request.name: sha256(request),
                             truth.name: sha256(truth)}]}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    return request, summary


def test_episode_partition_and_distinct_observed_truth(tmp_path):
    request, _ = fixture(tmp_path)
    report = audit(tmp_path, {"train": [10]})
    episode = report["partitions"]["train"]["episodes"][0]
    assert episode["new_confirmations"] == 1
    assert episode["confirmation_yield"] == 1.
    with pytest.raises(ValueError, match="multiple"):
        audit(tmp_path, {"train": [10], "validation": [10]})
    with pytest.raises(ValueError, match="exactly once"):
        audit(tmp_path, {"train": [11]})
    # If privileged truth is added to the training file, validation fails
    # even when the hash in the manifest has also been updated.
    row = json.loads(request.read_text())
    row["red_session_at_request"] = False
    request.write_text(json.dumps(row) + "\n")
    summary = json.loads((tmp_path / "summary.json").read_text())
    summary["episodes"][0][request.name] = sha256(request)
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="leaked"):
        audit(tmp_path, {"train": [10]})


def test_plan_enforces_cohort_roles(tmp_path):
    fixture(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"config": {"train_seeds": [10],
                                           "eval_seeds": [20]}}))
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["manifest"]["plan_sha256"] = sha256(plan)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="development cohort"):
        audit(tmp_path, {"audit": [10]}, plan)
