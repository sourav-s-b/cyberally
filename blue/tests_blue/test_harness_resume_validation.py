import copy

import pytest

from blue.training.harness_rl import validate_resume


def rows():
    config = {"seed": 3, "iters": 4, "eps_per_iter": 2, "steps": 400,
              "hidden": 64, "lr": 0.0003, "bonus": 0.25, "temp": 0.5,
              "max_age": 80, "ml_inputs": "both", "train_seeds": [11, 12],
              "red_agent": "finite", "state_shaping": "off",
              "state_shaping_weight": 0.05, "state_shaping_cap": 20,
              "diagnostics": False}
    manifest = {"config": config, "source_commit": "commit-a",
                "source_files": {"trainer.py": "hash-a"},
                "scorer_sha256": "scorer-a", "objective": "native",
                "red_agent": "finite", "state_shaping": {"term": None},
                "env_kw": {"red_agent": "finite"}, "runtime": {"python": "3.12"},
                "status": "started"}
    current = copy.deepcopy(manifest)
    current["config"]["resume"] = True
    ck = {"history": [{"iteration": 1}]}
    return manifest, current, ck


def test_interrupted_checkpoint_resumes_and_history_is_checkpoint_authoritative():
    old, current, ck = rows()
    assert validate_resume(old, current, ck) == ck["history"]
    old["history"] = ck["history"]
    assert validate_resume(old, current, ck) == ck["history"]


@pytest.mark.parametrize("field,value", [
    ("red_agent", "random"),
    ("source_files", {"trainer.py": "hash-b"}),
    ("objective", "shaped"),
])
def test_resume_rejects_attacker_source_and_objective_drift(field, value):
    old, current, ck = rows()
    old[field] = value
    with pytest.raises(ValueError, match="drift"):
        validate_resume(old, current, ck)


def test_resume_rejects_missing_or_complete_checkpoint():
    old, current, ck = rows()
    with pytest.raises(ValueError, match="checkpoint is missing"):
        validate_resume(old, current, ck, checkpoint_exists=False)
    old["status"] = "complete"
    with pytest.raises(ValueError, match="completed run"):
        validate_resume(old, current, ck)
