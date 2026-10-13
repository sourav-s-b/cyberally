"""Requested actions must be joined to observed outcomes without truth leakage."""

from types import SimpleNamespace
import json
import hashlib

import pytest

from blue.training.inspection_yield_data import (
    AnalyseLedger, check_frozen_inputs, json_floats,
)


def env(tick, waiting, result, last_analysis, state):
    tracker = SimpleNamespace(
        last_result={"h": result}, last_analysis={"h": last_analysis},
        state={"h": state})
    return SimpleNamespace(_tick=tick,
                           _awaiting={"a": ("h", "Analyse")} if waiting else {},
                           trackers={"a": tracker})


def test_delayed_success_uses_completion_tick_and_observed_confirmation():
    ledger = AnalyseLedger()
    row = ledger.request("a", "h", 8, {"risk": .4}, "UNKNOWN", None)
    ledger.reconcile(env(9, True, None, None, "UNKNOWN"))
    assert row["outcome"] is None
    ledger.reconcile(env(10, False, ("Analyse", "TRUE"), 10, "CONFIRMED"))
    assert row["outcome"] == {
        "status": "success", "completion_tick": 10,
        "duration_ticks": 2, "post_tracker_state": "CONFIRMED",
        "new_confirmation": True,
    }
    assert not ledger.pending


def test_failure_does_not_refresh_age_and_horizon_is_censored():
    ledger = AnalyseLedger()
    row = ledger.request("a", "h", 8, {}, "UNKNOWN", 3)
    ledger.reconcile(env(9, False, ("Analyse", "FALSE"), 3, "UNKNOWN"))
    assert row["outcome"]["status"] == "failure"
    assert row["outcome"]["new_confirmation"] is False
    later = ledger.request("a", "h", 11, {}, "UNKNOWN", 3)
    ledger.finalize()
    assert later["outcome"]["status"] == "censored_at_episode_end"
    assert later["outcome"]["new_confirmation"] is None


def test_pending_identity_and_completion_stamp_are_checked():
    ledger = AnalyseLedger()
    ledger.request("a", "h", 8, {}, "UNKNOWN", None)
    with pytest.raises(RuntimeError, match="already"):
        ledger.request("a", "h", 9, {}, "UNKNOWN", None)
    with pytest.raises(RuntimeError, match="identity"):
        ledger.reconcile(SimpleNamespace(_tick=9,
            _awaiting={"a": ("other", "Analyse")}, trackers={}))
    with pytest.raises(RuntimeError, match="completion stamp"):
        ledger.reconcile(env(10, False, ("Analyse", "TRUE"), None, "CONFIRMED"))


def test_missingness_is_null_and_reserved_seeds_fail_before_artifact_loading(tmp_path):
    assert json_floats([0., float("nan"), 1.]) == [0., None, 1.]
    with pytest.raises(ValueError, match="infinite"):
        json_floats([float("inf")])
    with pytest.raises(ValueError, match="reserved"):
        check_frozen_inputs([7809], tmp_path / "missing-plan.json",
                            tmp_path / "missing-scorer.pkl")


def test_only_frozen_training_and_development_seeds_are_allowed(tmp_path):
    scorer = tmp_path / "scorer.pkl"
    scorer.write_bytes(b"fixture")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"scorer_sha256": hashlib.sha256(b"fixture").hexdigest(),
                                "config": {"train_seeds": [7730],
                                           "eval_seeds": [8221],
                                           "env_kw": {"red_agent": "finite"}}}))
    assert check_frozen_inputs([7730, 8221], plan, scorer)["train_seeds"] == [7730]
    with pytest.raises(ValueError, match="frozen"):
        check_frozen_inputs([8222], plan, scorer)
