"""Decision-audit calculations and the final-seed boundary."""

import pytest

from blue.training.input_decision_audit import run, summarize


def test_candidate_ranking_uses_only_legal_candidates_in_each_cohort():
    rows = [{
        "tick": 7, "chosen_host": "clean_seen",
        "candidates": [
            {"host": "clean_seen", "risk": .9, "base_score": 3.,
             "visible": {"analysis_available": 1.}, "compromised_label": False},
            {"host": "bad_new", "risk": .7, "base_score": 1.,
             "visible": {"analysis_available": 0.}, "compromised_label": True},
            {"host": "clean_new", "risk": .2, "base_score": 2.,
             "visible": {"analysis_available": 0.}, "compromised_label": False},
        ],
    }]
    stats = summarize(rows)["early"]
    assert stats["all"]["risk_top1_on_positive_sets"] == 0.
    assert stats["never_inspected"]["risk_top1_on_positive_sets"] == 1.
    assert stats["never_inspected"]["random_top1_on_positive_sets"] == .5
    assert stats["never_inspected"]["actual_choice_top1_on_positive_sets"] == 0.
    assert stats["previously_inspected"]["risk_top1_on_positive_sets"] is None


def test_reserved_final_seed_is_rejected_before_loading_artifacts(tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        run(tmp_path / "missing.pkl", [7809], tmp_path / "out",
            tmp_path / "missing-plan.json")
