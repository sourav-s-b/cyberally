import json
from pathlib import Path

from blue.training.harness_specialist_c0 import screen_results


CONFIG = Path(__file__).resolve().parents[1] / "configs" / \
    "specialist-c0-audit-20261018.json"


def _episodes():
    return [{
        "coverage_miss": False,
        "recovery_false_completions_with_options": 0,
        "unresolved_verification_completions": 0,
        "unresolved_verification_with_multiple_choices": 0,
    } for _ in range(16)]


def test_c0_screen_requires_repeated_actionable_evidence():
    criteria = json.loads(CONFIG.read_text())["screening_criteria"]
    episodes = _episodes()
    for i in range(8):
        episodes[i]["unresolved_verification_with_multiple_choices"] = 4
    result = screen_results(episodes, criteria)
    gate = result["gates"]["verification_opportunity"]
    assert gate["episodes"] == 8
    assert gate["events"] == 32
    assert gate["qualifies_for_followup"] is True
    assert result["decision"].startswith("candidate signal only")


def test_unresolved_belief_without_alternative_choices_does_not_qualify():
    criteria = json.loads(CONFIG.read_text())["screening_criteria"]
    episodes = _episodes()
    # The caller records only unresolved results with >=2 eligible targets.
    # A high unresolved count outside that actionable subset is not a gate.
    for episode in episodes:
        episode["unresolved_verification_completions"] = 20
    result = screen_results(episodes, criteria)
    assert result["qualifying_followup_targets"] == []
    assert "stop before specialist training" in result["decision"]
