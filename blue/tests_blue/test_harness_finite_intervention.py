import json
from types import SimpleNamespace

import pytest

from blue.training.harness_finite_intervention import (
    InterventionHook, mine_targets, paired_prefix_check,
)


class Learner:
    def __init__(self, pick):
        self.pick = pick

    def __call__(self, env, agent, cands, scored):
        return self.pick


def target(**overrides):
    out = {"target_id": "m0-e7665-t12-blue_agent_1", "model_seed": 0,
           "source_episode_seed": 7665, "tick": 12,
           "agent": "blue_agent_1", "alternative_host": "h2"}
    out.update(overrides)
    return out


def test_lancer_deviation_only_changes_matching_decision_and_once():
    hook = InterventionHook(None, target(), "lancer_deviation", True)
    env = SimpleNamespace(_tick=11)
    assert hook(env, "blue_agent_1", ["h1", "h2"], [(2, "h1"), (1, "h2")]) == "h1"
    env._tick = 12
    assert hook(env, "blue_agent_0", ["h1", "h2"], [(2, "h1"), (1, "h2")]) == "h1"
    assert hook(env, "blue_agent_1", ["h1", "h2"], [(2, "h1"), (1, "h2")]) == "h2"
    assert hook(env, "blue_agent_1", ["h1", "h2"], [(2, "h1"), (1, "h2")]) == "h1"
    assert hook.outcome() == {"status": "changed", "reached": True,
                              "changed": True, "base_pick": "h1",
                              "intervention_pick": "h2"}


def test_lancer_deviation_reports_unavailable_host():
    hook = InterventionHook(None, target(), "lancer_deviation", True)
    result = hook(SimpleNamespace(_tick=12), "blue_agent_1", ["h1"], [(1, "h1")])
    assert result == "h1"
    assert hook.outcome()["status"] == "alternative_unavailable"


def test_actor_correction_substitutes_current_lancer_pick():
    hook = InterventionHook(Learner("h2"), target(), "actor_correction", True)
    result = hook(SimpleNamespace(_tick=12), "blue_agent_1", ["h1", "h2"],
                  [(2, "h1"), (1, "h2")])
    assert result == "h1"
    assert hook.outcome()["status"] == "changed"


def test_prefix_check_requires_only_target_agent_to_change_at_target(tmp_path):
    control = tmp_path / "control.jsonl"
    intervention = tmp_path / "intervention.jsonl"

    def row(tick, picked):
        agents = {"blue_agent_0": {"action": "Sleep", "host": None},
                  "blue_agent_1": {"action": "Analyse", "host": picked}}
        return json.dumps({"tick": tick, "agents": agents})

    control.write_text("\n".join([row(1, "a"), row(2, "b"), row(3, "c")]))
    intervention.write_text("\n".join([row(1, "a"), row(2, "x"), row(3, "c")]))
    assert paired_prefix_check(control, intervention, target(tick=1))["valid"]
    # An intervention can change episode termination after the target.
    intervention.write_text("\n".join([row(1, "a"), row(2, "x")]))
    assert paired_prefix_check(control, intervention, target(tick=1))["valid"]
    intervention.write_text("\n".join([row(1, "x"), row(2, "b"), row(3, "c")]))
    with pytest.raises(ValueError, match="diverged before or outside"):
        paired_prefix_check(control, intervention, target(tick=1))


def test_miner_uses_blue_candidate_rows_and_diversifies_source_episodes(tmp_path):
    audit_root = tmp_path / "audit"
    model = audit_root / "s0"
    model.mkdir(parents=True)
    report = {"red_agent": "finite", "cells": {"actor": {"7665": {"return": -300},
                                                               "7666": {"return": -200}}}}
    (model / "report.json").write_text(json.dumps(report))
    for seed in (7665, 7666):
        rows = []
        for tick in (4, 5):
            rows.append(json.dumps({"agents": {"blue_agent_1": {
                "candidate_audit": {"tick": tick, "actor_pick": f"a{tick}",
                    "lancer_pick": "top", "candidates": [
                        {"host": f"a{tick}", "lancer_rank": tick}]}}}}))
        (model / f"actor_{seed}.jsonl").write_text("\n".join(rows))
    for model_seed in range(1, 6):
        other = audit_root / f"s{model_seed}"
        other.mkdir()
        (other / "report.json").write_text(json.dumps(
            {"red_agent": "finite", "cells": {"actor": {}}}))

    selected = mine_targets(audit_root, 2)
    assert len(selected) == 2
    assert selected[0]["source_episode_seed"] == 7665
    assert selected[0]["tick"] == 5  # larger Lancer-rank loss breaks the tie
    assert selected[1]["source_episode_seed"] == 7666
