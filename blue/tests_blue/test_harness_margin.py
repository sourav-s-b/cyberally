"""Behavioral checks for the margin-bounded deployment rule."""
import json
from types import SimpleNamespace

import numpy as np
import pytest


def test_bounded_choice_requires_margin_and_budget():
    from blue.training.harness_margin import bounded_choice
    assert bounded_choice(1.5, 1.0, 2, "teacher", "alt") == ("alt", True, 1)
    assert bounded_choice(1.0, 1.0, 2, "teacher", "alt") == ("alt", True, 1)
    assert bounded_choice(0.99, 1.0, 2, "teacher", "alt") == ("teacher", False, 2)
    assert bounded_choice(-0.4, 1.0, 2, "teacher", "alt") == ("teacher", False, 2)
    assert bounded_choice(0.5, 0.5, 2, "teacher", "alt") == ("alt", True, 1)
    assert bounded_choice(2.0, 1.0, 0, "teacher", "alt") == ("teacher", False, 0)
    assert bounded_choice(0.0, 0.0, 3, "teacher", "alt") == ("alt", True, 2)
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            bounded_choice(bad, 1.0, 2, "teacher", "alt")


def test_margin_actor_hook_validates_arguments():
    torch = pytest.importorskip("torch")
    from blue.training.harness_margin import MarginActorHook
    actor = torch.nn.Linear(2, 1)
    scorer = SimpleNamespace(reset=lambda: None)
    for mass in (0.0, 1.0, -0.2, 1.2):
        with pytest.raises(ValueError):
            MarginActorHook(actor, scorer, mass=mass, tau=1.0, budget=1)
    for tau in (None, -0.5, float("nan")):
        with pytest.raises(ValueError):
            MarginActorHook(actor, scorer, mass=0.5, tau=tau, budget=1)
    for budget in (-1, 1.5, "2"):
        with pytest.raises(ValueError):
            MarginActorHook(actor, scorer, mass=0.5, tau=1.0, budget=budget)


def test_hook_decision_path_budget_and_reset():
    pytest.importorskip("torch")
    import torch as th
    torch = th
    from blue.training.harness_margin import MarginActorHook

    th.manual_seed(0)
    # One-neuron head on a 4-wide feature row: residual = w . x, control it
    # by writing weights directly.
    actor = torch.nn.Linear(4, 1)
    with torch.no_grad():
        actor.weight.copy_(torch.tensor([[3.0, 0.0, 0.0, 0.0]]))
        actor.bias.copy_(torch.tensor([0.0]))
    scorer = SimpleNamespace(reset=lambda: None)
    hook = MarginActorHook(actor, scorer, mass=0.5, tau=1.0, budget=1)
    env = SimpleNamespace(_tick=7)
    teacher_row = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    alt_row = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    scored = [(1.0, "teacher_host"), (0.9, "alt_host")]
    cand_by_host = {"teacher_host": teacher_row, "alt_host": alt_row}

    def features(env_, agent_, cands):
        return np.stack([cand_by_host[h] for h in cands])

    hook.features = features
    hook.actor = actor

    picked = hook(env, "blue_agent_0", ["teacher_host", "alt_host"], scored)
    row = hook.decisions[-1]
    assert picked == "alt_host" and row["overrode"] is True
    assert hook.budget_left == 0 and hook.overrode == 1
    picked = hook(env, "blue_agent_1", ["teacher_host", "alt_host"], scored)
    assert picked == "teacher_host" and not hook.decisions[-1]["overrode"]
    assert hook.overrode == 1
    hook.reset()
    assert hook.budget_left == 1 and hook.decisions == []
    picked = hook(env, "blue_agent_1", ["teacher_host", "alt_host"], scored)
    row = hook.decisions[-1]
    assert picked == "alt_host" and row["overrode"] is True


def test_margin_profile_reports_distribution():
    from blue.training.harness_margin import margin_profile
    rows = [{"margin": 0.1}, {"margin": 0.5}, {"margin": 2.4},
            {"margin": 3.7}, {"margin": 1.0}]
    prof = margin_profile(rows)
    assert prof["n"] == 5
    assert prof["n_positive"] == 5
    assert prof["n_gt_2"] == 2
    assert prof["n_gt_3"] == 1
    assert prof["max"] == pytest.approx(3.7)
    with pytest.raises(ValueError):
        margin_profile([])
    for bad in (float("nan"), -np.inf, np.inf):
        with pytest.raises(ValueError):
            margin_profile([{"margin": bad}])


def test_load_margin_model_refuses_incomplete_or_foreign_arms(tmp_path):
    torch = pytest.importorskip("torch")
    from blue.training.harness_margin import load_margin_model
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"status": "paused", "config": {"policy_version": "x"}}))
    with pytest.raises(ValueError):
        load_margin_model(tmp_path, "unused.pkl", "current-residual-v1", None, 0.25)
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"status": "complete",
         "config": {"policy_version": "anchored-categorical-v1"}}))
    with pytest.raises(ValueError):
        load_margin_model(tmp_path, "unused.pkl", "current-residual-v1", None, 0.25)
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"status": "complete",
         "config": {"policy_version": "current-residual-v1"}}))
    with pytest.raises(FileNotFoundError):
        load_margin_model(tmp_path, "unused.pkl", "current-residual-v1", None, 0.25)
