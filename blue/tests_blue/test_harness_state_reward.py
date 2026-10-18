import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from blue.harness.state_reward import TERMS, confirm_debt, make_reward_hook

ROOT = Path(__file__).resolve().parents[2]


def test_training_reward_has_no_privileged_compromise_access():
    source = (ROOT / "blue/harness/state_reward.py").read_text()
    assert "_compromised_set" not in source
    assert "true_compromised" not in source
    assert TERMS == ("confirm",)


def test_confirm_debt_uses_only_blue_tracker(monkeypatch):
    from blue.harness import state_reward
    monkeypatch.setattr(state_reward, "zone_severity", lambda host, phase: 0.5)
    env = SimpleNamespace(
        _tick=10,
        trackers={"blue_agent_0": SimpleNamespace(state={
            "h0": "CONFIRMED", "h1": "VERIFY", "h2": "UNKNOWN"})},
    )
    assert confirm_debt(env, 400) == pytest.approx(1.0)


def test_hook_is_native_by_default_and_optional_penalty_is_blue_visible():
    hook = make_reward_hook(term="confirm", weight=0.25, horizon=400)
    env = SimpleNamespace(_tick=1, trackers={})
    assert hook(env=env, tick=1, native=-12.0) == -12.0
    with pytest.raises(ValueError, match="unknown shaping term"):
        make_reward_hook(term="detection")


def test_eval_and_policy_inputs_do_not_import_privileged_shaping():
    eval_source = (ROOT / "blue/training/harness_eval.py").read_text()
    for path in ("blue/harness/features.py", "blue/harness/policy.py"):
        source = (ROOT / path).read_text()
        assert "state_reward" not in source
        assert "_compromised_set" not in source
    assert "state_reward" not in eval_source
    assert "native only" in eval_source


def test_harness_rl_unshaped_path_is_native():
    source = (ROOT / "blue/training/harness_rl.py").read_text()
    tree = ast.parse(source)
    assert any(isinstance(n, ast.FunctionDef) and n.name == "validate_resume"
               for n in ast.walk(tree))
    assert 'args.state_shaping == "off"' in source
