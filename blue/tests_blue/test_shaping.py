"""Reward-shaping tests: pure bonus rule + constructor wiring.

Shaping is training-only scaffolding (eval stays native). Privileged true
state may inform REWARDS; only Blue-visible observations inform policies.
"""

import numpy as np
import pytest

import blue.core.wrapper as wrapper
from blue.core.wrapper import shaping_event_bonus, SHAPING_DEFAULTS


def test_bonus_clear_confirm_vandalism():
    assert shaping_event_bonus("Remove", True, False, False) == SHAPING_DEFAULTS["clear"]
    assert shaping_event_bonus("Restore", True, False, False) == SHAPING_DEFAULTS["clear"]
    assert shaping_event_bonus("Analyse", False, False, True) == SHAPING_DEFAULTS["confirm"]
    assert shaping_event_bonus("Remove", False, False, False) == SHAPING_DEFAULTS["vandalism"]
    assert shaping_event_bonus("Restore", False, True, False) == SHAPING_DEFAULTS["vandalism"]


def test_bonus_failed_clear_is_zero_not_punished():
    # Failed clears of privileged attackers are correct behavior
    # (escalation handles them); shaping must not punish them.
    assert shaping_event_bonus("Remove", True, True, False) == 0.0
    assert shaping_event_bonus("Restore", True, True, False) == 0.0
    assert shaping_event_bonus("Analyse", False, False, False) == 0.0
    assert shaping_event_bonus("Sleep", False, False, False) == 0.0
    assert shaping_event_bonus("Monitor", True, True, True) == 0.0


def test_shaping_constructor_validation():
    env = wrapper.CC4MARLEnv(shaping=True)
    assert env.shaping == SHAPING_DEFAULTS
    assert env.get_env_info()["shaping"] == SHAPING_DEFAULTS
    env2 = wrapper.CC4MARLEnv()
    assert env2.shaping is None
    assert env2.get_env_info()["shaping"] is None
    with pytest.raises(ValueError):
        wrapper.CC4MARLEnv(shaping={"bogus": 1.0})


def test_shaping_short_rollout_wiring():
    # Exercises the per-tick privileged scans + completion hook without
    # needing any action to complete (Sleep never completes anything).
    for shaping in (None, True):
        env = wrapper.CC4MARLEnv(seed=7629, steps=5, shaping=shaping)
        env.reset(seed=7629)
        total = 0.0
        for _ in range(5):
            _, rewards, terminated, truncated, _ = env.step(
                {a: 0 for a in wrapper.BLUE_AGENTS})
            total += float(rewards[0]) if isinstance(rewards, list) else float(rewards)
            if terminated or truncated:
                break
        assert np.isfinite(total)
