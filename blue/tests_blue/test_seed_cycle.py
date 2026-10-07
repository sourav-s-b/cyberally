"""Seed-cycling tests for multi-episode training resets.

EPyMARL's EpisodeRunner calls bare env.reset() every episode, so without a
cycle only the first episode carries an explicit seed and later episodes
continue the RNG stream unrecorded. seed_cycle=[...] consumes the next entry
per bare reset (wrapping around) and records every applied seed in
env.reset_seeds for the run manifest. Uses Sleep red/green for speed; reward
*values* are not asserted.
"""
import numpy as np
import pytest
import blue.core.wrapper as wrapper
from CybORG.Agents import SleepAgent


@pytest.fixture
def calm(monkeypatch):
    monkeypatch.setattr(wrapper, "DiscoveryFSRed", SleepAgent)
    monkeypatch.setattr(wrapper, "EnterpriseGreenAgent", SleepAgent)


def test_cycle_order_wraps(calm):
    env = wrapper.CC4MARLEnv(steps=3, seed_cycle=[11, 22])
    for _ in range(4):
        env.reset()
    assert env.reset_seeds == [11, 22, 11, 22]


def test_explicit_reset_does_not_advance_cycle(calm):
    env = wrapper.CC4MARLEnv(steps=3, seed_cycle=[11, 22])
    env.reset()
    env.reset(seed=99)
    env.reset()
    env.reset()
    assert env.reset_seeds == [11, 99, 22, 11]


def test_seed_setter_wins_once_without_advancing(calm):
    env = wrapper.CC4MARLEnv(steps=3, seed_cycle=[11, 22])
    env.reset()
    env.seed(77)
    env.reset()
    env.reset()
    assert env.reset_seeds == [11, 77, 22]


def test_invalid_cycle_rejected(calm):
    with pytest.raises(ValueError, match="seed_cycle"):
        wrapper.CC4MARLEnv(steps=3, seed_cycle=[])
    with pytest.raises(ValueError, match="seed_cycle"):
        wrapper.CC4MARLEnv(steps=3, seed_cycle=[11, "x"])
    with pytest.raises(ValueError, match="seed_cycle"):
        wrapper.CC4MARLEnv(steps=3, seed_cycle=[True])


def test_default_behavior_unchanged(calm):
    env = wrapper.CC4MARLEnv(steps=3)
    env.reset()
    env.reset()
    assert env.reset_seeds == [7629, None]


def test_cycled_episodes_are_reproducible(calm):
    first = wrapper.CC4MARLEnv(steps=3, seed_cycle=[41, 42])
    second = wrapper.CC4MARLEnv(steps=3, seed_cycle=[41, 42])
    for _ in range(2):
        obs_a, _ = first.reset()
        obs_b, _ = second.reset()
        for a, b in zip(obs_a, obs_b):
            np.testing.assert_array_equal(a, b)
    assert first.reset_seeds == second.reset_seeds == [41, 42]
