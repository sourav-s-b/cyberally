"""Red-variant config (proposal 15, phase 1 Blue-only part).

The wrapper's red behavior is a validated config flag, defaulting to the
historical DiscoveryFSRed. Each named red must construct, reset, and step a
short episode; Sleep red must earn ~zero penalties (sanity that the flag
actually changes the adversary). Longer comparisons run through the pool.
"""
import pytest

import cc4_epymarl_wrapper as wrapper
from blue_baselines import SleepBaseline, run_episode


def test_unknown_red_rejected():
    with pytest.raises(ValueError, match="red_agent"):
        wrapper.CC4MARLEnv(steps=10, red_agent="nope")


def test_all_reds_step_short_episode():
    for name in ("discovery", "finite", "verbose", "random", "sleep"):
        env = wrapper.CC4MARLEnv(steps=10, red_agent=name)
        env.reset(seed=7629)
        assert env.red_agent == name
        info = env.get_env_info()
        assert info["red_agent"] == name
        result = run_episode(SleepBaseline(), seed=7629, steps=10,
                             red_agent=name)
        assert isinstance(result["cumulative_return"], float)
        env.close()


def test_sleep_red_earns_nothing():
    # No adversary activity: Sleep blue vs Sleep red over 30 steps.
    env = wrapper.CC4MARLEnv(steps=30, red_agent="sleep")
    env.reset(seed=7629)
    total = 0.0
    for _ in range(29):
        _, rewards, terminated, truncated, _ = env.step(
            {a: 0 for a in wrapper.BLUE_AGENTS})
        total += float(rewards[0])
        if terminated or truncated:
            break
    assert total == 0.0
    env.close()
