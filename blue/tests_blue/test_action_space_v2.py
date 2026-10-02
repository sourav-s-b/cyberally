"""foundation-v2 per-agent bounds: widths, masks, equivalence, coercion.

Uses Sleep red/green for speed. The default (global-bound) mode is pinned by
test_foundation.py; here the per-agent mode must match the scenario-derived
bounds 17/17/17/17/51 with identical host indexing.
"""
import numpy as np
import pytest
import cc4_epymarl_wrapper as wrapper
import blue_baselines as baselines
from CybORG.Agents import SleepAgent


@pytest.fixture
def v2(monkeypatch):
    monkeypatch.setattr(wrapper, "DiscoveryFSRed", SleepAgent)
    monkeypatch.setattr(wrapper, "EnterpriseGreenAgent", SleepAgent)
    env = wrapper.CC4MARLEnv(steps=30, per_agent_bounds=True)
    env.reset()
    return env


def test_per_agent_widths_and_env_info(v2):
    assert v2.max_hosts_per_agent == [17, 17, 17, 17, 51]
    assert v2.n_actions_per_agent == [53, 53, 53, 53, 155]
    assert v2.obs_size_per_agent == [170, 170, 170, 170, 510]
    assert [len(v2.get_obs_agent(i)) for i in range(5)] == v2.obs_size_per_agent
    assert [len(v2.get_avail_agent_actions(i)) for i in range(5)] == v2.n_actions_per_agent
    assert len(v2.get_state()) == 1190 == v2.get_state_size()
    # Scalar accessors stay max-width for homogeneous consumers.
    assert (v2.obs_size, v2.n_actions) == (510, 155)
    info = v2.get_env_info()
    assert info["per_agent_bounds"] is True
    assert info["n_actions_per_agent"] == [53, 53, 53, 53, 155]
    assert info["wrapper_version"] == "foundation-v3"


def test_no_dead_slots_inside_live_range(v2):
    for i, agent in enumerate(wrapper.BLUE_AGENTS):
        count = len(v2.hostnames[agent])
        assert count <= v2.max_hosts_per_agent[i]
        assert v2.get_host_presence(i).sum() == count
        assert np.all(v2.get_obs_agent(i)[count * 10:] == 0)
        mask = v2.get_avail_agent_actions(i)
        assert np.all(mask[2 + 3 * count:] == 0)
        assert mask[2 + 3 * (count - 1):2 + 3 * count].all()


def test_out_of_agent_range_coerces_to_sleep(v2):
    # Index legal under the global width but outside agent 0's own 53 slots.
    v2.step({"blue_agent_0": 100})
    assert "blue_agent_0" not in v2._awaiting
    assert baselines.decode_index(v2, "blue_agent_0", 100) == ("Sleep", None)


def test_explicit_max_hosts_rejected_with_per_agent_bounds():
    with pytest.raises(ValueError, match="per_agent_bounds"):
        wrapper.CC4MARLEnv(steps=30, max_hosts=16, per_agent_bounds=True)


def test_facade_reports_own_agent_widths():
    narrow = wrapper.CC4BlueWrapper(blue_id="blue_agent_0", per_agent_bounds=True)
    wide = wrapper.CC4BlueWrapper(blue_id="blue_agent_4", per_agent_bounds=True)
    assert (narrow.max_hosts, narrow.n_actions, narrow.obs_size) == (17, 53, 170)
    assert (wide.max_hosts, wide.n_actions, wide.obs_size) == (51, 155, 510)


def test_masked_random_trajectory_matches_global_mode(monkeypatch):
    monkeypatch.setattr(wrapper, "DiscoveryFSRed", SleepAgent)
    monkeypatch.setattr(wrapper, "EnterpriseGreenAgent", SleepAgent)
    runs = []
    for kwargs in ({}, {"per_agent_bounds": True}):
        env = wrapper.CC4MARLEnv(steps=30, **kwargs)
        env.reset(seed=7629)
        policy = baselines.MaskedRandomBaseline(seed=11)
        rewards = [env.step({a: policy.select(env, a) for a in wrapper.BLUE_AGENTS})[1]
                   for _ in range(8)]
        runs.append(rewards)
    assert runs[0] == runs[1]


def test_unmasked_random_runs_under_per_agent_bounds(monkeypatch):
    monkeypatch.setattr(wrapper, "DiscoveryFSRed", SleepAgent)
    monkeypatch.setattr(wrapper, "EnterpriseGreenAgent", SleepAgent)
    result = baselines.run_episode(baselines.UnmaskedRandomBaseline(seed=5),
                                   seed=42, steps=10, snapshot_steps=(5,),
                                   per_agent_bounds=True)
    assert result["steps"] == 9
    assert len(result["trace"]) == 9 * 5
