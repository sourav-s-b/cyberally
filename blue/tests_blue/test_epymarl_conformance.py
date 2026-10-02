"""EPyMARL runner-conformance tests (no EPyMARL install needed).

Covers the exact seams the pinned EpisodeRunner (uoe-agents/epymarl
@cbc38c09) touches: constructor kwargs, scalar common reward, tensor actions,
info["episode_limit"], and the close/render/seed/save_replay/get_stats
lifecycle. Uses Sleep red/green for speed; reward *values* are not asserted.
"""
import numpy as np
import pytest
import cc4_epymarl_wrapper as wrapper
from CybORG.Agents import SleepAgent


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(wrapper, "DiscoveryFSRed", SleepAgent)
    monkeypatch.setattr(wrapper, "EnterpriseGreenAgent", SleepAgent)
    env = wrapper.CC4MARLEnv(steps=30)
    env.reset()
    return env


def test_runner_kwargs_accepted_and_validated():
    env = wrapper.CC4MARLEnv(steps=30, common_reward=True,
                             reward_scalarisation="mean")
    assert env.common_reward and env.reward_scalarisation == "mean"
    with pytest.raises(ValueError, match="scalarisation"):
        wrapper.CC4MARLEnv(steps=30, reward_scalarisation="median")


def test_common_reward_scalar_matches_per_agent_entry(quiet):
    list_env, scalar_env = quiet, wrapper.CC4MARLEnv(steps=30, common_reward=True)
    scalar_env.reset(seed=7629)
    list_env.reset(seed=7629)
    for _ in range(4):
        actions = {a: 0 for a in wrapper.BLUE_AGENTS}
        _, rewards_list, _, _, _ = list_env.step(actions)
        _, reward_scalar, _, _, _ = scalar_env.step(actions)
        assert isinstance(reward_scalar, float)
        assert reward_scalar == float(rewards_list[0])
        assert len(set(rewards_list)) == 1


def test_episode_limit_flag_only_at_horizon_end(quiet):
    env = quiet
    _, _, _, _, info = env.step([0] * 5)
    assert info == {"episode_limit": False}
    while True:
        _, _, terminated, truncated, info = env.step([0] * 5)
        if terminated or truncated:
            break
    assert info == {"episode_limit": True}


def test_lifecycle_methods(quiet):
    assert quiet.get_stats() == {}
    quiet.render()
    quiet.close()
    with pytest.raises(NotImplementedError, match="replay"):
        quiet.save_replay()


def test_seed_setter_applies_on_next_reset(quiet):
    ref = wrapper.CC4MARLEnv(steps=30)
    ref.reset(seed=7629)
    quiet.seed(7629)
    quiet.reset()
    assert quiet.hostnames == ref.hostnames


def test_torch_tensor_joint_action():
    torch = pytest.importorskip("torch")
    env = wrapper.CC4MARLEnv(steps=30)
    env.reset(seed=42)
    actions = torch.zeros(1, 5, dtype=torch.long)
    obs, rewards, terminated, truncated, info = env.step(actions[0])
    assert len(obs) == 5 and set(info) == {"episode_limit"}
    assert not (terminated or truncated)
