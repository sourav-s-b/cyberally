"""Regression tests using real CybORG; privileged setup/labels stay here only."""
from copy import deepcopy
from unittest.mock import patch
import numpy as np
import pytest
import cc4_epymarl_wrapper as wrapper
from blue_action_masking import BlueZoneTracker
from CybORG.Agents import SleepAgent
from CybORG.Shared.Session import Session


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(wrapper, "DiscoveryFSRed", SleepAgent)
    monkeypatch.setattr(wrapper, "EnterpriseGreenAgent", SleepAgent)
    env = wrapper.CC4MARLEnv(steps=30)
    env.reset()
    return env


def action_index(env, agent, host, action):
    return 2 + 3 * env.hostnames[agent].index(host) + wrapper.ACTION_TEMPLATES.index(action)


def test_single_reset_seed_and_same_world():
    env = wrapper.CC4MARLEnv(steps=30)
    with patch.object(env.cyborg, "reset", wraps=env.cyborg.reset) as reset:
        first, _ = env.reset(seed=7629)
        assert reset.call_count == 1
    topology = deepcopy(env.hostnames)
    for agent in wrapper.BLUE_AGENTS:
        base = env.cyborg.get_observation(agent)
        np.testing.assert_equal(env.views[agent], {h: base[h] for h in env.hostnames[agent]})
    trajectory = [env.step([0]*5)[1] for _ in range(5)]
    second, _ = env.reset(seed=7629)
    assert env.hostnames == topology
    np.testing.assert_array_equal(first, second)
    assert trajectory == [env.step([0]*5)[1] for _ in range(5)]
    env.reset(seed=7630)
    assert env.hostnames != topology
    env.reset(seed=7629)
    with patch.object(env.cyborg, "reset", wraps=env.cyborg.reset) as reset:
        env.reset()
        assert reset.call_args.kwargs["seed"] is None


def test_full_hq_coverage_and_padding(quiet):
    env = quiet
    assert len(env.hostnames["blue_agent_4"]) > 16
    assert env.obs_size == 510 and env.n_actions == 155
    for i, agent in enumerate(wrapper.BLUE_AGENTS):
        count = len(env.hostnames[agent])
        assert env.get_host_presence(i).sum() == count
        assert np.all(env.get_obs_agent(i)[count*10:] == 0)
        assert np.all(env.get_avail_agent_actions(i)[2+3*count:] == 0)
        assert env.get_avail_agent_actions(i)[2+3*(count-1):2+3*count].all()


def test_small_capacity_rejected():
    env = wrapper.CC4MARLEnv(max_hosts=16, steps=30)
    with pytest.raises(ValueError, match="truncate"):
        env.reset()
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0]*5)


def test_pending_action_keeps_original_target(quiet):
    env = quiet
    agent = "blue_agent_0"
    first, second = env.hostnames[agent][:2]
    env.step({agent: action_index(env, agent, first, "Analyse")})
    assert env._awaiting[agent] == (first, "Analyse")
    assert env.get_avail_agent_actions(0).sum() == 1
    env.step({agent: action_index(env, agent, second, "Restore")})
    assert agent not in env._awaiting
    assert env.trackers[agent].last_analysis[first] == 2
    assert env.trackers[agent].last_remediation[second] is None
    assert env.get_avail_agent_actions(0).sum() > 1


def test_passive_event_on_other_host_during_analyse(quiet):
    env = quiet
    agent = "blue_agent_0"
    target, other = env.hostnames[agent][:2]
    env.env.state.hosts[other].events.process_creation.append(
        {"pid": 98765, "process_name": "observed-test-process"})
    env.step({agent: action_index(env, agent, target, "Analyse")})
    assert any(p["PID"] == 98765 for p in env.views[agent][other]["Processes"])
    assert env.observed_at[agent][other]["Processes"] == 1
    saved = deepcopy(env.views[agent][other]["Processes"])
    env.step([0]*5)
    assert env.views[agent][other]["Processes"] == saved
    assert env.observed_at[agent][other]["Processes"] == 1  # not a new event


def test_failed_action_releases_pending_without_erasing_belief(quiet):
    env = quiet
    agent = "blue_agent_0"
    host = env.hostnames[agent][0]
    tracker = env.trackers[agent]
    tracker.state[host] = "CONFIRMED"
    env.step({agent: action_index(env, agent, host, "Analyse")})
    # Inject a terminal failure at the observation boundary; success from passive
    # Monitor must not stand in for the action result.
    env._consume(agent, {"success": "FALSE"})
    assert agent not in env._awaiting and not tracker.pending_until
    assert tracker.state[host] == "CONFIRMED"


def test_no_timeout_based_unmasking(quiet):
    env = quiet
    env.step({"blue_agent_0": 2})
    env._tick = 2
    with pytest.raises(RuntimeError, match="overdue"):
        env._consume("blue_agent_0", {"success": "IN_PROGRESS"})
    assert env.get_avail_agent_actions(0).sum() == 1


def test_confirmation_survives_first_empty_scan():
    tracker = BlueZoneTracker(["h"])
    tracker.note_analyse_result("h", {"Files": [{"Known File": "UNKNOWN"}]}, 0)
    for step in (1, 3):
        tracker.note_action_issued("h", "Analyse", 2, step)
        assert tracker.state["h"] == "CONFIRMED"
        tracker.note_analyse_result("h", {}, step+2)
        if step == 1:
            assert tracker.state["h"] == "CONFIRMED"
    assert tracker.state["h"] == "UNKNOWN"


def test_evidence_mask_explicit_and_restore_reachable(quiet):
    env = quiet
    host = env.hostnames["blue_agent_0"][0]
    assert env.get_avail_agent_actions(0)[4] == 1
    env.mask_mode = "evidence"
    assert env.get_avail_agent_actions(0)[4] == 0
    env.trackers["blue_agent_0"].note_analyse_result(
        host, {"Files": [{"Known File": "UNKNOWN"}]}, 0)
    assert env.get_avail_agent_actions(0)[4] == 1
    # Agent-visible action validity still wins over belief.
    with patch.object(env.env, "get_action_space", wraps=env.env.get_action_space) as space:
        invalid = deepcopy(space("blue_agent_0"))
        invalid["session"][0] = False
        space.return_value = invalid
        assert env.get_avail_agent_actions(0).sum() == 1


@pytest.mark.parametrize("username,action,remaining", [
    ("ubuntu", "Remove", False), ("root", "Remove", True),
    ("root", "Restore", False)])
def test_live_remediation(quiet, username, action, remaining):
    env = quiet
    agent = "blue_agent_0"
    host = next(h for h in env.hostnames[agent] if "server_host" in h)
    state = env.env.state
    session = Session(ident=None, hostname=host, username=username,
                      agent="red_agent_1", pid=None, session_type="RedAbstractSession")
    state.add_session(session)
    # Make the process visible through the real passive monitoring path.
    state.hosts[host].events.process_creation.append(
        {"pid": session.pid, "process_name": "test-intrusion"})
    env.step([0]*5)
    idx = action_index(env, agent, host, action)
    duration = wrapper.CLASSES[action](session=0, agent=agent, hostname=host).duration
    reward_total = env.step({agent: idx})[1][0]
    for _ in range(duration-1):
        reward_total += env.step([0]*5)[1][0]
    assert env.trackers[agent].last_result[host] == (action, "TRUE")
    assert env.trackers[agent].state[host] == "VERIFY"
    present = any(s.hostname == host for s in state.sessions["red_agent_1"].values())
    assert present is remaining
    if action == "Restore":
        # Native Restore action cost is retained, once per team, in every member's
        # identical reward (the training adapter will select the common reward).
        assert reward_total == -1  # native cost is charged when requested


def test_latest_files_replaced_after_empty_analysis(quiet):
    env = quiet
    agent = "blue_agent_0"
    host = env.hostnames[agent][0]
    env.views[agent][host]["Files"] = [{"Known File": "UNKNOWN"}]
    env.trackers[agent].state[host] = "CONFIRMED"
    env.step({agent: 2})
    env.step([0]*5)
    assert env.views[agent][host]["Files"] == []
    assert env.trackers[agent].state[host] == "CONFIRMED"


def test_native_episode_boundary_and_invalid_actions(quiet):
    env = quiet
    while True:
        _, rewards, terminated, truncated, _ = env.step(np.array([-1]*5))
        assert len(set(rewards)) == 1
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert env._tick == env.episode_limit - 1  # preserve current native behavior
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0]*5)


def test_single_facade_matches_joint():
    joint = wrapper.CC4MARLEnv(steps=30)
    single = wrapper.CC4BlueWrapper(steps=30)
    joint.reset(seed=42)
    single.reset(seed=42)
    for action in (2, 5, 4, 5, 0, 0, 0):
        obs, rewards, term, trunc, _ = joint.step({"blue_agent_0": action})
        single_obs, mask, reward, done = single.step(action)
        np.testing.assert_array_equal(single_obs, obs[0])
        np.testing.assert_array_equal(mask, joint.get_avail_agent_actions(0))
        assert reward == rewards[0] and done == (term or trunc)
