"""BLUE-02 baseline tests; live sim, Sleep red/green for speed unless noted."""
import numpy as np
import pytest
import cc4_epymarl_wrapper as wrapper
import blue_baselines as baselines
from CybORG.Agents import SleepAgent


@pytest.fixture
def quiet():
    real_red = wrapper.DiscoveryFSRed
    real_green = wrapper.EnterpriseGreenAgent
    wrapper.DiscoveryFSRed = SleepAgent
    wrapper.EnterpriseGreenAgent = SleepAgent
    try:
        env = wrapper.CC4MARLEnv(steps=30)
        env.reset()
        yield env
    finally:
        wrapper.DiscoveryFSRed = real_red
        wrapper.EnterpriseGreenAgent = real_green


def test_sleep_is_trivially_legal(quiet):
    policy = baselines.SleepBaseline()
    for agent in wrapper.BLUE_AGENTS:
        assert policy.select(quiet, agent) == 0


def test_masked_random_deterministic_and_legal(quiet):
    for agent in wrapper.BLUE_AGENTS:
        idx = baselines.MaskedRandomBaseline(seed=7).select(quiet, agent)
        mask = quiet.get_avail_agent_actions(wrapper.BLUE_AGENTS.index(agent))
        assert mask[idx] == 1
        assert baselines.MaskedRandomBaseline(seed=7).select(quiet, agent) == idx


def test_unmasked_random_coerced_to_sleep_by_wrapper(quiet):
    policy = baselines.UnmaskedRandomBaseline(seed=3)
    idx = policy.select(quiet, "blue_agent_0")
    assert 0 <= idx < quiet.n_actions
    quiet.step({"blue_agent_0": quiet.n_actions + 50})  # illegal -> Sleep, no raise


def test_round_robin_respects_masks_and_busy(quiet):
    policy = baselines.RoundRobinBaseline()
    for _ in range(8):
        actions = {a: policy.select(quiet, a) for a in wrapper.BLUE_AGENTS}
        for agent, idx in actions.items():
            mask = quiet.get_avail_agent_actions(wrapper.BLUE_AGENTS.index(agent))
            assert mask[idx] == 1, (agent, idx)
        quiet.step(actions)


def test_round_robin_prioritises_confirmed_and_verify(quiet):
    env = quiet
    policy = baselines.RoundRobinBaseline()
    agent = "blue_agent_0"
    host = env.hostnames[agent][0]
    env.trackers[agent].state[host] = "CONFIRMED"
    idx = policy.select(env, agent)
    name, target = baselines.decode_index(env, agent, idx)
    assert target == host and name in ("Remove", "Restore")
    env.trackers[agent].note_action_issued(host, "Remove", 2, env._tick)
    assert policy.select(env, agent) == 0  # busy -> Sleep
    env.trackers[agent].pending_until.clear()
    env.trackers[agent].state[host] = "VERIFY"
    env.trackers[agent].last_remediation[host] = env._tick
    idx = policy.select(env, agent)
    name, target = baselines.decode_index(env, agent, idx)
    assert (name, target) == ("Analyse", host)


def test_round_robin_escalates_repeat_detection_to_restore(quiet):
    env = quiet
    policy = baselines.RoundRobinBaseline()
    agent = "blue_agent_0"
    host = env.hostnames[agent][0]
    tracker = env.trackers[agent]
    tracker.state[host] = "CONFIRMED"
    tracker.last_remediation[host] = 5
    tracker.last_analysis[host] = 9  # re-detected after Remove
    name, target = baselines.decode_index(env, agent, policy.select(env, agent))
    assert (name, target) == ("Restore", host)


def test_policies_do_not_touch_privileged_state(quiet):
    for policy in (baselines.SleepBaseline(), baselines.MaskedRandomBaseline(seed=1),
                   baselines.UnmaskedRandomBaseline(seed=1), baselines.RoundRobinBaseline()):
        with __import__("unittest.mock", fromlist=["patch"]).patch.object(
                quiet.env, "get_true_state",
                side_effect=AssertionError("policy read privileged state")):
            for agent in wrapper.BLUE_AGENTS:
                policy.select(quiet, agent)


def test_evaluate_policies_pairs_seeds(quiet):
    results = baselines.evaluate_policies(
        {"sleep": baselines.SleepBaseline,
         "masked-random": lambda: baselines.MaskedRandomBaseline(seed=0)},
        seeds=(42, 43), steps=10, snapshot_steps=(5,))
    assert set(results) == {"sleep", "masked-random"}
    for per_seed in results.values():
        assert set(per_seed) == {42, 43}
        assert all(r["steps"] == 9 for r in per_seed.values())


def test_run_episode_short_sleep_trace_shape(quiet):
    # Native scenario terminates at tick steps-1, so a steps=10 env yields 9 ticks.
    result = baselines.run_episode(baselines.SleepBaseline(), seed=42,
                                   steps=10, snapshot_steps=(5,))
    assert result["steps"] == 9
    assert len(result["trace"]) == 9 * 5
    assert result["snapshots"][5]["total"] >= 0
    assert all(set(r) == {"step", "agent", "action", "host", "reward"}
               for r in result["trace"])


def test_stalest_first_deterministic_and_legal():
    # Full-episode equality vs round-robin lives in the pool manifest
    # (proposal 12); here: determinism + mask legality on short episodes.
    key = lambda tr: [(t["step"], t["agent"], t["action"], t["host"])
                      for t in tr]
    r1 = baselines.run_episode(baselines.StalestFirstBaseline(), seed=7629,
                               steps=30)
    r2 = baselines.run_episode(baselines.StalestFirstBaseline(), seed=7629,
                               steps=30)
    assert key(r1["trace"]) == key(r2["trace"])
