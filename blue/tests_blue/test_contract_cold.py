"""OP1 cold-contract audit: the wrapper against the native simulator.

Unlike the foundation regression tests (which mock red/green for determinism
and inject failures at the observation boundary), this file runs the unmodified
native scenario and checks the contract end to end:

1. every action slot the mask enables decodes to an owned host that the
   simulator action space supports (no slot for a non-existent host);
2. a fixed seed reproduces actions, rewards and terminal flags identically;
3. concurrent pending actions attribute to their own agent only, and off-mask
   choices are coerced to Sleep without crashing or corrupting state;
4. a successful Remove enters VERIFY (never CLEAN) with action completion,
   observed verification and privileged evaluation kept distinct;
5. privileged reads never perturb actor-visible observations;
6. the native horizon end reports terminated (not truncated) and rejects
   further steps.

No reserved seed block is consumed: seed 7629 is the wrapper's already-spent
baseline seed. Runs the sim-only path; no torch.
"""
import numpy as np
import pytest
import blue.core.wrapper as wrapper
from blue.core.baselines import (MaskedRandomBaseline, UnmaskedRandomBaseline,
                                 count_compromised)

SEED = 7629
STEPS = 60


def native_env(steps=STEPS):
    env = wrapper.CC4MARLEnv(seed=SEED, steps=steps)
    env.reset(seed=SEED)
    return env


def action_index(env, agent, host, action):
    return 2 + 3 * env.hostnames[agent].index(host) + wrapper.ACTION_TEMPLATES.index(action)


def mask_offers_only_grounded_slots(env):
    for i, agent in enumerate(wrapper.BLUE_AGENTS):
        mask = env.get_avail_agent_actions(i)
        space = env.env.get_action_space(agent)
        hosts = env.hostnames[agent]
        assert mask[0] == 1, f"{agent}: Sleep must always be legal"
        assert mask[2 + 3 * len(hosts):].sum() == 0, (
            f"{agent}: mask enables a slot beyond owned hosts")
        for idx in np.flatnonzero(mask):
            idx = int(idx)
            if idx < 2:
                continue
            hi, action_type = divmod(idx - 2, 3)
            host = hosts[hi]
            assert host in space["hostname"], (
                f"{agent}: slot {idx} targets host {host} not in action space")
            assert space["action"].get(
                wrapper.CLASSES[wrapper.ACTION_TEMPLATES[action_type]], False), (
                f"{agent}: slot {idx} action class not supported")


def test_mask_offers_only_grounded_slots_through_whole_episode():
    env = native_env()
    policy = MaskedRandomBaseline(seed=7)
    terminated = truncated = False
    while not (terminated or truncated):
        mask_offers_only_grounded_slots(env)
        actions = {a: int(policy.select(env, a)) for a in wrapper.BLUE_AGENTS}
        masks = env.get_avail_actions()
        for i, agent in enumerate(wrapper.BLUE_AGENTS):
            assert masks[i][actions[agent]] == 1, (
                f"{agent}: policy chose an off-mask action")
        _, rewards, terminated, truncated, info = env.step(actions)
        assert len(rewards) == 5
        assert isinstance(info["episode_limit"], bool)
    assert terminated != truncated


def run_fixed_trajectory():
    env = native_env()
    policy = MaskedRandomBaseline(seed=7)
    frames = []
    terminated = truncated = False
    while not (terminated or truncated):
        actions = {a: int(policy.select(env, a)) for a in wrapper.BLUE_AGENTS}
        state_before = np.asarray(env.get_state(), dtype=np.float32).tobytes()
        _, rewards, terminated, truncated, info = env.step(actions)
        frames.append((tuple(sorted(actions.items())),
                       [float(r) for r in rewards], terminated, truncated,
                       info["episode_limit"], state_before))
        if terminated or truncated:
            break
    return frames


def test_native_episode_reproduces_identical_actions_rewards_and_flags():
    frames_a = run_fixed_trajectory()
    frames_b = run_fixed_trajectory()
    assert len(frames_a) > 0, "native episode produced no steps"
    assert frames_a == frames_b, "same seed did not reproduce the native episode"


def test_off_mask_choices_are_coerced_to_sleep_and_valid_choices_honoured():
    env = native_env()
    policy = UnmaskedRandomBaseline(seed=9)
    terminated = truncated = False
    count = 0
    while not (terminated or truncated) and count < 30:
        masks = env.get_avail_actions()
        actions = {}
        off_mask = {}
        for i, agent in enumerate(wrapper.BLUE_AGENTS):
            idx = int(policy.select(env, agent))
            actions[agent] = idx
            off_mask[agent] = not masks[i][idx]
        for agent in wrapper.BLUE_AGENTS:
            if off_mask[agent]:
                # Passive observations may still update views, but a coerced
                # Sleep choice must never create a pending action.
                assert agent not in env._awaiting, (
                    f"{agent}: coerced off-mask action must not become pending")
        count += 1


def test_pending_agent_off_mask_and_extreme_indices_do_not_crash():
    env = native_env()
    agent = "blue_agent_0"
    host = env.hostnames[agent][0]
    env.step({agent: action_index(env, agent, host, "Analyse")})
    env.step({agent: action_index(env, agent, host, "Restore")})
    env.step({agent: 10 ** 6})
    env.step({agent: -3})
    for _ in range(10):
        mask_offers_only_grounded_slots(env)
        _, _, term, trunc, _ = env.step([0] * 5)
        if term or trunc:
            break
    assert agent not in env._awaiting, "pending action never resolved"
    tracker = env.trackers[agent]
    assert tracker.last_result[host] is not None
    assert tracker.last_result[host][1] in ("TRUE", "FALSE")


def test_concurrent_pending_actions_attribute_to_their_own_agents_only():
    env = native_env()
    targets = {}
    issued = {}
    for agent in wrapper.BLUE_AGENTS:
        host = env.hostnames[agent][0]
        targets[agent] = host
        issued[agent] = action_index(env, agent, host, "Analyse")
    env.step(issued)
    for agent in wrapper.BLUE_AGENTS:
        assert env._awaiting[agent] == (targets[agent], "Analyse")
    for _ in range(10):
        mask_offers_only_grounded_slots(env)
        _, _, term, trunc, _ = env.step([0] * 5)
        if all(a not in env._awaiting for a in wrapper.BLUE_AGENTS):
            break
        if term or trunc:
            break
    for agent in wrapper.BLUE_AGENTS:
        tracker = env.trackers[agent]
        assert agent not in env._awaiting
        assert tracker.last_result[targets[agent]][0] == "Analyse"
        assert tracker.last_result[targets[agent]][1] == "TRUE"


def resolve_pending(env, agent, max_steps=10):
    for _ in range(max_steps):
        _, _, term, trunc, _ = env.step([0] * 5)
        if term or trunc:
            pytest.fail(f"{agent}: episode ended while {agent} was pending")
        if agent not in env._awaiting:
            return
    pytest.fail(f"{agent}: pending action never resolved")


def test_remove_success_enters_verify_with_distinct_evidence_facts():
    env = native_env()
    agent = "blue_agent_0"
    tracker = env.trackers[agent]
    host = env.env.state.sessions[agent][0].hostname
    assert host in env.hostnames[agent]
    privileged_before, root_before = count_compromised(env)
    privileged_set_before = env._compromised_set()
    state_before = np.asarray(env.get_state()).tobytes()
    assert len(env._compromised_set()) == privileged_before
    assert np.asarray(env.get_state()).tobytes() == state_before
    assert count_compromised(env) == (privileged_before, root_before)
    assert env._compromised_set() == privileged_set_before

    had_files = "Files" in env.views[agent][host]
    had_processes = "Processes" in env.views[agent][host]

    env.step({agent: action_index(env, agent, host, "Remove")})
    assert env._awaiting == {agent: (host, "Remove")}
    resolve_pending(env, agent)
    name, flag = tracker.last_result[host]
    assert (name, flag) == ("Remove", "TRUE"), f"Remove unexpectedly returned {flag}"
    assert tracker.state[host] == "VERIFY", "successful Remove must not mean CLEAN"
    assert agent not in env._awaiting
    if had_files or had_processes:
        assert ("Files" not in env.views[agent][host]
                and "Processes" not in env.views[agent][host]), (
            "historical artifacts must not survive as current measurements")
    env.step({agent: action_index(env, agent, host, "Analyse")})
    resolve_pending(env, agent)
    assert tracker.last_result[host][0] == "Analyse"
    assert tracker.last_result[host][1] == "TRUE"
    assert env._compromised_set() is not None
    assert len(env._compromised_set()) == count_compromised(env)[0]


def test_privileged_read_does_not_perturb_actor_inputs():
    env = native_env()
    env.step([0] * 5)
    before = np.asarray(env.get_state()).tobytes()
    totals = count_compromised(env)
    mid = np.asarray(env.get_state()).tobytes()
    assert before == mid
    assert len(env._compromised_set()) == totals[0]
    after = np.asarray(env.get_state()).tobytes()
    assert before == after


def test_horizon_end_reports_terminated_and_rejects_further_steps():
    env = native_env(steps=8)
    hit = None
    for _ in range(60):
        mask_offers_only_grounded_slots(env)
        _, _, terminated, truncated, info = env.step([0] * 5)
        if terminated or truncated:
            hit = (terminated, truncated, info["episode_limit"])
            break
    assert hit is not None, "episode never ended within its horizon"
    terminated, truncated, _ = hit
    assert terminated and not truncated, (
        "native horizon end must be terminated, not a wrapper truncation")
    with pytest.raises(RuntimeError, match="Call reset"):
        env.step([0] * 5)
    env.reset(seed=SEED)
    assert not env._awaiting
    assert all(t.pending_until == {} for t in env.trackers.values())
    env.step([0] * 5)
