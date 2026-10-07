"""BLUE-04 temporal/belief feature tests (no EPyMARL install needed).

Temporal inputs are Blue-visible belief/observation bookkeeping only — never
privileged truth. Default (no groups) is byte-identical v2 behaviour; enabled
groups widen the host vector and must not alter masks, transitions or
rewards (trace-invariance test). Uses Sleep red/green for speed.
"""
import numpy as np
import pytest
import blue.core.wrapper as wrapper
from blue.core.masking import BlueZoneTracker
from blue.core.obs_features import host_to_temporal
import blue.core.baselines as baselines
from blue.policies.eval_mappo import policy_dims
from CybORG.Agents import SleepAgent


FULL = ("ages", "belief", "freshness", "mission")


@pytest.fixture
def calm(monkeypatch):
    monkeypatch.setattr(wrapper, "DiscoveryFSRed", SleepAgent)
    monkeypatch.setattr(wrapper, "EnterpriseGreenAgent", SleepAgent)


def test_default_off_is_v2_identical(calm):
    env = wrapper.CC4MARLEnv(steps=30)
    info = env.get_env_info()
    assert info["temporal_features"] == []
    assert info["host_vector_len"] == 10
    assert info["obs_shape"] == 510
    assert info["obs_size_per_agent"] == [510] * 5
    assert info["wrapper_version"] == "foundation-v3"
    assert info["include_root_session"] is True


def test_bundle_dims(calm):
    env = wrapper.CC4MARLEnv(steps=30, temporal_features=FULL)
    info = env.get_env_info()
    assert info["host_vector_len"] == 20  # 10 base + 2 + 5 + 2 + 1
    assert info["obs_shape"] == 51 * 20
    assert info["temporal_features"] == list(FULL)


def test_bundle_dims_per_agent_bounds(calm):
    env = wrapper.CC4MARLEnv(steps=30, per_agent_bounds=True,
                             temporal_features=FULL)
    assert env.get_env_info()["obs_size_per_agent"] == [340] * 4 + [1020]


def test_partial_groups_and_drop_root_session(calm):
    env = wrapper.CC4MARLEnv(steps=30, temporal_features=("ages",))
    assert env.get_env_info()["host_vector_len"] == 12
    env = wrapper.CC4MARLEnv(steps=30, temporal_features=FULL,
                             include_root_session=False)
    assert env.get_env_info()["host_vector_len"] == 19  # 9 + 10
    assert env.get_env_info()["obs_shape"] == 51 * 19


def test_invalid_groups_rejected(calm):
    with pytest.raises(ValueError, match="unknown temporal groups"):
        wrapper.CC4MARLEnv(steps=30, temporal_features=("ages", "telepathy"))
    with pytest.raises(ValueError, match="unknown temporal groups"):
        host_to_temporal(None, "h", {}, 0, 30, groups=("nope",))


def test_post_reset_row_layout(calm):
    env = wrapper.CC4MARLEnv(steps=30, temporal_features=FULL)
    env.reset(seed=7629)
    row = env.get_obs_agent(0)[:20]
    # ages: never scanned -> maximally stale sentinel, distinct from fresh 0.0
    np.testing.assert_array_equal(row[10:12], [1.0, 1.0])
    # belief: UNKNOWN one-hot
    np.testing.assert_array_equal(row[12:17], [1, 0, 0, 0, 0])
    # freshness: snapshot just taken; pending: agent free
    np.testing.assert_array_equal(row[17:19], [0.0, 0.0])
    # mission: tick 0
    assert row[19] == 0.0


def test_belief_onehot_unit():
    tracker = BlueZoneTracker(["h"])
    tracker.state["h"] = "CONFIRMED"
    out = host_to_temporal(tracker, "h", {"Files": 4}, 9, 30,
                           groups=("belief",))
    assert out == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_age_decays_after_live_analyse(calm):
    env = wrapper.CC4MARLEnv(steps=30, temporal_features=FULL)
    env.reset(seed=7629)
    agent = "blue_agent_0"
    env.step({agent: 2})  # Analyse host 0 (index 2 + 3*0 + Analyse(0))
    for _ in range(6):
        if agent not in env._awaiting:
            break
        env.step({a: 0 for a in wrapper.BLUE_AGENTS})
    assert agent not in env._awaiting
    scanned = env.get_obs_agent(0)[:20][10:12]
    others = env.get_obs_agent(0)[20:40][10:12]
    assert scanned[0] < 0.5
    assert others[0] == 1.0  # never scanned: sentinel, not zero-fill


def test_drop_root_removes_only_index_five(calm):
    full = wrapper.CC4MARLEnv(steps=30, temporal_features=FULL)
    full.reset(seed=7629)
    drop = wrapper.CC4MARLEnv(steps=30, temporal_features=FULL,
                              include_root_session=False)
    drop.reset(seed=7629)
    row_full = full.get_obs_agent(0)[:20]
    row_drop = drop.get_obs_agent(0)[:19]
    base_full = list(row_full[:10])
    del base_full[5]
    np.testing.assert_array_equal(row_drop[:9], base_full)
    np.testing.assert_array_equal(row_drop[9:], row_full[10:])


def test_trace_invariance_temporal_on_off(calm):
    off = baselines.run_episode(baselines.MaskedRandomBaseline(seed=11),
                                seed=7629, steps=60, snapshot_steps=(30,))
    on = baselines.run_episode(baselines.MaskedRandomBaseline(seed=11),
                               seed=7629, steps=60, snapshot_steps=(30,),
                               temporal_features=FULL)
    assert off["cumulative_return"] == on["cumulative_return"]
    assert off["snapshots"] == on["snapshots"]
    assert [t["action"] for t in off["trace"]] == [t["action"] for t in on["trace"]]


def test_policy_dims_derive_from_env_info():
    assert policy_dims({"n_agents": 5, "obs_shape": 510,
                        "n_actions": 155}) == (5, 515, 155)
    assert policy_dims({"n_agents": 5, "obs_shape": 1020,
                        "n_actions": 155}) == (5, 1025, 155)


def test_greedy_policy_loads_matching_checkpoint_and_selects(calm, tmp_path):
    """Train-venv-only regression test: policy construction must use
    policy_dims output directly as the net input (no double-added agent
    id) and select() must work without module-global torch."""
    torch = pytest.importorskip("torch")
    agents_mod = pytest.importorskip("modules.agents")
    from types import SimpleNamespace as SN

    from blue.policies.eval_mappo import GreedyCheckpointPolicy
    env = wrapper.CC4MARLEnv(steps=30, temporal_features=("ages",))
    env.reset(seed=7629)
    info = env.get_env_info()
    n_agents, obs_dim, n_actions = policy_dims(info)
    net = agents_mod.REGISTRY["rnn"](obs_dim,
                                     SN(hidden_dim=64, n_actions=n_actions,
                                        use_rnn=True))
    torch.save(net.state_dict(), tmp_path / "agent.th")
    policy = GreedyCheckpointPolicy(str(tmp_path), env_info=info)
    for agent in wrapper.BLUE_AGENTS:
        action = policy.select(env, agent)
        assert 0 <= action < n_actions
        assert env.get_avail_agent_actions(
            wrapper.BLUE_AGENTS.index(agent))[action] == 1
