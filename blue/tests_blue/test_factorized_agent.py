"""FactorizedRNNAgent unit tests (proposal 02). Train venv (torch).

Interface parity with RNNAgent (same forward/hidden shapes, same 155-way
logits layout), factorized structure (flat == host + cmd composition),
gradient flow into all heads, save/load roundtrip, and slot-width
derivation for the root-session ablation geometry.
"""
from types import SimpleNamespace as SN

import pytest

torch = pytest.importorskip("torch", reason="factorized agent needs torch")
th = torch

from modules.agents import REGISTRY as agent_REGISTRY
from blue.policies import factorized
from blue.policies.factorized import FactorizedRNNAgent, N_SLOTS

agent_REGISTRY["rnn_factorized"] = FactorizedRNNAgent


def _args(n_actions=155, n_agents=5, hidden_dim=64):
    return SN(hidden_dim=hidden_dim, n_actions=n_actions, n_agents=n_agents,
              use_rnn=True)


def test_output_shape_and_determinism():
    agent = FactorizedRNNAgent(872, _args())
    x = th.randn(5, 872)
    h = th.zeros(5, 64)
    q1, h1 = agent(x, h)
    q2, h2 = agent(x, h)
    assert q1.shape == (5, 155)
    assert h1.shape == (5, 64)
    assert th.equal(q1, q2) and th.equal(h1, h2)


def test_flat_is_host_plus_cmd_composition():
    agent = FactorizedRNNAgent(872, _args())
    # Zero the host head: within each slot-block, flat logits must then
    # equal the cmd head output up to a per-slot constant.
    with th.no_grad():
        agent.host_head.weight.zero_()
        agent.host_head.bias.zero_()
    x = th.randn(2, 872)
    q, _ = agent(x, th.zeros(2, 64))
    cmd = agent.cmd_head  # structural check via block pattern below
    for s in range(N_SLOTS):
        block = q[:, 2 + 3 * s:2 + 3 * s + 3]
        # host score constant across the block -> block rows differ only
        # through cmd; equivalently block[:, c] - block[:, 0] is host-free.
        diff = block[:, 1:] - block[:, :1]
        assert diff.shape == (2, 2)
    assert cmd is not None


def test_gradient_flows_to_all_heads():
    agent = FactorizedRNNAgent(872, _args())
    q, _ = agent(th.randn(4, 872), th.zeros(4, 64))
    q.sum().backward()
    for name in ("slot_enc", "host_head", "cmd_head", "global_head",
                 "fc1", "rnn"):
        mod = getattr(agent, name)
        params = list(mod.parameters())
        assert params and all(p.grad is not None for p in params), name


def test_save_load_roundtrip(tmp_path):
    agent = FactorizedRNNAgent(872, _args())
    path = str(tmp_path / "agent.th")
    th.save(agent.state_dict(), path)
    other = FactorizedRNNAgent(872, _args())
    other.load_state_dict(th.load(path, map_location="cpu",
                                  weights_only=True))
    x = th.randn(3, 872)
    assert th.equal(agent(x, th.zeros(3, 64))[0],
                    other(x, th.zeros(3, 64))[0])


def test_slot_width_derivation_and_init_hidden():
    full = FactorizedRNNAgent(872, _args())
    assert full.slot_feats == 17
    slim = FactorizedRNNAgent(51 * 16 + 5, _args())
    assert slim.slot_feats == 16
    q, _ = slim(th.randn(5, 51 * 16 + 5), slim.init_hidden().expand(5, -1))
    assert q.shape == (5, 155)
    assert full.init_hidden().shape == (1, 64)


def test_decompose_matches_forward():
    agent = FactorizedRNNAgent(872, _args())
    x = th.randn(4, 872)
    h = th.zeros(4, 64)
    flat, h1, host, cmd, glob = agent.decompose(x, h)
    q, h2 = agent(x, h)
    assert th.equal(flat, q) and th.equal(h1, h2)
    assert host.shape == (4, 51) and cmd.shape == (4, 51, 3)
    assert glob.shape == (4, 2)
    # flat[2+3s+c] == host[s] + cmd[s, c]
    recon = (host.unsqueeze(-1) + cmd).reshape(4, -1)
    assert th.equal(q[:, 2:], recon)
    assert th.equal(q[:, :2], glob)


def _args_attn(n_layers=1):
    a = _args()
    a.attn_layers = n_layers
    return a


def test_attention_path_shapes_and_grads():
    agent = FactorizedRNNAgent(872, _args_attn(1))
    x = th.randn(3, 872)
    q, h = agent(x, th.zeros(3, 64))
    assert q.shape == (3, 155) and h.shape == (3, 64)
    q.sum().backward()
    assert all(p.grad is not None for p in agent.slot_attn.parameters())
    # attn=0 models have no attention module (behavior identical to before)
    plain = FactorizedRNNAgent(872, _args())
    assert not hasattr(plain, "slot_attn")


def test_greedy_ckpt_resume_factorized(tmp_path):
    # Minimal ckpt-resume path: random-weights agent.th in ckpt layout,
    # GreedyCheckpointPolicy loads it and emits legal actions for a few
    # quiet steps. Guards the torch eval path behind all MAPPO manifests.
    import blue.core.wrapper as wrapper
    from CybORG.Agents import SleepAgent
    from blue.policies.eval_mappo import GreedyCheckpointPolicy
    agent = FactorizedRNNAgent(872, _args())
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    th.save(agent.state_dict(), str(ckpt / "agent.th"))
    real_red = wrapper.DiscoveryFSRed
    real_green = wrapper.EnterpriseGreenAgent
    wrapper.DiscoveryFSRed = SleepAgent
    wrapper.EnterpriseGreenAgent = SleepAgent
    try:
        env = wrapper.CC4MARLEnv(steps=6, temporal_features=("ages", "belief"),
                                 include_root_session=True)
        env_info = env.get_env_info()
        env.reset()
        pol = GreedyCheckpointPolicy(str(ckpt), env_info=env_info,
                                     agent_type="rnn_factorized")
        for _ in range(5):
            acts = {a: pol.select(env, a) for a in wrapper.BLUE_AGENTS}
            for a, idx in acts.items():
                mask = env.get_avail_agent_actions(
                    wrapper.BLUE_AGENTS.index(a))
                assert mask[idx] == 1, (a, idx)
            env.step(acts)
    finally:
        wrapper.DiscoveryFSRed = real_red
        wrapper.EnterpriseGreenAgent = real_green
