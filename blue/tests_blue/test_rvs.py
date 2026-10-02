"""RvS return-conditioning tests (proposal 14, rung 1). Train venv (torch).

RTG math, rtg_dim slot-split layout (default path unchanged), the
run_episode observe_step hook (no-op for heuristics), and eval
determinism of the RvS checkpoint policy on a short episode.
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="RvS needs torch")
th = torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/sourav/Projects/cyberally/third_party/epymarl/src")

from blue_rvs_pretrain import RTG_SCALE, compute_rtg

ENV_KW = {"temporal_features": ("ages", "belief"),
          "include_root_session": True}


def test_compute_rtg_suffix_sums_and_scale():
    rewards = np.array([[1.0, -2.0, 0.0, -1.0]])
    rtg = compute_rtg(rewards, np.array([4]))
    # suffix sums: [-2, -3, -1, -1] / 100
    expected = np.array([[-0.02, -0.03, -0.01, -0.01]], dtype=np.float32)
    assert rtg.shape == (1, 4)
    np.testing.assert_allclose(rtg, expected, rtol=1e-5)


def test_compute_rtg_respects_lengths():
    rewards = np.array([[1.0, 1.0, 1.0, 1.0]])
    rtg = compute_rtg(rewards, np.array([2]))
    assert rtg[0, 0] == pytest.approx(0.02)
    assert rtg[0, 1] == pytest.approx(0.01)
    assert rtg[0, 2] == 0.0 and rtg[0, 3] == 0.0


def test_rtg_dim_split_keeps_heads():
    from types import SimpleNamespace as SN
    import blue_factorized_agent as factorized
    args = SN(hidden_dim=16, n_actions=155, use_rnn=True, n_agents=5,
              attn_layers=0, rtg_dim=1)
    agent = factorized.FactorizedRNNAgent(873, args)
    assert agent.slot_feats == 17  # (873 - 5 - 1) / 51, same as 872-data
    x = th.randn(4, 873)
    h = th.zeros(4, 16)
    flat, hn, host, cmd, glob = agent.decompose(x, h)
    assert flat.shape == (4, 155) and host.shape == (4, 51)
    assert cmd.shape == (4, 51, 3) and glob.shape == (4, 2)
    # RTG actually conditions the output (not a dead input).
    x_hi = x.clone()
    x_hi[:, -1] += 5.0
    flat_hi, _, _, _, _ = agent.decompose(x_hi, h)
    assert not th.allclose(flat, flat_hi)
    # Gradient reaches the RTG column of fc1.
    agent.zero_grad()
    flat.sum().backward()
    assert agent.fc1.weight.grad[:, -1].abs().sum() > 0


def test_observe_hook_noop_for_heuristics():
    from blue_baselines import SleepBaseline, run_episode
    assert not hasattr(SleepBaseline(), "observe_step")
    run = run_episode(SleepBaseline(), seed=7629, steps=10, **ENV_KW)
    assert isinstance(run["cumulative_return"], float)
    assert run["steps"] <= 10  # short episodes may truncate on phases


def test_rvs_eval_deterministic_short_episode():
    ckpt = "results/models/rvs_14/0"
    pytest.importorskip("torch")
    if not os.path.exists(os.path.join(ckpt, "agent.th")):
        pytest.skip("rvs_14 ckpt not trained")
    from blue_baselines import run_episode
    from blue_eval_rvs import RvSCheckpointPolicy
    import cc4_epymarl_wrapper as wrapper
    probe = wrapper.CC4MARLEnv(steps=30, **ENV_KW)
    env_info = probe.get_env_info()
    probe.close()
    returns, traces = [], []
    for _ in range(2):
        p = RvSCheckpointPolicy(ckpt, target_return=-50.0, env_info=env_info)
        run = run_episode(p, seed=7701, steps=30, **ENV_KW)
        returns.append(run["cumulative_return"])
        traces.append([(t["agent"], t["action"]) for t in run["trace"]])
    assert returns[0] == returns[1]
    assert traces[0] == traces[1]
    # RTG tracker actually moved during the episode.
    p = RvSCheckpointPolicy(ckpt, target_return=-50.0, env_info=env_info)
    run_episode(p, seed=7701, steps=30, **ENV_KW)
    assert p.rtg != -50.0
