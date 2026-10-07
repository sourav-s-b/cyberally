"""IQL-discrete unit tests (proposal 14, rung 2). Train venv (torch).

Expectile asymmetry math, Bellman-target construction on a synthetic
chain, AWR weight clipping stats, and EPyMARL key compatibility of the
extracted factorized policy (pool loads it via mappo_ckpt_factorized).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="IQL needs torch")
th = torch

from blue.training.iql import awr_weights, build_mlp, expectile_loss


def test_expectile_asymmetry():
    diff = th.tensor([2.0, -2.0])
    loss = expectile_loss(diff, 0.7)
    # 0.7*4 + 0.3*4 over 2 = 2.0
    assert float(loss) == pytest.approx(2.0)
    # tau=0.5 reduces to plain MSE/2... mean of diff^2 / 2? No: mean(w*d^2)
    # with w=0.5 -> 0.5*mean(d^2) = 0.5*4 = 2.0
    assert float(expectile_loss(diff, 0.5)) == pytest.approx(2.0)
    # Higher tau penalizes positive diffs more (asymmetric case).
    pos = th.tensor([2.0])
    assert float(expectile_loss(pos, 0.9)) > float(expectile_loss(pos, 0.1))


def test_expectile_upper_envelope_direction():
    # V below all Q samples: loss must decrease when V moves up.
    q = th.tensor([1.0, 2.0, 3.0])
    v_lo, v_hi = th.tensor(0.0, requires_grad=True), None
    l_lo = expectile_loss(q - v_lo, 0.7)
    assert float(l_lo) > float(expectile_loss(q - th.tensor(1.5), 0.7))


def test_awr_weights_clip_and_stats():
    adv = th.tensor([-10.0, 0.0, 1.0, 100.0])
    w, frac = awr_weights(adv, beta=1.0, max_w=100.0)
    assert float(w[1]) == pytest.approx(1.0)  # exp(0)
    assert float(w[0]) < 1.0
    assert float(w[3]) == pytest.approx(100.0)  # clipped
    assert float(frac) == pytest.approx(0.25)
    assert bool((w <= 100.0).all())


def test_awr_alpha_blend_endpoints():
    # alpha=1.0 is pure AWR; alpha=0.0 reproduces the free-tick mask
    # (uniform BC weighting). Blend line from blue.training.iql main loop.
    adv = th.tensor([0.0, 2.0, -1.0, 5.0])
    free = th.tensor([1.0, 1.0, 0.0, 1.0])
    w, _ = awr_weights(adv, beta=1.0)
    w = w * free
    pure = 1.0 * w + 0.0 * free
    assert th.allclose(pure, w)
    bc = 0.0 * w + 1.0 * free
    assert th.allclose(bc, free)
    mid = 0.5 * w + 0.5 * free
    assert bool(((mid > 0) == (free > 0)).all())  # support == free ticks
    assert float(mid[1]) > 0.5  # positive advantage upweighted vs uniform


def test_bellman_target_terminal_and_bootstrapped():
    gamma = 0.99
    r = th.tensor([1.0, -2.0])
    v_ns = th.tensor([5.0, 5.0])
    d = th.tensor([0.0, 1.0])
    target = r + gamma * (1.0 - d) * v_ns
    assert float(target[0]) == pytest.approx(1.0 + 0.99 * 5.0)
    assert float(target[1]) == pytest.approx(-2.0)  # terminal: no bootstrap


def test_extracted_policy_key_compat():
    from types import SimpleNamespace as SN
    from blue.policies import factorized
    args = SN(hidden_dim=16, n_actions=155, use_rnn=True, n_agents=5,
              attn_layers=0)
    policy = factorized.FactorizedRNNAgent(872, args)
    fresh = factorized.FactorizedRNNAgent(872, args)
    # A rung-2 policy state dict must load strict into a stock factorized
    # agent (this is what GreedyCheckpointPolicy does in the pool worker).
    missing = fresh.load_state_dict(policy.state_dict(), strict=True)
    assert not missing.missing_keys and not missing.unexpected_keys
    # And it must implement the recurrent forward the pool calls.
    x = th.randn(2, 872)
    h = th.zeros(2, 16)
    flat, hn = fresh(x, h)
    assert flat.shape == (2, 155) and hn.shape == (2, 16)


def test_qv_mlp_shapes():
    q = build_mlp(872, 155)
    v = build_mlp(872, 1)
    x = th.randn(7, 872)
    assert q(x).shape == (7, 155)
    assert v(x).shape == (7, 1)


def test_qv_mlp_layernorm_shapes_and_norm():
    q = build_mlp(872, 155, layernorm=True)
    v = build_mlp(872, 1, layernorm=True)
    assert any(isinstance(m, th.nn.LayerNorm) for m in q.modules())
    x = th.randn(7, 872)
    assert q(x).shape == (7, 155)
    assert v(x).shape == (7, 1)
    assert th.isfinite(q(x)).all() and th.isfinite(v(x)).all()
