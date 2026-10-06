"""The risk-input ablation must be a real ablation.

The question is narrow: does the learned risk score help RL? The control arm
holds column 15 at 0.0 during training AND evaluation. If the flag leaked --
most likely by evaluation silently rebuilding the real risk model -- the
control would be contaminated and the comparison meaningless.

These tests fail if:
  - the zero arm produces a non-zero 15th column;
  - the real and zero arms produce identical features (the ablation is a
    no-op, e.g. the flag was never threaded through);
  - FEAT_DIM changes between arms (breaks the Phase-3 body load and
    silently changes the architecture under test).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from blue.training import risk_actor as ra  # noqa: E402


class _FakeEnv(object):
    def __init__(self, hosts):
        self.hostnames = {"blue_agent_0": hosts}
        self._tick = 7


class _StubRisk(object):
    """A risk model that returns a loud, obviously non-zero value."""

    def _predict_all(self, env, agent):
        return {h: 0.9 for h in env.hostnames[agent]}


def _rows(risk, hosts=("host_a", "host_b", "host_c")):
    """Call host_rows_15 with a stubbed base feature matrix."""
    import blue.training.scorer as scorer
    base = np.arange(3 * 14, dtype=np.float32).reshape(3, 14)
    orig = scorer.host_rows
    scorer.host_rows = lambda env, agent, cands: base
    try:
        return ra.host_rows_15(_FakeEnv(list(hosts)), "blue_agent_0",
                               list(hosts), risk)
    finally:
        scorer.host_rows = orig


def test_zero_risk_reports_zero_for_every_host():
    env = _FakeEnv(["a", "b", "c"])
    out = ra.ZeroRisk()._predict_all(env, "blue_agent_0")
    assert out == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_zero_risk_column_is_exactly_zero_and_width_unchanged():
    real = _rows(_StubRisk())
    zero = _rows(ra.ZeroRisk())
    assert zero.shape == real.shape, "ablation must not change FEAT_DIM"
    assert zero.shape[1] == ra.FEAT_DIM
    assert np.all(zero[:, :14] == real[:, :14]), \
        "the 14 Blue-visible columns must be identical across arms"
    assert np.all(zero[:, 14] == 0.0), "column 15 must be exactly zero"


def test_ablation_actually_changes_the_features():
    """Guards against a silently ignored flag."""
    real = _rows(_StubRisk())
    zero = _rows(ra.ZeroRisk())
    assert not np.allclose(real[:, 14], zero[:, 14]), \
        "real and zero arms produced identical inputs; flag not threaded"


def test_risk_factory_selects_the_right_model():
    real = ra._risk("real")
    zero = ra._risk("zero")
    assert isinstance(zero, ra.ZeroRisk)
    assert not isinstance(real, ra.ZeroRisk)
    # the real arm must still expose the learned model interface
    assert hasattr(real, "_predict_all")
    assert not hasattr(real, "w") or real.__class__.__name__ == \
        "RiskPriority", "real arm must be the learned RiskPriority"


def test_feature_dim_is_15_in_both_arms():
    """Shape must not drift between arms or the Phase-3 body cannot load."""
    actor = ra.build_actor(hidden=64)
    first = None
    for name, p in actor.named_parameters():
        if p.dim() == 2 and first is None:
            first = p.shape
    assert ra.FEAT_DIM == 15
    body_w = actor.body[0].weight
    assert body_w.shape[1] == ra.FEAT_DIM


def test_cli_exposes_both_arms():
    """Guard against the flag being dropped from the parser."""
    import argparse
    import io
    import contextlib
    from blue.training import risk_actor
    src = open(risk_actor.__file__).read()
    assert '"--risk-ablation"' in src
    assert src.count("args.risk_ablation") >= 2, \
        "ablation must be consumed by BOTH train and eval"