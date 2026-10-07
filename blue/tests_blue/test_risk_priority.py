"""RiskPriority scorer (proposal 01 step 3). Live sim fixture, synthetic model.

Model weights are hand-set in tmp files (tests must not depend on the
gitignored results/ artifacts). The real-model proof is the pool eval.
"""
import os
import pickle

import pytest
from CybORG.Agents import SleepAgent

import blue.core.wrapper as wrapper
from blue.policies.hybrid import HybridBluePolicy, RiskPriority, make_priority


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


def _toy_model(path):
    """P(comp) rises only with unknown-file count (feature 6 of 21)."""
    w = [0.0] * 21
    w[6] = 5.0
    model = {"w": w, "b": -2.0,
             "mean": [0.0] * 21, "std": [1.0] * 21, "n_feats": 21}
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def test_risk_prefers_unknown_files(quiet, tmp_path):
    path = _toy_model(str(tmp_path / "m.pkl"))
    scorer = RiskPriority(model_path=path)
    agent = "blue_agent_0"
    h0, h1 = quiet.hostnames[agent][:2]
    quiet.views[agent][h0]["Files"] = [
        {"File Name": "x", "Known File": "UNKNOWN", "Density": 0.9}]
    assert scorer(quiet, agent, h0) > scorer(quiet, agent, h1)


def test_risk_batch_cache_consistent(quiet, tmp_path):
    path = _toy_model(str(tmp_path / "m.pkl"))
    scorer = RiskPriority(model_path=path)
    agent = "blue_agent_0"
    hosts = quiet.hostnames[agent]
    first = [scorer(quiet, agent, h) for h in hosts]
    assert scorer._cache_key == (agent, quiet._tick)
    second = [scorer(quiet, agent, h) for h in hosts]
    assert first == second


def test_risk_missing_model_loud(tmp_path):
    with pytest.raises(FileNotFoundError):
        RiskPriority(model_path=str(tmp_path / "absent.pkl"))


def test_risk_geometry_mismatch_loud(tmp_path):
    bad = {"w": [0.0] * 17, "b": 0.0,
           "mean": [0.0] * 17, "std": [1.0] * 17, "n_feats": 17}
    path = str(tmp_path / "m17.pkl")
    with open(path, "wb") as f:
        pickle.dump(bad, f)
    with pytest.raises(ValueError, match="geometry"):
        RiskPriority(model_path=path)


def test_make_priority_risk_and_policy_reset(tmp_path, quiet):
    path = _toy_model(str(tmp_path / "m.pkl"))
    policy = HybridBluePolicy(priority_fn="risk",
                              priority_kwargs={"model_path": path})
    assert isinstance(policy.priority_fn, RiskPriority)
    agent = "blue_agent_0"
    policy.priority_fn(quiet, agent, quiet.hostnames[agent][0])
    assert policy.priority_fn._cache_key is not None
    policy.reset()
    assert policy.priority_fn._cache_key is None
    assert os.path.exists(path)
