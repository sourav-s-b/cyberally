"""Behavioral checks for the opt-in harness, not aggregate-performance claims."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from blue.harness.features import ACTOR_NAMES, BlueFeatures, NAMES, neural_rows
from blue.harness.policy import HarnessPolicy
from blue.training.harness_ml import validate_splits


class VisibleOnlyEnv:
    def __init__(self):
        self._tick = 0
        self.episode_limit = 400
        self._awaiting = {}
        self.hostnames = {"a": ["operational_zone_a_subnet_server_host_1", "h2"]}
        self.subnets = {"a": {h: [] for h in self.hostnames["a"]}}
        self.views = {"a": {h: {} for h in self.hostnames["a"]}}
        self.observed_at = {"a": {h: {} for h in self.hostnames["a"]}}
        self.trackers = {"a": SimpleNamespace(
            state={}, last_analysis={}, last_remediation={}, last_result={}, empty_strikes={})}

    @property
    def env(self):
        raise AssertionError("privileged state must never be accessed")

    @property
    def state(self):
        raise AssertionError("privileged state must never be accessed")


def test_missing_is_not_observed_zero():
    env = VisibleOnlyEnv()
    h = env.hostnames["a"][0]
    pipe = BlueFeatures()
    missing = pipe.rows(env, "a", [h])[0]
    assert np.isnan(missing[NAMES.index("n_files")])
    env._tick = 1
    env.views["a"][h]["Files"] = []
    env.observed_at["a"][h]["Files"] = 1
    observed = pipe.rows(env, "a", [h])[0]
    assert observed[NAMES.index("n_files")] == 0
    assert observed[NAMES.index("Files_available")] == 1
    assert missing[NAMES.index("Files_available")] == 0


def test_deltas_idempotent_candidate_order_and_new_episode():
    env = VisibleOnlyEnv()
    h = env.hostnames["a"][0]
    env.views["a"][h]["Files"] = []
    env.observed_at["a"][h]["Files"] = 0
    pipe = BlueFeatures()
    pipe.rows(env, "a")
    env._tick = 1
    env.views["a"][h]["Files"] = [{"Known File": "UNKNOWN", "Density": 0.9}]
    env.observed_at["a"][h]["Files"] = 1
    a = pipe.rows(env, "a", [h])
    b = pipe.rows(env, "a", list(reversed(env.hostnames["a"])))
    np.testing.assert_equal(a[0], b[1])
    assert a[0, NAMES.index("delta_unknown_files")] == 1
    env._tick = 2
    assert pipe.rows(env, "a", [h])[0, NAMES.index("delta_unknown_files")] == 0
    env._tick = 0
    with pytest.raises(ValueError, match="clock"):
        pipe.rows(env, "a")
    pipe.reset()
    env.observed_at["a"][h]["Files"] = 0
    assert pipe.rows(env, "a", [h])[0, NAMES.index("delta_unknown_files")] == 0


def test_future_timestamp_and_nonfinite_evidence_rejected():
    env = VisibleOnlyEnv()
    h = env.hostnames["a"][0]
    env.observed_at["a"][h]["Files"] = 10
    with pytest.raises(ValueError, match="timestamp"):
        BlueFeatures().rows(env, "a")
    env.observed_at["a"][h]["Files"] = 0
    env.views["a"][h]["Files"] = [{"Density": float("inf")}]
    with pytest.raises(ValueError, match="nonfinite"):
        BlueFeatures().rows(env, "a")


def test_completion_age_and_busy_are_separate():
    env = VisibleOnlyEnv()
    h = env.hostnames["a"][0]
    env._tick = 20
    env.trackers["a"].last_analysis[h] = 5
    env._awaiting["a"] = (h, "Analyse")
    row = BlueFeatures().rows(env, "a", [h])[0]
    assert row[NAMES.index("analysis_age")] == pytest.approx(15 / 400)
    assert row[NAMES.index("busy")] == 1
    assert row[NAMES.index("analysis_available")] == 1


def test_neural_geometry_scaling_and_public_phase():
    env = VisibleOnlyEnv()
    env._tick = 268
    X = BlueFeatures().rows(env, "a")
    assert X[0, NAMES.index("phase_2")] == 1
    assert X[0, NAMES.index("role_server")] == 1
    Z = neural_rows(X, np.zeros(len(NAMES)), np.ones(len(NAMES)))
    assert np.isfinite(Z).all()
    with pytest.raises(ValueError, match="normalization"):
        neural_rows(X, np.zeros(len(NAMES)), np.zeros(len(NAMES)))


def test_split_leakage_and_final_seed_block_rejected():
    validate_splits({"train": [7706], "calibration": [7701], "test": [7703]})
    with pytest.raises(ValueError, match="leakage"):
        validate_splits({"train": [7706], "calibration": [7706], "test": [7703]})
    with pytest.raises(ValueError, match="protected"):
        validate_splits({"train": [7901], "calibration": [7701], "test": [7703]})


def test_nonfinite_guard_threshold_rejected():
    for value in (float("nan"), float("inf"), 0, -1):
        with pytest.raises(ValueError):
            HarnessPolicy(max_age=value)


def test_live_guard_preempts_fixation_preserves_remediation_and_busy():
    from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv
    from blue.core.baselines import action_index, decode_index
    from blue.training.mappo_guide import ENV_KW
    env = CC4MARLEnv(seed=7706, steps=40, **ENV_KW)
    env.reset(seed=7706)
    agent = BLUE_AGENTS[0]
    mask = env.get_avail_agent_actions(0)
    hosts = [h for h in env.hostnames[agent] if mask[action_index(env, agent, h, "Analyse")]]
    preferred, overdue = hosts[:2]
    calls = []

    def fixed_hook(env, agent, cands, scored):
        calls.append(agent)
        return preferred if preferred in cands else cands[0]

    policy = HarnessPolicy(fixed_hook, max_age=10)
    env._tick = 20
    env.trackers[agent].last_analysis = {h: 20 for h in env.hostnames[agent]}
    env.trackers[agent].last_analysis[overdue] = 0
    chosen = policy.select(env, agent)
    assert decode_index(env, agent, chosen) == ("Analyse", overdue)
    assert calls == []
    assert policy.last_request["branch"] == "sweep_max_age"
    env.trackers[agent].state[preferred] = "CONFIRMED"
    chosen = policy.select(env, agent)
    assert decode_index(env, agent, chosen) == ("Remove", preferred)
    assert policy.last_request["branch"] == "remediation"
    assert calls == []
    env._awaiting[agent] = (preferred, "Remove")
    assert policy.select(env, agent) == 0
    assert policy.last_request["branch"] == "pending_wait"
    env.close()


def test_tree_scorer_grouped_fit_calibration_and_roundtrip(tmp_path):
    pytest.importorskip("sklearn")
    from blue.training.harness_ml import DEFAULT_SPLITS, fit
    from blue.harness.features import VERSION
    from blue.harness.scoring import MLScorer, sha256
    rng = np.random.default_rng(3)
    episode = np.repeat(sum(DEFAULT_SPLITS.values(), []), 400)
    X = rng.normal(size=(len(episode), len(NAMES))).astype(np.float32)
    for name in ("belief_CONFIRMED", "belief_VERIFY"):
        X[:, NAMES.index(name)] = 0
    # Nonlinear task that is poorly represented by a linear separator.
    y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(np.int8)
    data = tmp_path / "data.npz"
    labels = data.with_suffix(".labels.npz")
    np.savez_compressed(data, X=X, episode_seed=episode)
    np.savez_compressed(labels, y=y)
    data.with_suffix(".json").write_text(json.dumps({
        "version": VERSION, "features": NAMES, "splits": DEFAULT_SPLITS,
        "data_sha256": sha256(data), "labels_sha256": sha256(labels)}))
    report = fit(data, tmp_path / "model.pkl")
    assert report["hgb_test"]["pr_auc"] > report["logistic_test"]["pr_auc"] + 0.2
    assert set(report["per_test_episode"]) == {"7703", "7704"}
    scorer = MLScorer(tmp_path / "model.pkl")
    train = np.isin(episode, DEFAULT_SPLITS["train"])
    np.testing.assert_allclose(scorer.bundle["mean"], X[train].mean(0), atol=1e-6)
    env = VisibleOnlyEnv()
    row = scorer.actor_rows(env, "a", env.hostnames["a"])
    assert row.shape == (2, len(ACTOR_NAMES))
    assert np.isfinite(row).all()
    assert ((row[:, -2:] >= 0) & (row[:, -2:] <= 1)).all()
    again = scorer.actor_rows(env, "a", list(reversed(env.hostnames["a"])))
    np.testing.assert_equal(row, again[::-1])


def test_recorder_log_probability_matches_executed_distribution(monkeypatch):
    import torch as th
    from blue.training.harness_rl import HarnessRecorder, build_actor
    from blue.training import mappo_guide as mg
    monkeypatch.setattr(mg, "joint_context", lambda *args: np.zeros(mg.JOINT_CTX_DIM))
    scorer = SimpleNamespace(actor_rows=lambda env, agent, hs: np.zeros(
        (len(hs), len(ACTOR_NAMES)), dtype=np.float32))
    rec = HarnessRecorder(build_actor(), scorer, 0.25,
                          th.Generator().manual_seed(7), 0.5, "both")
    rec._policy_ref = None
    hosts = ["a", "b", "c"]
    scored = [(0.0, h) for h in hosts]
    selected = rec(SimpleNamespace(_tick=0), "agent", hosts, scored)
    row = rec.rows[0]
    assert hosts[row["choice"]] == selected
    assert row["logp"] == pytest.approx(-np.log(3))
    assert rec.sampled_disagree == int(selected != "c")
    assert rec.agree == 1  # greedy ranking differs from sampled override metric
    rec.close_episode([-1, -2])
    assert row["ret"] == pytest.approx(-1 - mg.GAMMA * 2)
    assert row["feats"].shape == (mg.MAX_CANDS, len(ACTOR_NAMES))


def test_ml_controls_change_only_score_columns():
    from blue.harness.policy import HarnessActorHook
    scorer = SimpleNamespace(actor_rows=lambda *args: np.ones((2, len(ACTOR_NAMES)), dtype=np.float32))
    for arm, expected in (("zero", [0, 0]), ("risk", [1, 0]), ("both", [1, 1])):
        F = HarnessActorHook(None, scorer, ml_inputs=arm).features(None, None, ["a", "b"])
        np.testing.assert_equal(F[:, :-2], 1)
        np.testing.assert_equal(F[0, -2:], expected)


def test_pilot_pairs_training_seeds_and_rejects_cache_drift(tmp_path, monkeypatch):
    import json
    from blue.training import harness_pilot as pilot
    cfg = {'development_eval_seeds': [8221, 8222], 'train_episode_seeds': [7706],
           'training_rng_seeds': [0, 1], 'arms': ['zero', 'risk', 'both'], 'max_age': 80, 'steps': 40}
    config = tmp_path / 'pilot.json'
    config.write_text(json.dumps(cfg))
    for seed in cfg['training_rng_seeds']:
        for arm in cfg['arms']:
            (tmp_path / f'{arm}_s{seed}').mkdir()
    monkeypatch.setattr(pilot, 'load_hook', lambda *args: (None, {}))
    calls = []
    def episode(policy, seed, steps, **kw):
        calls.append(seed)
        return {'seed': seed, 'return': -seed, 'coverage': 1, 'ticks': steps}
    monkeypatch.setattr(pilot.mg, 'run_team_episode', episode)
    pilot.evaluate(config, tmp_path, 'unused')
    report = json.loads((tmp_path / 'report.json').read_text())
    assert len(report['runs']) == 6 and len(report['matched_pairs']) == 2
    assert len(calls) == 16  # 2 baselines + 6 actors, 2 common episodes each
    pilot.evaluate(config, tmp_path, 'unused')
    assert len(calls) == 16
    cfg['steps'] = 41
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='baseline configuration drift'):
        pilot.evaluate(config, tmp_path, 'unused')
