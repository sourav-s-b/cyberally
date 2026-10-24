"""Calibrated supervised risk and separate benign-reference ML novelty.

Pickles are trusted local artifacts only. Neither score is a clean-host label
or a proof that an action improves return. sklearn is imported lazily.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np

from blue.harness.features import BlueFeatures, NAMES, VERSION, neural_rows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_bundle(bundle):
    if bundle.get("version") != VERSION or tuple(bundle.get("features", ())) != NAMES:
        raise ValueError("scorer preprocessing version mismatch")
    neural_rows(np.zeros((1, len(NAMES))), bundle["mean"], bundle["scale"])
    if bundle.get("score_semantics") != "calibrated-compromise-risk+benign-novelty-percentile":
        raise ValueError("unknown score semantics")
    for name in ("risk_model", "anomaly_model"):
        if bundle[name].n_features_in_ != len(NAMES):
            raise ValueError("model geometry mismatch")
    ref = np.asarray(bundle["benign_score_reference"])
    if ref.ndim != 1 or len(ref) == 0 or not np.isfinite(ref).all() or (np.diff(ref) < 0).any():
        raise ValueError("invalid benign anomaly reference")


class MLScorer:
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self.sha256 = sha256(path)
        with open(path, "rb") as f:
            self.bundle = pickle.load(f)
        validate_bundle(self.bundle)
        self.features = BlueFeatures()
        self.reset()

    def reset(self):
        self.features.reset()
        self._cache = {}
        self._env = None

    def observe(self, env, agent):
        if self._env is not env:
            self.reset()
            self._env = env
        return self.features.rows(env, agent)

    def score(self, env, agent, hosts=None):
        X = self.observe(env, agent)
        key = (agent, int(env._tick))
        cached = self._cache.get(agent)
        if cached is None or cached[0] != key:
            b = self.bundle
            risk = b["calibrator"].predict(b["risk_model"].predict_proba(X)[:, 1])
            filled = np.where(np.isnan(X), b["mean"], X)
            novelty = -b["anomaly_model"].score_samples(filled)
            # Percentile against benign TRAIN rows; not P(compromised).
            ref = b["benign_score_reference"]
            novelty = np.searchsorted(ref, novelty, side="right") / len(ref)
            if (not np.isfinite(risk).all() or not np.isfinite(novelty).all()
                    or (risk < 0).any() or (risk > 1).any()):
                raise ValueError("nonfinite ML output")
            table = {h: (float(p), float(n)) for h, p, n in
                     zip(env.hostnames[agent], risk, novelty)}
            self._cache[agent] = (key, table)
        table = self._cache[agent][1]
        hosts = env.hostnames[agent] if hosts is None else hosts
        return np.asarray([table[h] for h in hosts], dtype=np.float32).reshape(-1, 2)

    def actor_rows(self, env, agent, hosts):
        self.observe(env, agent)
        X = self.features.rows(env, agent, hosts)
        scaled = neural_rows(X, self.bundle["mean"], self.bundle["scale"])
        return np.concatenate((scaled, np.isfinite(X).astype(np.float32),
                               self.score(env, agent, hosts)), axis=1)

    def _predict_all(self, env, agent):
        """Compatibility for diagnostic-only legacy 15-column checkpoints."""
        return dict(zip(env.hostnames[agent], self.score(env, agent)[:, 0]))
