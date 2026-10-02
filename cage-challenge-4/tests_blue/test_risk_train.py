"""Risk-trainer math utilities on synthetic data (no sim).

Separability: deterministic logreg must reach ~1.0 AUC on well-separated
classes and ~0.5 on pure noise. Guards the numpy implementation, not the
simulator (simulator-side collection is exercised by blue_risk_data runs
whose manifests are committed).
"""
import numpy as np

from blue_train_risk import auc, sigmoid, train_logreg


def test_sigmoid_calibrated():
    assert sigmoid(np.array([0.0]))[0] == 0.5
    assert sigmoid(np.array([1000.0]))[0] > 0.999
    assert sigmoid(np.array([-1000.0]))[0] < 0.001


def test_logreg_separates_and_ranks_noise_at_chance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    w, b = train_logreg(Xs, y, iters=500, seed=0)
    assert auc(y, sigmoid(Xs @ w + b)) > 0.99
    noise = rng.normal(size=400)
    assert 0.4 < auc(y, noise) < 0.6


def test_auc_degenerate_labels():
    assert np.isnan(auc(np.zeros(10), np.arange(10)))
    assert np.isnan(auc(np.ones(10), np.arange(10)))
