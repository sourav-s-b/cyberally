"""Train the risk scorer: L2 logistic regression (numpy-only, no sklearn).

Predicts P(host compromised now | 17 Blue-visible features) from the
dataset collected by blue.training.risk_data. Privileged labels are training
targets only -- the scorer sees features at select time. Class-balanced
via pos_weight; deterministic full-batch gradient descent (fixed seed).

Writes results/risk_model_<id>.pkl (gitignored weights) + a committed
training manifest docs/proposals/manifests/risk-train-<id>.json with
config, AUC overall + on UNKNOWN-belief rows only, and dataset hash.
"""
import argparse
import hashlib
import json
import os
import pickle
import time

import numpy as np

MANIFEST_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "docs", "proposals", "manifests")


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def train_logreg(X, y, l2=1.0, lr=0.5, iters=500, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, X.shape[1])
    b = 0.0
    n_pos = y.sum()
    pos_weight = (len(y) - n_pos) / max(n_pos, 1)
    sw = np.where(y == 1, pos_weight, 1.0)
    for _ in range(iters):
        p = sigmoid(X @ w + b)
        err = (p - y) * sw
        w -= lr * (X.T @ err / len(y) + l2 * w / len(y))
        b -= lr * err.mean()
    return w, b


def auc(y, s):
    y = np.asarray(y, dtype=float)
    order = np.argsort(np.asarray(s, dtype=float))
    ys = y[order]
    n_pos = ys.sum()
    n_neg = len(ys) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, len(ys) + 1)
    return float((ranks[ys == 1].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser(description="Train risk scorer")
    ap.add_argument("--data", default="blue/results/risk_data_v1.npz")
    ap.add_argument("--out", default="blue/results/risk_model_v1.pkl")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--target", default="undetected",
                    choices=["now", "undetected"],
                    help="now: P(compromised); undetected: train on rows whose "
                    "belief is neither CONFIRMED nor VERIFY (the sweep "
                    "ordering target; rules pre-filter the rest)")
    cli = ap.parse_args()
    z = np.load(cli.data)
    X, y = z["X"].astype(np.float64), z["y"].astype(np.float64)
    if cli.target == "undetected":
        # Belief one-hot: UNKNOWN..VERIFY at cols 12..16.
        keep = ~((X[:, 15] == 1.0) | (X[:, 16] == 1.0))
        X, y = X[keep], y[keep]
    data_sha = hashlib.sha256(open(cli.data, "rb").read()).hexdigest()[:16]
    mean, std = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mean) / std
    w, b = train_logreg(Xs, y, l2=cli.l2, lr=cli.lr, iters=cli.iters,
                        seed=cli.seed)
    s = sigmoid(Xs @ w + b)
    metrics = {"auc_all": auc(y, s)}
    # Belief one-hot occupies feats 12..16 (UNKNOWN..VERIFY); UNKNOWN-only
    # AUC measures added value where rules have no evidence.
    unk = X[:, 12] == 1.0
    metrics["auc_unknown_only"] = auc(y[unk], s[unk])
    metrics["unknown_frac"] = float(unk.mean())
    model = {"w": w, "b": float(b), "mean": mean, "std": std,
             "n_feats": X.shape[1],
             "config": {"l2": cli.l2, "lr": cli.lr, "iters": cli.iters,
                        "seed": cli.seed}}
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    with open(cli.out, "wb") as f:
        pickle.dump(model, f)
    run_id = cli.run_id or time.strftime("risk-train-%Y%m%dT%H%M%S")
    manifest = {"run_id": run_id, "model_path": cli.out,
                "data": cli.data, "data_sha256": data_sha,
                "n_rows": len(y), "pos_rate": float(y.mean()),
                "metrics": metrics, "config": model["config"]}
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    with open(os.path.join(MANIFEST_DIR, f"{run_id}.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {cli.out} + manifest {run_id}: {metrics}")


if __name__ == "__main__":
    main()
