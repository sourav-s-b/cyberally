"""Collect, train and evaluate harness ML with disjoint episode splits.

No random row split. Privileged targets live in a separate label artifact.
The protected final block is rejected for collection AND training.
Default seeds are previously used development seeds, not fresh/final tests.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

from blue.harness.features import BlueFeatures, NAMES, VERSION
from blue.harness.scoring import sha256

DEFAULT_SPLITS = {"train": [7706, 7707, 7708, 7709],
                  "calibration": [7701, 7702], "test": [7703, 7704]}


def validate_splits(splits):
    seen = set()
    for name in ("train", "calibration", "test"):
        seeds = splits[name]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("empty or duplicate episode split")
        if seen.intersection(seeds):
            raise ValueError("episode leakage between splits")
        if any(7809 <= int(s) <= 8200 for s in seeds):
            raise ValueError("protected final seed block")
        seen.update(seeds)


def paths(path):
    path = Path(path)
    return path, path.with_suffix(".labels.npz"), path.with_suffix(".json")


def collect(out, splits, steps):
    from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv
    from blue.policies.ordered import LancerValues, OrderedPolicy
    from blue.training.risk_data import true_compromised
    from blue.training.mappo_guide import ENV_KW

    validate_splits(splits)
    X, y, seeds, ticks, agents, hosts = [], [], [], [], [], []
    for split in ("train", "calibration", "test"):
        for seed in splits[split]:
            env = CC4MARLEnv(seed=seed, steps=steps, **ENV_KW)
            env.reset(seed=seed)
            # Frozen data-collection policy. No model fitted on these labels yet.
            policy = OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5), max_age=80)
            pipe = BlueFeatures()
            count = 0
            for _ in range(steps):
                truth = true_compromised(env)  # LABEL ONLY
                for agent in BLUE_AGENTS:
                    hs = env.hostnames[agent]
                    X.extend(pipe.rows(env, agent))
                    y.extend(int(h in truth) for h in hs)
                    seeds.extend([seed] * len(hs))
                    ticks.extend([env._tick] * len(hs))
                    agents.extend([agent] * len(hs))
                    hosts.extend(hs)
                    count += len(hs)
                actions = {a: int(policy.select(env, a)) for a in BLUE_AGENTS}
                _, _, terminal, truncated, _ = env.step(actions)
                if terminal or truncated:
                    break
            print(f"collected {split} seed={seed} rows={count}", flush=True)
            env.close()
    data, labels, manifest = paths(out)
    data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data, X=np.asarray(X, dtype=np.float32),
                        episode_seed=seeds, tick=ticks, agent=agents, host=hosts)
    np.savez_compressed(labels, y=np.asarray(y, dtype=np.int8))
    meta = {"version": VERSION, "features": NAMES, "splits": splits,
            "steps": steps, "env_kw": ENV_KW,
            "collector": "LancerValues(fruitless_decay=0.5)+max_age=80",
            "seed_status": "previously used development seeds; not final evaluation",
            "labels": "privileged red-session presence, training/evaluation only",
            "data_sha256": sha256(data), "labels_sha256": sha256(labels),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    meta["source_files"] = {str(p): sha256(p) for p in (
        Path(__file__), Path("blue/harness/features.py"),
        Path("blue/core/wrapper.py"), Path("blue/policies/ordered.py"))}
    manifest.write_text(json.dumps(meta, indent=2) + "\n")


def metrics(y, p):
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    both = len(np.unique(y)) == 2
    return {"n": len(y), "prevalence": float(np.mean(y)),
            "pr_auc": float(average_precision_score(y, p)) if both else None,
            "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0, 1]))}


def episode_summary(episodes):
    """Equal weight per episode, never per host-tick; conditional on one fit.

    Paired t intervals describe episode variation in this development suite,
    not training-seed uncertainty or independent/fresh model confirmation.
    """
    from scipy.stats import t
    out = {"unit": "test episode, equally weighted", "n_episodes": len(episodes),
           "limitation": "Conditional on one fixed training/calibration split; reused development episodes, not fresh confirmation. No row-based CI.",
           "paired_hgb_minus_logistic": {}}
    for key in ("pr_auc", "roc_auc", "brier", "log_loss"):
        rows = [e for e in episodes.values() if e['hgb'][key] is not None and e['logistic'][key] is not None]
        delta = np.asarray([e['hgb'][key]-e['logistic'][key] for e in rows])
        if not len(delta):
            continue
        mean = float(delta.mean())
        radius = float(t.ppf(.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None
        out['paired_hgb_minus_logistic'][key] = {
            "n_episodes": len(delta), "mean": mean,
            "ci95_conditional_episode_t": [mean-radius, mean+radius] if radius is not None else None,
            "hgb_macro_mean": float(np.mean([e['hgb'][key] for e in rows])),
            "logistic_macro_mean": float(np.mean([e['logistic'][key] for e in rows]))}
    return out


def fit(data_path, out, random_state=0):
    from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn import __version__ as sklearn_version
    import scipy

    data, labels, manifest = paths(data_path)
    meta = json.loads(manifest.read_text())
    if meta["version"] != VERSION or tuple(meta["features"]) != NAMES:
        raise ValueError("dataset feature contract mismatch")
    if sha256(data) != meta["data_sha256"] or sha256(labels) != meta["labels_sha256"]:
        raise ValueError("dataset hash mismatch")
    splits = meta["splits"]
    validate_splits(splits)
    with np.load(data) as z:
        X, episode = z["X"], z["episode_seed"]
    with np.load(labels) as z:
        y = z["y"]
    if (X.shape != (len(y), len(NAMES)) or len(episode) != len(y)
            or np.isinf(X).any() or not np.isin(y, [0, 1]).all()):
        raise ValueError("invalid dataset")
    if set(np.unique(episode)) != set(sum(splits.values(), [])):
        raise ValueError("unassigned or missing episode in dataset")
    masks = {k: np.isin(episode, v) for k, v in splits.items()}
    # Sweep target only: CONFIRMED/VERIFY are handled by rules, not learned ranking.
    eligible = (X[:, NAMES.index("belief_CONFIRMED")] == 0) & (X[:, NAMES.index("belief_VERIFY")] == 0)
    masks = {k: v & eligible for k, v in masks.items()}
    for mask in masks.values():
        if len(np.unique(y[mask])) != 2:
            raise ValueError("each split must contain both classes")
    train = masks["train"]
    cal = masks["calibration"]
    test = masks["test"]
    # Fit normalization on TRAIN only; no all-NaN column warning/zero scale.
    counts = np.isfinite(X[train]).sum(0)
    mean = np.divide(np.nansum(X[train], axis=0), counts,
                     out=np.zeros(len(NAMES)), where=counts > 0)
    centered = X[train] - mean
    scale = np.sqrt(np.divide(np.nansum(centered ** 2, axis=0), counts,
                              out=np.ones(len(NAMES)), where=counts > 0))
    scale = np.where(scale > 1e-6, scale, 1.0)
    params = dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=40,
                  learning_rate=0.08, l2_regularization=2.0,
                  early_stopping=False, random_state=random_state)
    tree = HistGradientBoostingClassifier(**params).fit(X[train], y[train])
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        tree.predict_proba(X[cal])[:, 1], y[cal])
    filled = np.where(np.isnan(X), mean, X)
    baseline = LogisticRegression(C=1.0, max_iter=400, random_state=random_state).fit(
        ((filled - mean) / scale)[train], y[train])
    baseline_cal = IsotonicRegression(out_of_bounds="clip").fit(
        baseline.predict_proba(((filled - mean) / scale)[cal])[:, 1], y[cal])
    benign = np.flatnonzero(train & (y == 0))
    rng = np.random.default_rng(random_state)
    if len(benign) > 20000:
        benign = rng.choice(benign, 20000, replace=False)
    anomaly = IsolationForest(n_estimators=100, max_samples=256,
                              random_state=random_state, n_jobs=1).fit(filled[benign])
    reference = np.sort(-anomaly.score_samples(filled[benign]))
    runtime = {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn_version}
    b = {"version": VERSION, "features": NAMES,
         "score_semantics": "calibrated-compromise-risk+benign-novelty-percentile",
         "risk_model": tree, "calibrator": calibrator, "anomaly_model": anomaly,
         "benign_score_reference": reference, "mean": mean, "scale": scale,
         "dataset_manifest": meta, "sklearn_version": sklearn_version, "runtime": runtime}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(b, f)
    p = calibrator.predict(tree.predict_proba(X[test])[:, 1])
    bp = baseline_cal.predict(baseline.predict_proba(((filled - mean) / scale)[test])[:, 1])
    novelty = np.searchsorted(reference, -anomaly.score_samples(filled[test]), side="right") / len(reference)
    report = {"version": VERSION, "splits": splits, "model_sha256": sha256(out),
              "dataset_sha256": meta["data_sha256"], "parameters": params,
              "sklearn_version": sklearn_version, "runtime": runtime,
              "source_files": {str(p): sha256(p) for p in (
                  Path(__file__), Path("blue/harness/features.py"), Path("blue/harness/scoring.py"))},
              "hgb_test": metrics(y[test], p), "logistic_test": metrics(y[test], bp),
              "novelty_test": {"pr_auc": metrics(y[test], novelty)["pr_auc"],
                               "semantics": "benign-reference percentile; not probability"},
              "per_test_episode": {}, "claim": "development detector comparison only; no RL improvement established"}
    for seed in splits["test"]:
        m = episode[test] == seed
        report["per_test_episode"][str(seed)] = {"hgb": metrics(y[test][m], p[m]),
                                                  "logistic": metrics(y[test][m], bp[m])}
        novelty_metrics = metrics(y[test][m], novelty[m])
        report['per_test_episode'][str(seed)]['novelty'] = {
            k: novelty_metrics[k] for k in ('n', 'prevalence', 'pr_auc', 'roc_auc')}
    report['episode_summary'] = episode_summary(report['per_test_episode'])
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("collect", "fit"))
    ap.add_argument("--data", default="blue/results/gpt_harness/data.npz")
    ap.add_argument("--model", default="blue/results/gpt_harness/scorer.pkl")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    for name, seeds in DEFAULT_SPLITS.items():
        ap.add_argument("--" + name + "-seeds", type=int, nargs="+", default=seeds)
    args = ap.parse_args()
    if args.mode == "collect":
        collect(args.data, {k: getattr(args, k + "_seeds") for k in DEFAULT_SPLITS}, args.steps)
    else:
        fit(args.data, args.model, args.seed)


if __name__ == "__main__":
    main()
