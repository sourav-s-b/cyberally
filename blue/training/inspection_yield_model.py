"""Feasibility fit for *observed* new-confirmation yield of selectable Analyse.

Only pre-request Blue inputs from completed ``sweep_scorer`` actions are used;
forced verification and coverage-guard actions cannot be chosen by the
learned hook and must not inflate its predictive score. The
episode partitions are frozen first by ``inspection_yield_audit``; host rows
are never randomly split. This is an outcome-prediction diagnostic, not a
counterfactual action-value estimator and not a deployed policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from blue.harness.consequence_context import NAMES_EXTRA, consequence_context
from blue.harness.scoring import sha256
from blue.training.inspection_yield_audit import read_jsonl

VERSION = "blue-inspection-yield-model-v1"
EXTRA_NAMES = ("current_compromise_risk", "novelty_percentile",
               "lancer_base_score")


def feature_vector(row, agents, *, horizon=400, max_age=80):
    pre = row["pre_input"]
    values = pre["feature_values"] + pre["event_memory_values"] + [
        pre["risk"], pre["novelty_percentile"], pre["base_score"]]
    values += list(consequence_context(
        row["host"], pre["feature_values"], pre["event_memory_values"],
        horizon=horizon, max_age=max_age))
    values += [float(row["agent"] == agent) for agent in agents]
    return [np.nan if v is None else float(v) for v in values]


def load_partition(root, seeds, agents, *, horizon=400, max_age=80):
    X, y, group, risk = [], [], [], []
    for seed in seeds:
        rows = read_jsonl(root / f"episode_{seed}_requests.jsonl")
        for row in rows:
            outcome = row["outcome"]
            if (outcome["status"] != "success" or
                    row["pre_input"]["decision_branch"] != "sweep_scorer"):
                continue  # only completed decisions the learned hook could alter
            X.append(feature_vector(row, agents, horizon=horizon,
                                    max_age=max_age))
            y.append(int(outcome["new_confirmation"]))
            group.append(int(seed))
            risk.append(float(row["pre_input"]["risk"]))
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    group = np.asarray(group, dtype=np.int32)
    risk = np.asarray(risk, dtype=np.float32)
    if X.ndim != 2 or len(X) == 0 or not np.isfinite(np.nan_to_num(X)).all():
        raise ValueError("empty or invalid selected-action table")
    return X, y, group, risk


def episode_metrics(y, p, group):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    out = []
    for seed in sorted(set(int(g) for g in group)):
        mask = group == seed
        yt, pt = y[mask], p[mask]
        prevalence = float(yt.mean())
        out.append({
            "seed": seed, "n_completed": len(yt), "n_confirmed": int(yt.sum()),
            "prevalence": prevalence,
            "pr_auc": (float(average_precision_score(yt, pt))
                       if len(set(yt)) == 2 else None),
            "roc_auc": (float(roc_auc_score(yt, pt))
                        if len(set(yt)) == 2 else None),
            "brier": float(brier_score_loss(yt, pt)),
        })
    return out


def fit(root, out):
    from sklearn.ensemble import HistGradientBoostingClassifier
    import sklearn
    import joblib

    root, out = Path(root), Path(out)
    split_path = root / "episode_partitions.json"
    split = json.loads(split_path.read_text())
    if split["source_summary_sha256"] != sha256(root / "summary.json"):
        raise ValueError("episode partition source changed")
    summary = json.loads((root / "summary.json").read_text())
    # The frozen architecture plan uses max_age=80. The collector's manifest
    # records this in the policy string; do not infer another scale silently.
    if "max_age=80.0" not in summary["manifest"]["policy"]:
        raise ValueError("unexpected frozen coverage threshold")
    names = (summary["manifest"]["feature_names"] +
             summary["manifest"]["history_names"] + list(EXTRA_NAMES) +
             list(NAMES_EXTRA))
    agents = [f"blue_agent_{i}" for i in range(5)]
    names += ["agent_" + a for a in agents]
    parts = {}
    for name, block in split["partitions"].items():
        parts[name] = load_partition(
            root, block["episode_seeds"], agents,
            horizon=int(summary["manifest"]["steps"]),
            max_age=80.0)
    if set(parts) != {"train", "validation", "audit"}:
        raise ValueError("need train/validation/audit episode partitions")
    X, y, group, _ = parts["train"]
    if X.shape[1] != len(names) or len(set(y)) < 2:
        raise ValueError("feature geometry or training label balance invalid")
    # Each episode contributes equal *total* fit weight. This does not make
    # within-episode rows independent for uncertainty calculations.
    counts = {int(seed): int((group == seed).sum()) for seed in set(group)}
    weight = np.asarray([1. / counts[int(seed)] for seed in group])
    weight *= len(weight) / weight.sum()
    hyperparameters = {
        "max_iter": 100, "max_leaf_nodes": 15,
        "min_samples_leaf": 40, "learning_rate": .05,
        "l2_regularization": 1., "random_state": 0,
    }
    context_start = len(summary["manifest"]["feature_names"]) + \
        len(summary["manifest"]["history_names"]) + len(EXTRA_NAMES)
    columns = {
        "standard_inputs": (list(range(context_start)) +
                            list(range(context_start + len(NAMES_EXTRA), len(names)))),
        "with_consequence_context": list(range(len(names))),
    }
    models = {}
    for variant, use in columns.items():
        model = HistGradientBoostingClassifier(**hyperparameters)
        model.fit(X[:, use], y, sample_weight=weight)
        models[variant] = model
    report = {
        "version": VERSION,
        "claim_scope": "selected, unguarded sweep outcome; not causal inspection value",
        "partition_sha256": sha256(split_path),
        "input_summary_sha256": sha256(root / "summary.json"),
        "feature_names": names,
        "model": "HistGradientBoostingClassifier",
        "hyperparameters": hyperparameters,
        "variants": {name: [names[i] for i in use]
                     for name, use in columns.items()},
        "runtime": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "fitter_sha256": sha256(__file__),
        "consequence_source_sha256": sha256(
            Path(__file__).parents[1] / "harness" / "consequence_context.py"),
        "public_reward_schedule_sha256": sha256(
            Path(__file__).parents[2] / "cage-challenge-4" / "CybORG" /
            "Shared" / "BlueRewardMachine.py"),
        "episode_results": {},
    }
    for name, (features, labels, groups, risk) in parts.items():
        report["episode_results"][name] = {
            variant: episode_metrics(
                labels, models[variant].predict_proba(features[:, use])[:, 1],
                groups)
            for variant, use in columns.items()
        }
        report["episode_results"][name]["compromise_risk_reference"] = (
            episode_metrics(labels, risk, groups))
    out.mkdir(parents=True, exist_ok=True)
    report["model_sha256"] = {}
    for variant, model in models.items():
        model_file = out / f"inspection_yield_{variant}.joblib"
        joblib.dump(model, model_file)
        report["model_sha256"][variant] = sha256(model_file)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    report = fit(a.input, a.out)
    for part, block in report["episode_results"].items():
        for variant, ep in block.items():
            valid = [r["pr_auc"] for r in ep if r["pr_auc"] is not None]
            if valid:
                print(f"{part}/{variant}: episodes={len(ep)} "
                      f"mean_episode_PR_AUC={sum(valid)/len(valid):.3f}")
            else:
                print(f"{part}/{variant}: no two-class episodes")


if __name__ == "__main__":
    main()
