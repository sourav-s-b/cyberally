"""Paired finite-Red learning gate; training seeds are the replication unit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_model(model_dir, eval_path, training_seed, eval_seeds,
                   train_seeds=(7706, 7707, 7708, 7709)):
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text())
    if manifest.get("status") != "complete" or len(manifest.get("history", [])) != 32:
        raise ValueError("incomplete model manifest: " + str(model_dir))
    config = manifest["config"]
    expected = {"seed": training_seed, "iters": 32, "eps_per_iter": 4,
                "steps": 400, "red_agent": "finite", "state_shaping": "off",
                "train_seeds": list(train_seeds)}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("model configuration drift: " + str(model_dir))
    if (manifest.get("red_agent") != "finite" or
            manifest.get("objective") != "native team reward reported; unfiltered fresh on-policy rollouts"):
        raise ValueError("model attacker/objective drift: " + str(model_dir))
    actor_hash = sha256(model_dir / "actor.th")
    if actor_hash != manifest.get("actor_sha256"):
        raise ValueError("actor hash mismatch: " + str(model_dir))
    for source, digest in manifest.get("source_files", {}).items():
        path = Path(source)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file() or sha256(path) != digest:
            raise ValueError("source hash mismatch: " + str(path))
    scorer = Path(manifest["scorer"])
    if not scorer.is_file() or sha256(scorer) != manifest.get("scorer_sha256"):
        raise ValueError("scorer hash mismatch: " + str(scorer))

    report = json.loads(Path(eval_path).read_text())
    if (report.get("red_agent") != "finite" or
            "native" not in report.get("reward", "") or
            report.get("config", {}).get("seeds") != list(eval_seeds) or
            report.get("config", {}).get("model_dir") != str(model_dir)):
        raise ValueError("evaluation config mismatch: " + str(eval_path))
    cells = report.get("cells", {}).get("actor", {})
    expected_seeds = set(map(str, eval_seeds))
    if set(cells) != expected_seeds:
        raise ValueError("evaluation seed mismatch: " + str(eval_path))
    if any(cells[str(s)].get("artifact_sha256") != actor_hash for s in eval_seeds):
        raise ValueError("evaluation actor mismatch: " + str(eval_path))
    returns = {str(s): float(cells[str(s)]["return"]) for s in eval_seeds}
    if not np.isfinite(list(returns.values())).all():
        raise ValueError("nonfinite actor return")
    return manifest, returns


def aggregate(model_dirs, eval_paths, lancer_path, context_path,
              train_seeds, eval_seeds=tuple(range(7665, 7689))):
    if len(model_dirs) != len(train_seeds) or len(eval_paths) != len(train_seeds):
        raise ValueError("one model and evaluation are required per training seed")
    if len(train_seeds) < 2:
        raise ValueError("at least two training seeds are required")
    baseline = json.loads(Path(lancer_path).read_text())
    if baseline.get("key") != {"red_agent": "finite", "steps": 400,
                               "seeds": sorted(eval_seeds)}:
        raise ValueError("Lancer baseline config mismatch")
    if set(baseline.get("returns", {})) != set(map(str, eval_seeds)):
        raise ValueError("Lancer baseline seed mismatch")
    lancer = {str(s): float(baseline["returns"][str(s)]["return"])
              for s in eval_seeds}

    context = json.loads(Path(context_path).read_text())
    if context.get("red_agent") != "finite" or set(context.get("cells", {})) != {"sleep", "masked_random"}:
        raise ValueError("context must contain finite-Red Sleep and masked-random")
    if any(set(rows) != set(map(str, eval_seeds)) for rows in context["cells"].values()):
        raise ValueError("context evaluation seed mismatch")
    context_returns = {
        arm: {str(s): float(context["cells"][arm][str(s)]["return"])
              for s in eval_seeds}
        for arm in ("sleep", "masked_random")}

    models, seed_means, source_identity = [], [], None
    for seed, model_dir, eval_path in zip(train_seeds, model_dirs, eval_paths):
        manifest, returns = validate_model(model_dir, eval_path, seed, eval_seeds)
        identity = (manifest.get("source_commit"), manifest.get("source_files"))
        if source_identity is None:
            source_identity = identity
        elif identity != source_identity:
            raise ValueError("training replicas have different source identities")
        differences = {str(s): returns[str(s)] - lancer[str(s)] for s in eval_seeds}
        tail_n = max(1, math.ceil(len(eval_seeds) * .10))
        row = {
            "training_seed": seed,
            "model_dir": str(model_dir),
            "evaluation_report": str(eval_path),
            "episode_returns": returns,
            "paired_episode_differences_vs_lancer": differences,
            "mean_return": float(np.mean(list(returns.values()))),
            "worst_return": float(min(returns.values())),
            "bottom_10_percent_mean": float(np.mean(sorted(returns.values())[:tail_n])),
            "mean_paired_difference": float(np.mean(list(differences.values()))),
            "worst_paired_difference": float(min(differences.values())),
            "bottom_10_percent_paired_difference_mean": float(
                np.mean(sorted(differences.values())[:tail_n])),
        }
        models.append(row)
        seed_means.append(row["mean_paired_difference"])

    values = np.asarray(seed_means, dtype=float)
    mean = float(values.mean())
    radius = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) /
                   math.sqrt(len(values)))
    interval = {"n_training_seeds": len(values),
                "per_training_seed": values.tolist(), "mean": mean,
                "ci95": [mean - radius, mean + radius], "df": len(values) - 1}
    low, high = interval["ci95"]
    if low > 0:
        decision = "exploratory win over Lancer on this evaluation block"
    elif high < 0:
        decision = "paired 95% training-seed interval is below zero; Blue underperforms Lancer on this evaluation block"
    else:
        decision = "interval overlaps zero; add finite-Red training seeds before considering LLM Red"
    return {
        "attacker": "finite", "reward": "native team return",
        "training_rng_seeds": list(train_seeds),
        "evaluation_seeds": list(eval_seeds),
        "lancer_returns": lancer,
        "lancer_mean_return": float(np.mean(list(lancer.values()))),
        "lancer_worst_return": float(min(lancer.values())),
        "context_returns": context_returns,
        "models": models,
        "paired_training_seed_interval_vs_lancer": interval,
        "decision": decision,
        "claim_limit": ("training seeds are the replication unit; the interval is exploratory "
                        "and conditional on the shared evaluation block"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--evaluations", nargs="+", required=True)
    p.add_argument("--train-seeds", nargs="+", type=int, required=True)
    p.add_argument("--lancer", required=True)
    p.add_argument("--context", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = aggregate(a.models, a.evaluations, a.lancer, a.context,
                       a.train_seeds)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["paired_training_seed_interval_vs_lancer"], indent=2))
    print(result["decision"])


if __name__ == "__main__":
    main()
