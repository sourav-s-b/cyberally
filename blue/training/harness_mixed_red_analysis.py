"""Paired, two-Regime summary for mixed-Red investigator PPO."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

REDS = ("finite", "discovery")
TRAIN_SEEDS = (0, 1, 2, 3, 4)
EVAL_SEEDS = tuple(range(7641, 7657))


def read_report(path):
    return json.loads(Path(path).read_text())


def get_cells(report, arm, red):
    if report.get("red_agent") != red:
        raise ValueError(f"wrong Red regime in {report}")
    if report.get("config", {}).get("seeds") != list(EVAL_SEEDS):
        raise ValueError(f"wrong evaluation seeds in {report}")
    cells = report.get("cells", {}).get(arm, {})
    if set(cells) != set(map(str, EVAL_SEEDS)):
        raise ValueError(f"incomplete {arm} cells in {report}")
    return cells


def aggregate(root, rng=20261025, bootstrap_n=10000):
    root = Path(root)
    baseline = {}
    actor_returns, differences, coverage = {}, {}, {}
    actor_hashes = {}
    for red in REDS:
        report = read_report(root / f"baseline_{red}" / "report.json")
        guard = get_cells(report, "guard", red)
        lancer = get_cells(report, "lancer", red)
        baseline[red] = {
            arm: {seed: float(cells[seed]["return"]) for seed in cells}
            for arm, cells in (("guard", guard), ("lancer", lancer))}
        coverage[f"guard_{red}"] = [float(guard[s]["coverage"])
                                     for s in sorted(guard)]
        coverage[f"lancer_{red}"] = [float(lancer[s]["coverage"])
                                      for s in sorted(lancer)]
        actor_returns[red], differences[red] = {}, {}
        for train_seed in TRAIN_SEEDS:
            path = root / f"eval_s{train_seed}_{red}" / "report.json"
            report = read_report(path)
            cells = get_cells(report, "actor", red)
            if report.get("config", {}).get("model_dir") != str(root / f"s{train_seed}"):
                raise ValueError(f"wrong checkpoint recorded in {path}")
            actor_hash = report["artifacts_before_run"].get(
                str(root / f"s{train_seed}" / "actor.th"))
            if not actor_hash:
                raise ValueError(f"missing actor artifact hash in {path}")
            if actor_hashes.setdefault(train_seed, actor_hash) != actor_hash:
                raise ValueError(f"model hash drift for training seed {train_seed}")
            actor_returns[red][train_seed] = {
                seed: float(cells[seed]["return"]) for seed in cells}
            differences[red][train_seed] = {
                seed: float(cells[seed]["return"])
                - baseline[red]["guard"][seed]
                for seed in cells}
            coverage[f"actor_s{train_seed}_{red}"] = [
                float(cells[s]["coverage"]) for s in sorted(cells)]

    # Equal weight to each Red, each training replicate, and each paired
    # evaluation seed. The nested bootstrap resamples training seeds and
    # paired seed IDs, retaining the within-seed pairing across Red regimes.
    by_model = {
        train_seed: float(np.mean([
            differences[red][train_seed][str(seed)]
            for red in REDS for seed in EVAL_SEEDS]))
        for train_seed in TRAIN_SEEDS}
    values = np.array(list(by_model.values()), dtype=float)
    mean = float(values.mean())
    radius = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1)
                   / math.sqrt(len(values)))

    rng = np.random.default_rng(rng)
    boot = np.empty(bootstrap_n, dtype=float)
    for i in range(bootstrap_n):
        sampled_models = rng.integers(0, len(TRAIN_SEEDS), len(TRAIN_SEEDS))
        sampled_seed_ids = rng.integers(0, len(EVAL_SEEDS), len(EVAL_SEEDS))
        red_means = []
        for red in REDS:
            block = [[differences[red][TRAIN_SEEDS[m]][str(EVAL_SEEDS[s])]
                      for s in sampled_seed_ids] for m in sampled_models]
            red_means.append(float(np.mean(block)))
        boot[i] = float(np.mean(red_means))
    boot_ci = [float(x) for x in np.quantile(boot, [.025, .975])]
    per_red = {
        red: float(np.mean([differences[red][m][str(s)]
                            for m in TRAIN_SEEDS for s in EVAL_SEEDS]))
        for red in REDS}
    all_coverage = [x for values in coverage.values() for x in values]
    min_coverage = min(all_coverage)
    gate = (mean >= 5.0 and boot_ci[0] > 0.0
            and all(value >= -5.0 for value in per_red.values())
            and min_coverage >= 0.98)
    return {
        "study_id": "mixed-red-investigator-ppo-20261025",
        "primary_contrast": "actor minus age-80 coordinator, equal-weight Finite and Discovery",
        "training_rng_seeds": list(TRAIN_SEEDS),
        "evaluation_seeds": list(EVAL_SEEDS),
        "paired_mean_gain": mean,
        "per_training_seed_pooled_gain": by_model,
        "training_seed_t_interval_95": [mean - radius, mean + radius],
        "nested_paired_bootstrap_95": boot_ci,
        "bootstrap_replicates": bootstrap_n,
        "per_red_mean_gain": per_red,
        "baseline_means": {
            red: {arm: float(np.mean(list(rows.values())))
                  for arm, rows in arms.items()}
            for red, arms in baseline.items()},
        "min_host_coverage_across_actor_and_baseline_cells": min_coverage,
        "coverage_by_cell": coverage,
        "success_gate": {
            "passed": bool(gate),
            "requirements": {
                "pooled_mean_gain_at_least_5": mean >= 5.0,
                "bootstrap_lower_bound_above_zero": boot_ci[0] > 0.0,
                "each_red_gain_at_least_minus_5": all(
                    value >= -5.0 for value in per_red.values()),
                "minimum_coverage_at_least_0_98": min_coverage >= 0.98},
            "failure_action": "retain as exploratory; deploy fixed coordinator"},
        "claim_limit": "Five training RNG seeds and one fixed paired evaluation block; conditional evidence, not broad generalization proof.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = aggregate(args.root)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"paired_mean_gain": result["paired_mean_gain"],
                      "bootstrap_ci95": result["nested_paired_bootstrap_95"],
                      "per_red_mean_gain": result["per_red_mean_gain"],
                      "success_gate": result["success_gate"]}, indent=2))


if __name__ == "__main__":
    main()
