"""Analyze cross-Red BC reports and reuse finite results for verbose's log-only variant."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

from blue.training import harness_bc_redsuite as suite
from blue.training import harness_dagger_confirm as confirm
from blue.training.harness_dagger import MODEL_SEEDS

MODES = ("discovery", "random", "sleep")
SEEDS = suite.SEEDS
CONTROLS = suite.CONTROLS


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_controls(controls):
    result = {}
    for name, cells in controls.items():
        values = [float(cells[str(seed)]["return"]) for seed in SEEDS]
        result[name] = {"mean": statistics.mean(values),
                        "p10": float(np.quantile(values, .1)), "min": min(values)}
    return result


def analyze_mode(out, red, plan):
    controls, models = {}, {}
    plan_hash = file_hash(Path(out) / "frozen_plan.json")
    expected = list(SEEDS)
    for control in CONTROLS:
        path = Path(out) / "evaluation" / red / "controls" / f"{control}.json"
        report = json.loads(path.read_text())
        if (report.get("task") != control or report.get("red_agent") != red
                or report.get("seeds") != expected
                or report.get("frozen_plan_sha256") != plan_hash
                or set(report.get("cells", {})) != set(map(str, SEEDS))):
            raise ValueError(f"incomplete or mismatched {red}/{control}")
        controls[control] = report["cells"]
    deltas, returns, agreements, coverage = {"lancer": [], "guard": []}, [], [], []
    per_model = {}
    for model_seed in MODEL_SEEDS:
        task = f"bc:s{model_seed}"
        model = plan["models"][task]
        path = Path(out) / "evaluation" / red / "bc" / f"s{model_seed}.json"
        report = json.loads(path.read_text())
        if (report.get("task") != task or report.get("red_agent") != red
                or report.get("seeds") != expected
                or report.get("frozen_plan_sha256") != plan_hash
                or report.get("actor_sha256") != model["actor_sha256"]
                or report.get("model_manifest_sha256") != model["manifest_sha256"]
                or set(report.get("cells", {})) != set(map(str, SEEDS))):
            raise ValueError(f"incomplete or mismatched {red}/{task}")
        cells = report["cells"]
        vals = [float(cells[str(seed)]["return"]) for seed in SEEDS]
        returns.extend(vals)
        agreements.extend(float(cells[str(seed)].get("lancer_agreement", math.nan)) for seed in SEEDS)
        coverage.extend(float(cells[str(seed)].get("coverage", math.nan)) for seed in SEEDS)
        per_model[str(model_seed)] = {"mean": statistics.mean(vals),
                                      "p10": float(np.quantile(vals, .1)), "min": min(vals)}
        for baseline in deltas:
            paired = [float(cells[str(seed)]["return"] - controls[baseline][str(seed)]["return"])
                      for seed in SEEDS]
            deltas[baseline].append(statistics.mean(paired))
    from blue.training.harness_dagger import mean_interval
    return {
        "bc": {"mean": statistics.mean(returns), "p10": float(np.quantile(returns, .1)),
               "min": min(returns), "mean_lancer_agreement": statistics.mean(agreements),
               "mean_coverage": statistics.mean(coverage), "per_model": per_model},
        "controls": summarize_controls(controls),
        "bc_minus_lancer": {"training_seed_means": deltas["lancer"], **mean_interval(deltas["lancer"])},
        "bc_minus_guard": {"training_seed_means": deltas["guard"], **mean_interval(deltas["guard"])},
    }


def finite_result():
    data = json.loads((suite.CONFIRM / "analysis.json").read_text())
    bc = data["arms"]["bc"]
    return {
        "bc": {"mean": bc["mean_return"], "p10": bc["p10_return"],
               "min": bc["min_return"], "mean_lancer_agreement": bc["mean_lancer_agreement"],
               "mean_coverage": bc["mean_coverage"]},
        "controls": data["controls"], "bc_minus_lancer": bc["paired_lancer"],
        "bc_minus_guard": bc["paired_guard"],
        "evaluation_seeds": data["evaluation_seeds"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scorer", required=True, type=Path)
    args = parser.parse_args()
    plan = suite.load_plan(args.out, args.scorer)
    modes = {red: analyze_mode(args.out, red, plan) for red in MODES}
    finite = finite_result()
    modes["finite"] = {key: value for key, value in finite.items() if key != "evaluation_seeds"}
    modes["verbose"] = {
        **modes["finite"], "behavior_alias_of": "finite",
        "evaluation_seeds": finite["evaluation_seeds"],
        "equivalence_evidence": {
            "variant_source": "cage-challenge-4/CybORG/Agents/SimpleAgents/FSMRedVariants.py",
            "base_source": "cage-challenge-4/CybORG/Agents/SimpleAgents/FiniteStateRedAgent.py",
            "variant_source_sha256": file_hash("cage-challenge-4/CybORG/Agents/SimpleAgents/FSMRedVariants.py"),
            "base_source_sha256": file_hash("cage-challenge-4/CybORG/Agents/SimpleAgents/FiniteStateRedAgent.py"),
            "reason": "VerboseFSRed only changes print_action_output and print_obs_output in __init__; it overrides no Red decision or transition methods.",
        },
    }
    passes = all(modes[name]["bc_minus_lancer"]["low95"] > 0
                 for name in ("finite", "discovery", "random", "sleep"))
    result = {
        "claim": "fixed finite-Red BC checkpoints evaluated across distinct in-tree Red behaviors",
        "red_modes": modes, "passes_lancer_all_distinct_regimes": passes,
        "evaluation_seeds_by_regime": {
            "finite": finite["evaluation_seeds"], "verbose": finite["evaluation_seeds"],
            **{red: list(SEEDS) for red in MODES}},
        "training_seed_unit": list(MODEL_SEEDS),
        "note": "Intervals are across five fixed BC models and conditional on one episode block per behavior; no retraining variance is measured. Verbose is a logging-only alias of finite, not independent evidence.",
    }
    path = args.out / "analysis.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"analysis": str(path), "passes_lancer_all_distinct_regimes": passes,
                      "per_red": {red: {"bc_mean": r["bc"]["mean"],
                                        "lancer_mean": r["controls"]["lancer"]["mean"],
                                        "bc_minus_lancer": r["bc_minus_lancer"]}
                                  for red, r in modes.items()}}, indent=2))


if __name__ == "__main__":
    main()
