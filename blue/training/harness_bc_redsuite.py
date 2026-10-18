"""Evaluate frozen finite-Red BC models against Lancer across other Red modes."""
from __future__ import annotations
import argparse, hashlib, json, math, statistics
from pathlib import Path
import numpy as np
from blue.core.baselines import MaskedRandomBaseline, SleepBaseline
from blue.harness.policy import HarnessPolicy
from blue.policies.ordered import LancerValues, OrderedPolicy
from blue.training import harness_dagger_confirm as confirm
from blue.training import mappo_guide as mg
from blue.training.harness_dagger import CONTROLS, MODEL_SEEDS, DaggerHook, load_model
from blue.training.harness_redesign import env_kwargs
from blue.training.harness_rl import check_seeds, source_identity

REDS = ("discovery", "verbose", "random", "sleep")
SEEDS = tuple(range(8401, 8433))
TRAINING = Path("blue/results/finite_red_dagger_bc_pilot_20261018")
CONFIRM = Path("blue/results/finite_red_dagger_bc_confirm_20261018")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def interval(values):
    from blue.training.harness_dagger import mean_interval
    return mean_interval(values)

def validate_seeds():
    check_seeds(list(SEEDS))
    from blue.training.seed_audit import build_used, reserved_conflicts
    used = build_used()
    self_path = str(Path(__file__).resolve().relative_to(Path.cwd()))
    conflicts = [(s, sorted(src for src in used[s] if src != self_path))
                 for s in SEEDS if s in used
                 and any(src != self_path for src in used[s])]
    if conflicts:
        raise ValueError(f"evaluation seeds already used: {conflicts}")
    reserved = reserved_conflicts(min(SEEDS), max(SEEDS))
    if reserved:
        raise ValueError(f"evaluation seeds reserved: {reserved}")

def make_plan(scorer):
    prior = confirm.load_plan(CONFIRM, scorer)
    return {
        "purpose": "cross-Red evaluation of fixed BC checkpoints; no retraining",
        "source_commit": source_identity(),
        "evaluator_sha256": digest(Path(__file__).as_posix()),
        "confirmation_plan_sha256": digest(CONFIRM / "frozen_plan.json"),
        "confirmation_analysis_sha256": digest(CONFIRM / "analysis.json"),
        "training_plan_sha256": digest(TRAINING / "frozen_plan.json"),
        "scorer_sha256": prior["scorer_sha256"],
        "red_agents": list(REDS), "evaluation_seeds": list(SEEDS),
        "model_seeds": list(MODEL_SEEDS), "steps": 400,
        "native_reward_only": True, "max_age": 80,
        "controls": list(CONTROLS),
        "models": {f"bc:s{s}": prior["models"][f"bc:s{s}"] for s in MODEL_SEEDS},
        "training_source_files": prior["training_source_files"],
        "primary_unit": "model training seed; paired episode differences averaged within model",
    }

def freeze(out, scorer):
    validate_seeds()
    plan = make_plan(scorer)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "frozen_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"out": str(out), "red_agents": REDS,
                      "seeds": [SEEDS[0], SEEDS[-1]], "n_seeds": len(SEEDS),
                      "episode_jobs": len(REDS) * (len(MODEL_SEEDS) + len(CONTROLS)) * len(SEEDS)}, indent=2))

def load_plan(out, scorer):
    plan = json.loads((Path(out) / "frozen_plan.json").read_text())
    if digest(Path(__file__).as_posix()) != plan["evaluator_sha256"]:
        raise ValueError("cross-Red evaluator source changed")
    if digest(CONFIRM / "frozen_plan.json") != plan["confirmation_plan_sha256"]:
        raise ValueError("finite confirmation plan changed")
    if digest(CONFIRM / "analysis.json") != plan["confirmation_analysis_sha256"]:
        raise ValueError("finite confirmation analysis changed")
    if digest(TRAINING / "frozen_plan.json") != plan["training_plan_sha256"]:
        raise ValueError("BC training plan changed")
    current = make_plan(scorer)
    if current["scorer_sha256"] != plan["scorer_sha256"] or current["models"] != plan["models"]:
        raise ValueError("scorer or trained BC models changed")
    return plan

def control_policy(name, seed):
    if name == "lancer":
        return OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5))
    if name == "guard":
        return HarnessPolicy(max_age=80)
    if name == "sleep":
        return SleepBaseline()
    if name == "masked_random":
        return MaskedRandomBaseline(seed=seed)
    raise ValueError(f"unknown control {name}")

def run_task(out, scorer, red, task):
    plan = load_plan(out, scorer)
    if red not in REDS:
        raise ValueError(f"unknown Red mode: {red}")
    model_seed = None
    if task in CONTROLS:
        model, arm, name = None, "controls", task
        identity = {"task": task, "red_agent": red, "seeds": list(SEEDS),
                    "frozen_plan_sha256": digest(Path(out) / "frozen_plan.json")}
    else:
        if task not in plan["models"]:
            raise ValueError(f"unknown task: {task}")
        model_seed = int(task.split(":s")[1])
        model, arm, name = plan["models"][task], "bc", f"s{model_seed}"
        identity = {"task": task, "red_agent": red,
                    "model_manifest_sha256": model["manifest_sha256"],
                    "actor_sha256": model["actor_sha256"], "seeds": list(SEEDS),
                    "frozen_plan_sha256": digest(Path(out) / "frozen_plan.json")}
    path = Path(out) / "evaluation" / red / arm / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    if path.exists():
        report = json.loads(path.read_text())
        if any(report.get(k) != v for k, v in identity.items()):
            raise ValueError(f"resume identity changed: {path}")
        rows = report.get("cells", {})
    if model is not None:
        model_dir = TRAINING / "models" / "bc" / f"s{model_seed}"
        actor, manifest = load_model(model_dir, plan["training_source_files"], plan["scorer_sha256"])
        if manifest["actor_sha256"] != model["actor_sha256"]:
            raise ValueError(f"BC artifact changed: {model_dir}")
    else:
        actor = None
    for seed in SEEDS:
        key = str(seed)
        if key in rows:
            continue
        if task in CONTROLS:
            result = mg.run_team_episode(control_policy(task, seed), seed, 400, **env_kwargs(red))
        else:
            hook = DaggerHook(scorer, actor=actor, beta=0.0,
                              seed=model_seed * 1_000_003 + seed * 2, collect=False)
            policy = HarnessPolicy(hook, max_age=80, record_requests=True)
            result = mg.run_team_episode(policy, seed, 400, hook=hook, recorder=hook,
                                         **env_kwargs(red))
            result["lancer_agreement"] = hook.agreement / max(hook.decisions, 1)
            result["actor_decisions"] = hook.decisions
        rows[key] = result
        path.write_text(json.dumps({**identity, "cells": rows}, indent=2) + "\n")
        print(f"{red} {task} seed={seed} return={result['return']:+.0f}", flush=True)
    print(f"complete {red} {task}: {path}", flush=True)

def analyze(out, scorer):
    plan = load_plan(out, scorer)
    finite = json.loads((CONFIRM / "analysis.json").read_text())
    finite_bc = finite["arms"]["bc"]
    results = {"finite": {
        "bc": {"mean": finite_bc["mean_return"], "p10": finite_bc["p10_return"],
               "min": finite_bc["min_return"],
               "mean_lancer_agreement": finite_bc["mean_lancer_agreement"],
               "mean_coverage": finite_bc["mean_coverage"]},
        "controls": finite["controls"],
        "bc_minus_lancer": finite_bc["paired_lancer"],
        "bc_minus_guard": finite_bc["paired_guard"],
        "source": "fresh finite-Red confirmation report; seeds 8253-8284",
    }}
    for red in REDS:
        controls, models = {}, {}
        for control in CONTROLS:
            p = Path(out) / "evaluation" / red / "controls" / f"{control}.json"
            report = json.loads(p.read_text())
            if report.get("seeds") != list(SEEDS) or set(report.get("cells", {})) != set(map(str, SEEDS)):
                raise ValueError(f"incomplete {red}/{control}")
            controls[control] = report["cells"]
        for model_seed in MODEL_SEEDS:
            p = Path(out) / "evaluation" / red / "bc" / f"s{model_seed}.json"
            report = json.loads(p.read_text())
            model = plan["models"][f"bc:s{model_seed}"]
            if (report.get("seeds") != list(SEEDS)
                    or report.get("actor_sha256") != model["actor_sha256"]
                    or report.get("model_manifest_sha256") != model["manifest_sha256"]
                    or set(report.get("cells", {})) != set(map(str, SEEDS))):
                raise ValueError(f"incomplete or mismatched {red}/bc:s{model_seed}")
            models[model_seed] = report["cells"]
        deltas, returns, agreement, coverage = {"lancer": [], "guard": []}, [], [], []
        per_model = {}
        for mseed, cells in models.items():
            vals = [float(cells[str(s)]["return"]) for s in SEEDS]
            returns.extend(vals)
            agreement.extend(float(cells[str(s)].get("lancer_agreement", math.nan)) for s in SEEDS)
            coverage.extend(float(cells[str(s)].get("coverage", math.nan)) for s in SEEDS)
            per_model[str(mseed)] = {"mean": statistics.mean(vals),
                                     "p10": float(np.quantile(vals, .1)), "min": min(vals)}
            for baseline in deltas:
                paired = [float(cells[str(s)]["return"] - controls[baseline][str(s)]["return"])
                          for s in SEEDS]
                deltas[baseline].append(statistics.mean(paired))
        results[red] = {
            "bc": {"mean": statistics.mean(returns), "p10": float(np.quantile(returns, .1)),
                   "min": min(returns), "mean_lancer_agreement": statistics.mean(agreement),
                   "mean_coverage": statistics.mean(coverage), "per_model": per_model},
            "controls": {name: {"mean": statistics.mean([c[str(s)]["return"] for s in SEEDS]),
                                "p10": float(np.quantile([c[str(s)]["return"] for s in SEEDS], .1)),
                                "min": min(c[str(s)]["return"] for s in SEEDS)}
                         for name, c in controls.items()},
            "bc_minus_lancer": {"training_seed_means": deltas["lancer"], **interval(deltas["lancer"])},
            "bc_minus_guard": {"training_seed_means": deltas["guard"], **interval(deltas["guard"])},
        }
    result = {"claim": "cross-Red evaluation of fixed finite-Red BC checkpoints",
              "training_seed_unit": list(MODEL_SEEDS), "evaluation_seeds": list(SEEDS),
              "red_modes": results,
              "passes_lancer_all_reds": all(v["bc_minus_lancer"]["low95"] > 0 for v in results.values()),
              "note": "Intervals are across five fixed model seeds and conditional on one shared 32-episode block per Red mode. No retraining variance is measured."}
    path = Path(out) / "analysis.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"analysis": str(path), "per_red": {
        red: {"bc_mean": vals["bc"]["mean"], "lancer_mean": vals["controls"]["lancer"]["mean"],
              "bc_minus_lancer": vals["bc_minus_lancer"], "bc_minus_guard": vals["bc_minus_guard"]}
        for red, vals in results.items()}, "passes_lancer_all_reds": result["passes_lancer_all_reds"]}, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scorer", required=True, type=Path)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--red")
    parser.add_argument("--task")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.out, args.scorer)
    elif args.red and args.task:
        run_task(args.out, args.scorer, args.red, args.task)
    elif args.analyze:
        analyze(args.out, args.scorer)
    else:
        parser.error("choose --freeze, --red with --task, or --analyze")

if __name__ == "__main__":
    main()
