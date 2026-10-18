"""Fresh-seed confirmation of frozen finite-Red BC and DAgger checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

from blue.core.baselines import MaskedRandomBaseline, SleepBaseline
from blue.harness.policy import HarnessPolicy
from blue.policies.ordered import LancerValues, OrderedPolicy
from blue.training import mappo_guide as mg
from blue.training.harness_dagger import (
    CONTROLS, MODEL_SEEDS, DaggerHook, digest, load_model, mean_interval,
)
from blue.training.harness_redesign import env_kwargs
from blue.training.harness_rl import check_seeds, source_identity

EVAL_SEEDS = tuple(range(8253, 8285))
ARMS = ("bc", "dagger")
BASE = Path("blue/results/finite_red_dagger_bc_pilot_20261018")
CONFIRM_SOURCE = Path(__file__).as_posix()


def make_plan(scorer_path):
    frozen = json.loads((BASE / "frozen_plan.json").read_text())
    if frozen["source_commit"] != source_identity():
        raise ValueError("checkout commit differs from the frozen model source commit")
    for rel, expected in frozen["source_files"].items():
        if digest(rel) != expected:
            raise ValueError(f"frozen model source changed: {rel}")
    scorer_hash = digest(scorer_path)
    if scorer_hash != frozen["scorer_sha256"]:
        raise ValueError("scorer differs from the frozen training scorer")
    models = {}
    for arm in ARMS:
        for model_seed in MODEL_SEEDS:
            model_dir = BASE / "models" / arm / f"s{model_seed}"
            manifest_path = model_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") != "complete":
                raise ValueError(f"incomplete model: {model_dir}")
            load_model(model_dir, frozen["source_files"], scorer_hash)
            models[f"{arm}:s{model_seed}"] = {
                "manifest_sha256": digest(manifest_path),
                "actor_sha256": manifest["actor_sha256"],
            }
    return {
        "purpose": "fresh-seed confirmation of fixed finite-Red BC and DAgger models",
        "source_commit": source_identity(),
        "confirmation_source_sha256": digest(CONFIRM_SOURCE),
        "training_artifacts": str(BASE),
        "training_frozen_plan_sha256": digest(BASE / "frozen_plan.json"),
        "training_source_files": frozen["source_files"],
        "scorer_sha256": scorer_hash,
        "red_agent": "finite",
        "native_reward_only": True,
        "max_age": 80,
        "steps": 400,
        "model_seeds": list(MODEL_SEEDS),
        "models": models,
        "evaluation_seeds": list(EVAL_SEEDS),
        "controls": list(CONTROLS),
        "primary_unit": "training model seed; paired episode differences averaged within model",
    }


def validate_fresh_seeds():
    check_seeds(list(EVAL_SEEDS))
    from blue.training.seed_audit import build_used, reserved_conflicts

    used = build_used()
    self_path = str(Path(__file__).resolve().relative_to(Path.cwd()))
    conflicts = [(seed, sorted(src for src in used[seed] if src != self_path))
                 for seed in EVAL_SEEDS if seed in used
                 and any(src != self_path for src in used[seed])]
    if conflicts:
        raise ValueError(f"confirmation seeds are already used: {conflicts}")
    reserved = reserved_conflicts(min(EVAL_SEEDS), max(EVAL_SEEDS))
    if reserved:
        raise ValueError(f"confirmation seeds overlap reserved block: {reserved}")


def freeze(out, scorer_path):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"confirmation output is not empty: {out}")
    validate_fresh_seeds()
    plan = make_plan(scorer_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / "frozen_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"out": str(out), "evaluation_seeds": list(EVAL_SEEDS),
                      "tasks": [*CONTROLS,
                                *(f"{a}:s{s}" for a in ARMS for s in MODEL_SEEDS)]}, indent=2))


def load_plan(out, scorer_path):
    out = Path(out)
    plan_path = out / "frozen_plan.json"
    plan = json.loads(plan_path.read_text())
    current = make_plan(scorer_path)
    if plan != current:
        raise ValueError("confirmation plan, frozen source, scorer, or model hashes changed")
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


def run_task(out, scorer_path, task):
    plan = load_plan(out, scorer_path)
    out = Path(out)
    if task in CONTROLS:
        path = out / "evaluation" / "controls" / f"{task}.json"
        identity = {"task": task, "seeds": list(EVAL_SEEDS),
                    "frozen_plan_sha256": digest(out / "frozen_plan.json")}
    else:
        arm, model_id = task.split(":s", 1)
        model_seed = int(model_id)
        if arm not in ARMS or model_seed not in MODEL_SEEDS:
            raise ValueError(f"invalid task: {task}")
        path = out / "evaluation" / arm / f"s{model_seed}.json"
        model_info = plan["models"][task]
        identity = {"task": task, "arm": arm, "model_seed": model_seed,
                    "model_manifest_sha256": model_info["manifest_sha256"],
                    "actor_sha256": model_info["actor_sha256"],
                    "seeds": list(EVAL_SEEDS),
                    "frozen_plan_sha256": digest(out / "frozen_plan.json")}
        model_dir = BASE / "models" / arm / f"s{model_seed}"
        actor, manifest = load_model(model_dir, plan["training_source_files"],
                                     plan["scorer_sha256"])
        if manifest["actor_sha256"] != identity["actor_sha256"]:
            raise ValueError(f"model artifact changed: {model_dir}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        report = json.loads(path.read_text())
        for key, value in identity.items():
            if report.get(key) != value:
                raise ValueError(f"resume identity mismatch for {path}: {key}")
        rows = report.get("cells", {})
    else:
        report = {**identity, "cells": {}}
        rows = report["cells"]

    for seed in EVAL_SEEDS:
        key = str(seed)
        if key in rows:
            continue
        if task in CONTROLS:
            policy = control_policy(task, seed)
            result = mg.run_team_episode(policy, seed, 400, **env_kwargs("finite"))
        else:
            hook = DaggerHook(scorer_path, actor=actor, beta=0.0,
                              seed=model_seed * 1_000_003 + seed * 2,
                              collect=False)
            policy = HarnessPolicy(hook, max_age=80, record_requests=True)
            result = mg.run_team_episode(policy, seed, 400, hook=hook, recorder=hook,
                                         **env_kwargs("finite"))
            result["lancer_agreement"] = hook.agreement / max(hook.decisions, 1)
            result["lancer_agreement_count"] = hook.agreement
            result["actor_decisions"] = hook.decisions
        rows[key] = result
        path.write_text(json.dumps({**identity, "cells": rows}, indent=2) + "\n")
        print(f"confirm {task} seed={seed} return={result['return']:+.0f}", flush=True)
    print(f"complete {task}: {path}", flush=True)


def analyze(out):
    out = Path(out)
    plan = json.loads((out / "frozen_plan.json").read_text())
    controls = {}
    for name in CONTROLS:
        report = json.loads((out / "evaluation" / "controls" / f"{name}.json").read_text())
        if report["seeds"] != list(EVAL_SEEDS) or sorted(report["cells"]) != list(map(str, EVAL_SEEDS)):
            raise ValueError(f"incomplete/mismatched control: {name}")
        controls[name] = report["cells"]

    per_model = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for model_seed in MODEL_SEEDS:
            report = json.loads((out / "evaluation" / arm / f"s{model_seed}.json").read_text())
            task = f"{arm}:s{model_seed}"
            expected = plan["models"][task]
            if (report["seeds"] != list(EVAL_SEEDS)
                    or report["actor_sha256"] != expected["actor_sha256"]
                    or report["model_manifest_sha256"] != expected["manifest_sha256"]
                    or sorted(report["cells"]) != list(map(str, EVAL_SEEDS))):
                raise ValueError(f"incomplete/mismatched model report: {task}")
            per_model[arm][model_seed] = report["cells"]

    result = {"claim": "fresh-seed confirmation of fixed BC/DAgger models",
              "evaluation_seeds": list(EVAL_SEEDS),
              "model_training_seeds": list(MODEL_SEEDS),
              "primary_unit": plan["primary_unit"],
              "controls": {}, "arms": {}, "dagger_minus_bc": {}}
    for name in CONTROLS:
        vals = [controls[name][str(s)]["return"] for s in EVAL_SEEDS]
        result["controls"][name] = {"mean": float(np.mean(vals)),
                                     "p10": float(np.quantile(vals, .1)),
                                     "min": float(np.min(vals)), "returns": vals}
    for arm in ARMS:
        seed_means = {"lancer": [], "guard": []}
        summary_models = {}
        returns_all, agreement_all, coverage_all = [], [], []
        for model_seed in MODEL_SEEDS:
            cells = per_model[arm][model_seed]
            returns = [float(cells[str(s)]["return"]) for s in EVAL_SEEDS]
            agreement = [float(cells[str(s)].get("lancer_agreement", math.nan))
                         for s in EVAL_SEEDS]
            coverage = [float(cells[str(s)].get("coverage", math.nan))
                        for s in EVAL_SEEDS]
            returns_all.extend(returns)
            agreement_all.extend(agreement)
            coverage_all.extend(coverage)
            summary_models[str(model_seed)] = {
                "mean_return": statistics.mean(returns),
                "p10_return": float(np.quantile(returns, .1)),
                "min_return": min(returns),
                "mean_lancer_agreement": statistics.mean(agreement),
                "mean_coverage": statistics.mean(coverage),
            }
            for baseline in ("lancer", "guard"):
                deltas = [float(cells[str(s)]["return"]
                                - controls[baseline][str(s)]["return"])
                          for s in EVAL_SEEDS]
                seed_means[baseline].append(statistics.mean(deltas))
        result["arms"][arm] = {
            "mean_return": statistics.mean(returns_all),
            "p10_return": float(np.quantile(returns_all, .1)),
            "min_return": min(returns_all),
            "mean_lancer_agreement": statistics.mean(agreement_all),
            "mean_coverage": statistics.mean(coverage_all),
            "per_model_seed": summary_models,
            "paired_lancer": {"training_seed_means": seed_means["lancer"],
                              **mean_interval(seed_means["lancer"])},
            "paired_guard": {"training_seed_means": seed_means["guard"],
                             **mean_interval(seed_means["guard"])},
        }
    dagger_bc = []
    for model_seed in MODEL_SEEDS:
        deltas = [float(per_model["dagger"][model_seed][str(s)]["return"]
                        - per_model["bc"][model_seed][str(s)]["return"])
                  for s in EVAL_SEEDS]
        dagger_bc.append(statistics.mean(deltas))
    result["dagger_minus_bc"] = {
        "training_seed_means": dagger_bc,
        **mean_interval(dagger_bc),
    }
    path = out / "analysis.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"analysis": str(path),
                      "bc_minus_lancer": result["arms"]["bc"]["paired_lancer"],
                      "bc_minus_guard": result["arms"]["bc"]["paired_guard"],
                      "dagger_minus_lancer": result["arms"]["dagger"]["paired_lancer"],
                      "dagger_minus_guard": result["arms"]["dagger"]["paired_guard"],
                      "dagger_minus_bc": result["dagger_minus_bc"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.out, args.scorer)
    elif args.task:
        run_task(args.out, args.scorer, args.task)
    elif args.analyze:
        load_plan(args.out, args.scorer)
        analyze(args.out)
    else:
        parser.error("choose --freeze, --task, or --analyze")


if __name__ == "__main__":
    main()
