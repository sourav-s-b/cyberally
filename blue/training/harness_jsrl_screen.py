"""Freeze and aggregate a finite-Red synchronous-A2C guide-prefix screen."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from blue.harness.scoring import MLScorer, sha256
from blue.training.harness_rl import check_seeds, source_identity

TRAIN_SEEDS = (7706, 7707, 7708, 7709)
EVAL_SEEDS = tuple(range(9001, 9033))
RNG_SEEDS = (0, 1, 2)
GUIDE_SCHEDULE = (267, 267, 134, 134, 67, 0, 0, 0)
SOURCE_FILES = (
    "blue/harness/guided.py", "blue/harness/learning.py",
    "blue/harness/redesign.py", "blue/harness/policy.py",
    "blue/harness/scoring.py", "blue/harness/features.py",
    "blue/harness/calibration.py", "blue/harness/numerics.py",
    "blue/training/harness_redesign.py", "blue/training/harness_jsrl_screen.py",
    "blue/training/harness_rl.py", "blue/training/mappo_guide.py",
    "blue/training/residual.py", "blue/core/wrapper.py",
    "blue/policies/ordered.py", "scripts/blue_xargs_pool.sh",
    "scripts/blue_jsrl_a2c_screen.sh",
)


def plan(scorer_path):
    scorer = MLScorer(scorer_path)
    return {
        "purpose": "finite-Red development screen; exploratory, not a superiority claim",
        "source_commit": source_identity(),
        "source_files": {p: sha256(p) for p in SOURCE_FILES},
        "scorer_sha256": scorer.sha256,
        "red_agent": "finite",
        "native_reward_only": True,
        "train_environment_seeds": TRAIN_SEEDS,
        "evaluation_seeds": EVAL_SEEDS,
        "training_rng_seeds": RNG_SEEDS,
        "arms": {"a2c": (0,) * len(GUIDE_SCHEDULE),
                 "jsrl_a2c": GUIDE_SCHEDULE},
        "algorithm": "synchronous A2C; one full-rollout actor update per iteration",
        "iterations": 8,
        "episodes_per_iteration": 4,
        "episode_horizon": 400,
        "evaluation_modes": ("greedy", "sampled"),
        "evaluation_replicas_per_model_seed": 1,
        "controls": ("lancer", "guard", "sleep", "masked_random"),
        "primary_unit": "training RNG seed; evaluation episodes are paired within seed",
        "evaluation_status": "fresh block audit required before setup",
    }


def freeze(root, scorer_path):
    root = Path(root)
    target = root / "frozen_plan.json"
    current = plan(scorer_path)
    if target.exists():
        if json.loads(target.read_text()) != json.loads(json.dumps(current)):
            raise ValueError("frozen plan or source/configuration drift")
        return
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"new screen root required: {root}")
    target.write_text(json.dumps(current, indent=2) + "\n")


def verify_frozen(root, scorer_path):
    frozen = json.loads((Path(root) / "frozen_plan.json").read_text())
    if frozen != json.loads(json.dumps(plan(scorer_path))):
        raise ValueError("frozen source/config/scorer mismatch")
    check_seeds(list(frozen["evaluation_seeds"]))
    return frozen


def preflight(root, scorer_path):
    """Run short real finite-Red routes through guide and learner branches."""
    import torch
    from blue.harness.policy import HarnessPolicy
    from blue.harness.guided import GuidePrefixHook
    from blue.harness.redesign import CentralValue, RedesignHook, SetActor
    from blue.training.harness_redesign import Recorder
    from blue.training import mappo_guide as mg

    frozen = verify_frozen(root, scorer_path)
    output = Path(root) / "preflight.json"
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    torch.manual_seed(123)
    actor, critic = SetActor(64), CentralValue(64, 20.)
    args = argparse.Namespace(bonus=.25, ml_inputs="risk", temp=.5, explore=.2,
                              seed=123, gamma=.99, gae_lambda=1.)
    rows = []
    for prefix in (0, 20):
        rec = Recorder(actor, critic, MLScorer(scorer_path), args, 0)
        hook = GuidePrefixHook(rec, prefix)
        policy = HarnessPolicy(hook, max_age=80)
        result = mg.run_team_episode(policy, 7706, 40, hook=hook, recorder=rec,
                                     **{"temporal_features": ("ages", "belief"),
                                        "include_root_session": True,
                                        "red_agent": "finite"})
        row = {**result, "prefix_ticks": prefix,
               "guide_decisions": hook.guide_decisions,
               "learner_decisions": hook.learner_decisions,
               "actor_rows": len(rec.rows)}
        if row["actor_rows"] != row["learner_decisions"]:
            raise RuntimeError("guide actions leaked into the A2C actor buffer")
        if prefix == 0 and row["guide_decisions"] != 0:
            raise RuntimeError("zero-prefix preflight unexpectedly used the guide")
        if prefix and not row["guide_decisions"]:
            raise RuntimeError("guide-prefix preflight never exercised the guide")
        rows.append(row)
    report = {"source_commit": frozen["source_commit"], "red_agent": "finite",
              "episodes": rows, "passed": True}
    output.write_text(json.dumps(report, indent=2) + "\n")


def eval_controls(root, scorer_path):
    import torch
    from blue.harness.policy import HarnessPolicy
    from blue.policies.ordered import LancerValues, OrderedPolicy
    from blue.core.baselines import SleepBaseline, MaskedRandomBaseline
    from blue.training import mappo_guide as mg

    frozen = verify_frozen(root, scorer_path)
    output = Path(root) / "controls.json"
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    cells = {name: {} for name in frozen["controls"]}
    for seed in frozen["evaluation_seeds"]:
        policies = {
            "lancer": OrderedPolicy(scorer=LancerValues(fruitless_decay=.5)),
            "guard": HarnessPolicy(max_age=80),
            "sleep": SleepBaseline(),
            "masked_random": MaskedRandomBaseline(seed=seed),
        }
        for name, policy in policies.items():
            cells[name][str(seed)] = mg.run_team_episode(
                policy, seed, frozen["episode_horizon"],
                temporal_features=("ages", "belief"), include_root_session=True,
                red_agent="finite")
        print(f"controls seed={seed} lancer={cells['lancer'][str(seed)]['return']:+.0f}",
              flush=True)
    report = {"source_commit": frozen["source_commit"], "red_agent": "finite",
              "seeds": list(frozen["evaluation_seeds"]), "cells": cells}
    output.write_text(json.dumps(report, indent=2) + "\n")


def eval_model(root, scorer_path, model_dir, arm, train_rng_seed):
    import torch
    from blue.training.harness_redesign import evaluation_episode, load

    frozen = verify_frozen(root, scorer_path)
    if arm not in frozen["arms"] or train_rng_seed not in frozen["training_rng_seeds"]:
        raise ValueError("unexpected arm or training RNG seed")
    manifest_path = Path(model_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("model is incomplete")
    cfg = manifest["config"]
    expected_prefix = list(frozen["arms"][arm])
    if (cfg.get("algorithm") != "a2c" or cfg.get("red_agent") != "finite"
            or cfg.get("seed") != train_rng_seed
            or cfg.get("guide_prefix_schedule") != expected_prefix
            or cfg.get("train_seeds") != list(frozen["train_environment_seeds"])
            or cfg.get("iters") != frozen["iterations"]
            or cfg.get("eps_per_iter") != frozen["episodes_per_iteration"]
            or cfg.get("steps") != frozen["episode_horizon"]):
        raise ValueError("model config does not match frozen screen")
    output = Path(root) / "eval" / arm / f"s{train_rng_seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    cells = {"greedy": {}, "sampled": {}}
    for seed in frozen["evaluation_seeds"]:
        for mode in ("greedy", "sample"):
            pol, _ = load(model_dir, scorer_path,
                          train_rng_seed * 1_000_003 + seed * 2 + (mode == "sample"),
                          "greedy-diagnostic" if mode == "greedy" else "sample")
            result = evaluation_episode(pol, seed, frozen["episode_horizon"], "finite")
            result["policy_seed"] = train_rng_seed * 1_000_003 + seed * 2 + (mode == "sample")
            result["guard"] = pol.guard_stats()
            cells["greedy" if mode == "greedy" else "sampled"][str(seed)] = result
        print(f"eval {arm} s{train_rng_seed} seed={seed} "
              f"greedy={cells['greedy'][str(seed)]['return']:+.0f} "
              f"sampled={cells['sampled'][str(seed)]['return']:+.0f}", flush=True)
    report = {"arm": arm, "training_rng_seed": train_rng_seed,
              "model_dir": str(Path(model_dir).resolve()),
              "model_manifest_sha256": sha256(manifest_path),
              "actor_sha256": manifest["actor_sha256"], "red_agent": "finite",
              "evaluation_seeds": list(frozen["evaluation_seeds"]), "cells": cells}
    output.write_text(json.dumps(report, indent=2) + "\n")


def _mean_interval(values):
    from scipy.stats import t
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    if len(x) < 2:
        return {"mean": mean, "low95": None, "high95": None}
    half = float(t.ppf(.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    return {"mean": mean, "low95": mean - half, "high95": mean + half}


def analyze(root, scorer_path):
    frozen = verify_frozen(root, scorer_path)
    base = json.loads((Path(root) / "controls.json").read_text())
    expected = list(map(str, frozen["evaluation_seeds"]))
    if base.get("seeds") != frozen["evaluation_seeds"]:
        raise ValueError("control seed list drift")
    for name in frozen["controls"]:
        if sorted(base["cells"][name]) != expected:
            raise ValueError(f"incomplete control: {name}")
    reports = {}
    for arm in frozen["arms"]:
        reports[arm] = {}
        for rng in frozen["training_rng_seeds"]:
            path = Path(root) / "eval" / arm / f"s{rng}.json"
            report = json.loads(path.read_text())
            if (report.get("arm") != arm or report.get("training_rng_seed") != rng
                    or report.get("evaluation_seeds") != frozen["evaluation_seeds"]
                    or report.get("red_agent") != "finite"):
                raise ValueError(f"evaluation identity/seed mismatch: {path}")
            reports[arm][rng] = report
    summary = {"claim": "exploratory finite-Red screen; no confirmatory claim",
               "paired_unit": "mean paired difference across evaluation episodes within each training RNG seed",
               "training_seed_intervals": {}, "tail_returns": {},
               "controls": {name: [base["cells"][name][s]["return"] for s in expected]
                            for name in frozen["controls"]}}
    for arm in frozen["arms"]:
        summary["training_seed_intervals"][arm] = {}
        summary["tail_returns"][arm] = {}
        for mode in ("greedy", "sampled"):
            per_rng = {}
            all_returns = []
            for rng in frozen["training_rng_seeds"]:
                report = reports[arm][rng]
                mode_key = "greedy" if mode == "greedy" else "sampled"
                if sorted(report["cells"][mode_key]) != expected:
                    raise ValueError(f"evaluation episode missing: {arm}/s{rng}/{mode}")
                actor = np.asarray([report["cells"][mode_key][s]["return"] for s in expected], float)
                lancer = np.asarray([base["cells"]["lancer"][s]["return"] for s in expected], float)
                guard = np.asarray([base["cells"]["guard"][s]["return"] for s in expected], float)
                per_rng[rng] = {"lancer": float(np.mean(actor - lancer)),
                                "guard": float(np.mean(actor - guard))}
                all_returns.extend(actor.tolist())
            summary["training_seed_intervals"][arm][mode] = {
                control: {"by_training_seed": {str(k): v[control] for k, v in per_rng.items()},
                          **_mean_interval([v[control] for v in per_rng.values()])}
                for control in ("lancer", "guard")}
            summary["tail_returns"][arm][mode] = {
                "n_episodes": len(all_returns), "minimum": float(np.min(all_returns)),
                "p10": float(np.percentile(all_returns, 10)),
                "mean": float(np.mean(all_returns))}
    summary["training_seed_intervals"]["jsrl_minus_plain_a2c"] = {}
    for mode in ("greedy", "sampled"):
        per_rng = {}
        for rng in frozen["training_rng_seeds"]:
            a = reports["a2c"][rng]["cells"]["greedy" if mode == "greedy" else "sampled"]
            b = reports["jsrl_a2c"][rng]["cells"]["greedy" if mode == "greedy" else "sampled"]
            per_rng[rng] = float(np.mean([b[s]["return"] - a[s]["return"] for s in expected]))
        summary["training_seed_intervals"]["jsrl_minus_plain_a2c"][mode] = {
            "by_training_seed": {str(k): v for k, v in per_rng.items()},
            **_mean_interval(list(per_rng.values()))}
    summary["recommend_confirmation"] = any(
        summary["training_seed_intervals"][arm]["greedy"][control]["low95"] is not None
        and summary["training_seed_intervals"][arm]["greedy"][control]["low95"] > 0
        for arm in frozen["arms"] for control in ("lancer", "guard"))
    summary["source_commit"] = frozen["source_commit"]
    summary["evaluation_seeds"] = list(frozen["evaluation_seeds"])
    out = Path(root) / "analysis.json"
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("freeze", "preflight", "controls", "eval-model", "analyze"))
    p.add_argument("--root", required=True)
    p.add_argument("--scorer", required=True)
    p.add_argument("--model-dir")
    p.add_argument("--arm", choices=("a2c", "jsrl_a2c"))
    p.add_argument("--train-rng-seed", type=int)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.mode == "freeze": freeze(args.root, args.scorer)
    elif args.mode == "preflight": preflight(args.root, args.scorer)
    elif args.mode == "controls": eval_controls(args.root, args.scorer)
    elif args.mode == "eval-model": eval_model(args.root, args.scorer, args.model_dir,
                                                 args.arm, args.train_rng_seed)
    else: analyze(args.root, args.scorer)
