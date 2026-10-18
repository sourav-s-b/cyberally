"""Freeze, preflight, and analyze the bounded finite-Red anchor pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from blue.harness.policy import HarnessActorHook, HarnessPolicy
from blue.harness.scoring import MLScorer, sha256
from blue.training.harness_anchor import AnchoredActorHook
from blue.training.harness_eval import episode
from blue.training.harness_rl import build_actor

TRAIN_SEEDS = (7706, 7707, 7708, 7709)
EVAL_SEEDS = tuple(range(7665, 7689))
ARMS = ("current", "anchored")
SOURCE_FILES = [
    "blue/training/harness_anchor.py", "blue/training/harness_anchor_pilot.py",
    "blue/training/harness_rl.py", "blue/training/harness_eval.py",
    "blue/training/mappo_guide.py", "blue/training/residual.py",
    "blue/harness/policy.py", "blue/harness/scoring.py",
    "blue/harness/features.py", "blue/policies/ordered.py",
    "blue/core/wrapper.py", "scripts/blue_anchor_pilot.sh",
]


def setup(root):
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "frozen_plan.json"
    files = {name: sha256(name) for name in SOURCE_FILES}
    plan = {"purpose": "development pilot; no RL superiority claim",
            "train_seeds": TRAIN_SEEDS, "evaluation_seeds": EVAL_SEEDS,
            "training_rng_seeds": (0, 1, 2), "iterations": 8,
            "episodes_per_iteration": 4, "horizon": 400,
            "policy_versions": ("current-residual-v1", "anchored-categorical-v1"),
            "reference_mass": 0.9, "reference_temp": 0.5,
            "reference_kl_coef": 0.05, "max_update_kl": 0.02,
            "sampling_seed_base": 900000,
            "source_files": files}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        snapshot_sha = existing.pop("snapshot_sha256", None)
        if existing != json.loads(json.dumps(plan)):
            raise ValueError("frozen pilot source or configuration drift")
        archive = root / "source_snapshot.zip"
        if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != snapshot_sha:
            raise ValueError("frozen source snapshot is missing or modified")
        return
    if any(root.iterdir()):
        raise FileExistsError("new pilot root required")
    archive = root / "source_snapshot.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in SOURCE_FILES:
            zf.write(name)
    plan["snapshot_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(plan, indent=2) + "\n")


def preflight(root, scorer_path):
    import torch as th
    if not (root / "frozen_plan.json").exists():
        raise FileNotFoundError("freeze pilot before preflight")
    out = root / "preflight"
    if out.exists():
        raise FileExistsError("preflight output already exists")
    out.mkdir()
    cells = {}
    for seed in TRAIN_SEEDS:
        for mode in ("guard", "current_greedy", "anchored_greedy",
                     "current_sampled", "anchored_sampled"):
            th.manual_seed(0)
            if mode == "guard":
                policy = HarnessPolicy(max_age=80)
            else:
                actor = build_actor(64)
                if any(bool(th.count_nonzero(p)) for p in actor.head.parameters()):
                    raise RuntimeError("actor output head is not zero at initialization")
                scorer = MLScorer(scorer_path)
                sample = mode.endswith("sampled")
                gen = (th.Generator().manual_seed(900000 + seed * 37)
                       if sample else None)
                hook = (AnchoredActorHook(actor, scorer, mass=0.9,
                                          temperature=0.5, sample=sample,
                                          generator=gen)
                        if mode.startswith("anchored") else
                        HarnessActorHook(actor, scorer, bonus=0.25,
                                         sample=sample, generator=gen,
                                         sample_temp=0.5))
                policy = HarnessPolicy(hook, max_age=80)
            result = episode(policy, seed, 400, out / f"{mode}_{seed}.jsonl",
                             red_agent="finite")
            cells.setdefault(mode, {})[str(seed)] = result
            print(f"preflight mode={mode} seed={seed} return={result['return']:+.0f}",
                  flush=True)
    for seed in TRAIN_SEEDS:
        key = str(seed)
        guard = Path(cells["guard"][key]["blue_trace"])
        expected = [json.loads(x)["agents"] for x in guard.read_text().splitlines()]
        expected = [{a: (v["action"], v["host"]) for a, v in row.items()}
                    for row in expected]
        for mode in ("current_greedy", "anchored_greedy"):
            trace = Path(cells[mode][key]["blue_trace"])
            actual = [json.loads(x)["agents"] for x in trace.read_text().splitlines()]
            actual = [{a: (v["action"], v["host"]) for a, v in row.items()}
                      for row in actual]
            if actual != expected:
                raise RuntimeError(f"zero-head {mode} differs from guard on {seed}")
    (out / "report.json").write_text(json.dumps({"cells": cells,
                                                  "greedy_guard_action_parity": True},
                                                 indent=2) + "\n")


def _cells(path, arm):
    report = json.loads(Path(path).read_text())
    cells = report.get("cells", {}).get(arm, {})
    if set(cells) != set(map(str, EVAL_SEEDS)):
        raise ValueError(f"evaluation seed mismatch: {path}")
    return report, {int(s): float(row["return"]) for s, row in cells.items()}


def analyze(root):
    from scipy.stats import t
    frozen = json.loads((root / "frozen_plan.json").read_text())
    if frozen["evaluation_seeds"] != list(EVAL_SEEDS):
        raise ValueError("pilot evaluation seeds drifted")
    for name, digest in frozen["source_files"].items():
        if sha256(name) != digest:
            raise ValueError(f"pilot source drift: {name}")
    _, lancer = _cells(root / "eval/lancer/report.json", "lancer")
    _, guard = _cells(root / "eval/guard/report.json", "guard")
    models = {}
    for arm in ARMS:
        models[arm] = {}
        for seed in range(3):
            path = root / f"train/{arm}/s{seed}"
            manifest = json.loads((path / "manifest.json").read_text())
            if manifest["status"] != "complete" or len(manifest["history"]) != 8:
                raise ValueError(f"incomplete pilot model: {path}")
            if manifest["config"]["policy_version"] != frozen["policy_versions"][ARMS.index(arm)]:
                raise ValueError("policy version drift")
            if manifest["actor_sha256"] != sha256(path / "actor.th"):
                raise ValueError("modified actor")
            for name, digest in manifest["source_files"].items():
                if sha256(name) != digest:
                    raise ValueError(f"training source drift: {name}")
            modes = {}
            for mode in ("greedy", "sampled"):
                report, returns = _cells(root / f"eval/{arm}_{mode}_s{seed}/report.json", "actor")
                if (report["config"]["actor_mode"] != mode or
                    any(cell["artifact_sha256"] != manifest["actor_sha256"]
                        for cell in report["cells"]["actor"].values())):
                    raise ValueError("evaluation model or mode drift")
                modes[mode] = returns
            models[arm][seed] = modes

    def summary(values):
        x = np.array(values, dtype=float)
        radius = float(t.ppf(.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
        return {"per_training_seed": x.tolist(), "mean": float(x.mean()),
                "exploratory_ci95": [float(x.mean() - radius), float(x.mean() + radius)]}

    def means(arm, mode, baseline):
        return [float(np.mean([models[arm][s][mode][seed] - baseline[seed]
                               for seed in EVAL_SEEDS])) for s in range(3)]

    anchored_vs_lancer = means("anchored", "greedy", lancer)
    anchored_vs_guard = means("anchored", "greedy", guard)
    anchored_vs_current = [float(np.mean([
        models["anchored"][s]["greedy"][seed] - models["current"][s]["greedy"][seed]
        for seed in EVAL_SEEDS])) for s in range(3)]
    sorted_lancer = sorted(lancer.values())[:3]
    actor_worst_three = [float(np.mean(sorted(models["anchored"][s]["greedy"].values())[:3]))
                         for s in range(3)]
    tail_ok = float(np.mean(actor_worst_three)) >= float(np.mean(sorted_lancer))
    gate = (float(np.mean(anchored_vs_lancer)) > 0 and
            float(np.mean(anchored_vs_guard)) > 0 and
            sum(x > 0 for x in anchored_vs_current) >= 2 and tail_ok)
    result = {"claim": "bounded development pilot; no held-out RL win",
              "evaluation_seeds": EVAL_SEEDS,
              "anchored_minus_lancer_greedy": summary(anchored_vs_lancer),
              "anchored_minus_guard_greedy": summary(anchored_vs_guard),
              "anchored_minus_current_greedy": summary(anchored_vs_current),
              "anchored_worst_three_return": actor_worst_three,
              "lancer_worst_three_return": sorted_lancer,
              "tail_gate_pass": tail_ok,
              "recommend_confirmation": gate,
              "all_episode_returns": models,
              "lancer_returns": lancer, "guard_returns": guard}
    (root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("all_episode_returns", "lancer_returns", "guard_returns")},
                     indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("setup", "preflight", "analyze"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--scorer", default="blue/results/gpt_harness/scorer.pkl")
    args = parser.parse_args()
    root = Path(args.root)
    if args.command == "setup":
        setup(root)
    elif args.command == "preflight":
        preflight(root, args.scorer)
    else:
        analyze(root)


if __name__ == "__main__":
    main()
