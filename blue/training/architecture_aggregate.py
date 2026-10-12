"""Combine architecture-study shards at the independent training-seed level."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from blue.training.architecture_study import ARMS


def _cell_seed_means(cells: dict) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for key, cell in cells.items():
        seed = key.split("/")[0]
        grouped.setdefault(seed, []).append(float(cell["return"]))
    return {seed: float(np.mean(values)) for seed, values in grouped.items()}


def _interval(values: list[float]) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 3:
        return {"mean": mean, "training_seed_differences": values.tolist(),
                "n_training_seeds": n, "t95": None}
    se = float(values.std(ddof=1) / np.sqrt(n))
    half = float(t.ppf(.975, n - 1) * se)
    return {"mean": mean, "training_seed_differences": values.tolist(),
            "n_training_seeds": n, "df": n - 1,
            "t95": [mean - half, mean + half]}


def aggregate(shard_dirs: list[Path], out: Path) -> dict:
    shards = []
    for folder in shard_dirs:
        plan = json.loads((folder / "plan.json").read_text())
        results = json.loads((folder / "results.json").read_text())
        active = plan["config"]["training_rng"]
        if len(active) != 1:
            raise ValueError(f"{folder}: expected one training seed per shard")
        shards.append((active[0], plan, results))

    shards.sort(key=lambda row: (row[0], tuple(row[1]["config"]["arms"])))
    if not shards:
        raise ValueError("no shards supplied")
    cohort = shards[0][1]["config"]["training_rng_cohort"]
    seeds = sorted(cohort)
    actual_seeds = sorted({row[0] for row in shards})
    if actual_seeds != seeds:
        raise ValueError(f"incomplete training RNG cohort: {actual_seeds} != {seeds}")
    first_plan = shards[0][1]
    arm_sets = {rng: set() for rng in seeds}
    for rng, plan, results in shards:
        a = dict(plan)
        b = dict(first_plan)
        a_cfg = dict(a["config"]); b_cfg = dict(b["config"])
        a_cfg["training_rng"] = b_cfg["training_rng"]
        a_cfg["arms"] = b_cfg["arms"]
        a["config"] = a_cfg; b["config"] = b_cfg
        if a != b:
            raise ValueError(f"frozen config/source/runtime drift in shard {rng}")
        active_arms = set(plan["config"]["arms"])
        if not active_arms or arm_sets[rng] & active_arms:
            raise ValueError(f"empty or overlapping architecture shard for training seed {rng}")
        arm_sets[rng].update(active_arms)
        if set(results["models"]) != {f"{arm}/s{rng}" for arm in active_arms}:
            raise ValueError(f"incomplete model outputs in shard {rng}: {sorted(active_arms)}")
    for rng, found in arm_sets.items():
        if found != set(ARMS):
            raise ValueError(f"incomplete architecture set for training seed {rng}: {sorted(found)}")

    eval_seeds = [str(s) for s in first_plan["config"]["eval_seeds"]]
    arms = list(ARMS)
    if any(found != set(arms) for found in arm_sets.values()):
        raise ValueError("architecture arm set/order differs from source")
    baseline_tables = {}
    all_models = {}
    for rng, _, results in shards:
        current = results["baselines"]
        if set(current) != {f"{name}/{s}" for name in
                            ("lancer", "guard", "sleep", "random")
                            for s in eval_seeds}:
            raise ValueError(f"missing baseline cells in training seed {rng}")
        for name in ("lancer", "guard", "sleep", "random"):
            for seed in eval_seeds:
                key = f"{name}/{seed}"
                value = current[key]["return"]
                if key in baseline_tables and baseline_tables[key] != value:
                    raise ValueError(f"baseline changed across shards at {key}")
                baseline_tables[key] = value
        for key, model in results["models"].items():
            arm = key.split("/")[0]
            for execution in ("greedy-diagnostic", "sample"):
                means = _cell_seed_means(model["cells"][execution])
                if set(means) != set(eval_seeds):
                    raise ValueError(f"missing {execution} eval cells: {arm}/s{rng}")
            if f"{arm}/s{rng}" in all_models:
                raise ValueError(f"duplicate architecture result {arm}/s{rng}")
            all_models[f"{arm}/s{rng}"] = model

    contrasts = {}
    for execution in ("greedy-diagnostic", "sample"):
        for arm in arms:
            for control in ("lancer", "guard", "untrained", "mlp64"):
                per_training_seed = []
                for rng in seeds:
                    model = all_models[f"{arm}/s{rng}"]
                    arm_means = _cell_seed_means(model["cells"][execution])
                    deltas = []
                    for seed in eval_seeds:
                        if control == "untrained":
                            ref = model["untrained"][execution][f"{seed}/0"]["return"]
                        elif control == "mlp64":
                            ref = _cell_seed_means(
                                all_models[f"mlp64/s{rng}"]["cells"][execution]
                            )[seed]
                        else:
                            ref = baseline_tables[f"{control}/{seed}"]
                        deltas.append(arm_means[seed] - ref)
                    per_training_seed.append(float(np.mean(deltas)))
                contrasts[f"{execution}:{arm}-minus-{control}"] = _interval(per_training_seed)

        graph_deltas = []
        for rng in seeds:
            visible = _cell_seed_means(all_models[f"graph128/s{rng}"]["cells"][execution])
            self_only = _cell_seed_means(all_models[f"graph_self128/s{rng}"]["cells"][execution])
            graph_deltas.append(float(np.mean([visible[s] - self_only[s] for s in eval_seeds])))
        contrasts[f"{execution}:graph128-minus-graph_self128"] = _interval(graph_deltas)

    report = {
        "claim": first_plan["claim"],
        "cohort_sha256": first_plan["config"].get("cohort_sha256"),
        "source_commit": first_plan["source_commit"],
        "runtime": first_plan["runtime"],
        "scorer_sha256": first_plan["scorer_sha256"],
        "training_rng_seeds": seeds,
        "training_episode_seeds": first_plan["config"]["train_seeds"],
        "evaluation_seeds": [int(s) for s in eval_seeds],
        "evaluation_note": "Reused development episodes; inference is conditional on this fixed cohort. No held-out claim.",
        "primary_execution": "greedy-diagnostic",
        "secondary_execution": "sample",
        "sampling_unit": "independent training RNG seed; evaluation episodes are shared blocks and sampled replicas are nested",
        "means": {},
        "contrasts": contrasts,
    }
    for execution in ("greedy-diagnostic", "sample"):
        report["means"][execution] = {}
        for arm in arms:
            per_seed = []
            for rng in seeds:
                cell_means = _cell_seed_means(all_models[f"{arm}/s{rng}"]["cells"][execution])
                per_seed.append(float(np.mean([cell_means[s] for s in eval_seeds])))
            report["means"][execution][arm] = _interval(per_seed)
        for name in ("lancer", "guard", "sleep", "random"):
            report["means"][execution][name] = float(np.mean(
                [baseline_tables[f"{name}/{s}"] for s in eval_seeds]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.shards, args.out), indent=2))
