"""Blue development comparison of a saved MAPPO policy and existing baselines.

Native return only: this is not the full Evaluation-owned incident evaluator.
Development seeds must be separate from training and the untouched final suite.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

from blue_mappo import ROOT, main as evaluate_mappo, sha256
from blue_baselines import (
    MaskedRandomBaseline, RoundRobinBaseline, SleepBaseline, run_episode,
)


def paired_summary(policy_episodes, baseline_episodes):
    policy = {episode["seed"]: episode for episode in policy_episodes}
    baseline = {episode["seed"]: episode for episode in baseline_episodes}
    if (not policy or len(policy) != len(policy_episodes)
            or len(baseline) != len(baseline_episodes) or policy.keys() != baseline.keys()):
        raise ValueError("Comparison requires identical unique nonempty seed suites")
    differences = []
    for seed, episode in policy.items():
        other = baseline[seed]
        if episode["joint_steps"] != other["joint_steps"]:
            raise ValueError("Comparison requires identical episode lengths")
        differences.append(episode["native_return"] - other["native_return"])
    return {
        "episodes": len(differences),
        "mappo_mean_return": statistics.mean(e["native_return"] for e in policy.values()),
        "baseline_mean_return": statistics.mean(e["native_return"] for e in baseline.values()),
        "mean_paired_return_difference": statistics.mean(differences),
        "paired_return_differences": differences,
        "mappo_wins": sum(d > 0 for d in differences),
        "ties": sum(d == 0 for d in differences),
        "interpretation": "Higher native return is better; positive difference favors MAPPO",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epymarl-root", type=Path, default=ROOT / ".tools/epymarl")
    options = parser.parse_args(argv)
    manifest = json.loads(options.manifest.read_text(encoding="utf-8"))
    if (len(set(options.eval_seeds)) != len(options.eval_seeds)
            or set(options.eval_seeds) & set(manifest["train_seeds"])):
        raise ValueError("Evaluation seeds must be unique and separate from training")
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = evaluate_mappo([
        "--evaluate-manifest", str(options.manifest),
        "--output", str(output / "mappo"),
        "--epymarl-root", str(options.epymarl_root),
        "--eval-seeds", *map(str, options.eval_seeds),
    ])
    episodes = {"mappo": result["evaluation"]}
    steps = manifest["config"]["env_args"]["steps"]
    factories = {"sleep": lambda seed: SleepBaseline(),
                 "masked-random": lambda seed: MaskedRandomBaseline(seed=seed),
                 "round-robin": lambda seed: RoundRobinBaseline()}
    for name, factory in factories.items():
        episodes[name] = []
        for seed in options.eval_seeds:
            started = time.perf_counter()
            episode = run_episode(factory(seed), seed=seed, steps=steps,
                                  mask_mode="validity", per_agent_bounds=False)
            record = {"seed": seed, "joint_steps": episode["steps"],
                      "native_return": episode["cumulative_return"],
                      "policy_seed": seed if name == "masked-random" else None,
                      "seconds": time.perf_counter() - started}
            episodes[name].append(record)
            # Keep progress available if a later episode fails.
            with (output / "baseline-episodes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"policy": name, **record}) + "\n")
            print(json.dumps({"policy": name, **record}), flush=True)
    report = {
        "schema_version": "blue-native-return-comparison-v1",
        "policy_manifest": str(options.manifest.resolve()),
        "policy_manifest_sha256": sha256(options.manifest),
        "comparison_source_sha256": sha256(Path(__file__)),
        "baseline_source_sha256": sha256(ROOT / "cage-challenge-4/blue_baselines.py"),
        "evaluation_seeds": options.eval_seeds, "episodes": episodes,
        "comparisons": {name: paired_summary(episodes["mappo"], episodes[name])
                        for name in factories},
        "limitations": ["Development suite; not an untouched final benchmark",
                        "Native return only; incident/service metrics remain Evaluation work",
                        "One training seed and one stochastic policy sample per evaluation seed"],
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["comparisons"], indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
