"""Freeze private inputs and five one-training-seed Kaggle kernel packages."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARMS = ["mlp64", "mlp128", "mlp256", "mlp512", "attention128",
        "attention256", "gru128", "graph128", "graph_self128"]
ARM_GROUPS = {
    "a": ["mlp64", "mlp128", "mlp256"],
    "b": ["mlp512", "attention128", "attention256"],
    "c": ["gru128", "graph128", "graph_self128"],
}
# AST/range and manifest audit found 7730-7800 free at commit 82632bc.
# These 71 development episode seeds are reserved by this architecture study.
TRAIN_EPISODE_SEEDS = list(range(7730, 7801))
EVAL_SEEDS = list(range(8221, 8253))
TRAINING_RNG_SEEDS = [0, 1, 2, 3, 4]
RESERVED_FINAL = [7809, 8200]
DATASET_ID = "souravsreekumar02/blue-architecture-longrun-20261012"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True,
                        help="frozen directory containing scorer.pkl and golden.npz")
    parser.add_argument("--upload-root", type=Path, required=True,
                        help="local staging directory, preferably under /tmp")
    args = parser.parse_args()

    flattened = [arm for group in ARM_GROUPS.values() for arm in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(ARMS):
        raise RuntimeError("architecture groups must partition the frozen arm list")

    for name in ("scorer.pkl", "golden.npz"):
        if not (args.artifact_dir / name).is_file():
            raise FileNotFoundError(args.artifact_dir / name)

    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if status:
        raise RuntimeError("commit source first; freeze must pin a clean tree")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    requirements = ROOT / "ops/kaggle/harness/requirements-legacy-lock.txt"
    study = {
        "experiment": "blue-architecture-longrun-finite-v1",
        "purpose": "development comparison of policy architectures and training budget",
        "source_repository": "https://github.com/sourav-s-b/cyberally.git",
        "source_commit": source_commit,
        "training_rng_seeds": TRAINING_RNG_SEEDS,
        "train_episode_seeds": TRAIN_EPISODE_SEEDS,
        "training_seed_allocation_audit": {
            "method": "AST-expand Python range calls and inspect tracked plus ignored structured Python/JSON seed lists; omit git-SHA substrings and nested virtualenv packages",
            "checked_commit": "243a6fd97dc29d59c7bfc677209d363412f6cc10",
            "also_checked": ["runs/blue-architecture-tests including ignored outputs", "worktrees/gpt-blue source and manifests on disk"],
            "checked_range": "7730-7800",
            "result": "no recorded consumption; outside protected 7809-8200 and finite-Red 8301-8400 blocks",
        },
        "evaluation_seeds": EVAL_SEEDS,
        "evaluation_note": (
            "These episode IDs were already used for development under other "
            "attacker/model configurations; this is not a held-out or final claim."
        ),
        "reserved_final_seeds": RESERVED_FINAL,
        "arms": ARMS,
        "iterations": 32,
        "eps_per_iter": 4,
        "steps": 400,
        "sample_eval_replicas": 1,
        "execution_modes": ["greedy-diagnostic", "sample"],
        "red_agent": "finite",
        "max_age": 80.0,
        "gamma": 0.99,
        "bonus": 0.25,
        "temperature": 0.5,
        "explore": 0.2,
        "actor_lr": 0.0003,
        "critic_lr": 0.0003,
        "critic_hidden": 64,
        "value_scale": 20.0,
        "epochs": 4,
        "target_kl": 0.02,
        "ppo_clip": 0.2,
        "max_grad_norm": 1.0,
        "entropy_coef": 0.0,
        "env_kw": {
            "temporal_features": ["ages", "belief"],
            "include_root_session": True,
            "red_agent": "finite",
        },
        "runtime_requirements": "ops/kaggle/harness/requirements-legacy-lock.txt",
        "runtime_requirements_sha256": sha(requirements),
        "runtime_versions": {
            "python": "3.11.16",
            "torch_num_threads": 1,
            "numpy": "1.26.4",
            "scipy": "1.12.0",
            "torch": "2.2.0+cpu",
            "scikit-learn": "1.7.2",
            "gym": "0.26.2",
            "gymnasium": "0.28.1",
        },
        "input_hashes": {
            "scorer.pkl": sha(args.artifact_dir / "scorer.pkl"),
            "golden.npz": sha(args.artifact_dir / "golden.npz"),
        },
        "preflight": {
            "seed": 8233,
            "return": -137.0,
            "ticks": 399,
            "completed_analysis_coverage": 1.0,
            "completed_remediations": 78,
            "analysis_confirmations": 78,
        },
        "success_criteria": {
            "primary_metric": "native Blue team return, higher is better",
            "primary_execution": "greedy-diagnostic",
            "controls": ["lancer", "guard"],
            "replication_unit": "training RNG seed, n=5, df=4",
            "evaluation_unit": "shared episode seed, averaged within each trained model",
            "decision": (
                "Development/model-selection evidence only. No architecture is "
                "declared superior from this reused evaluation cohort; retain all "
                "arms and nominate a candidate for a separately authorized final suite."
            ),
            "report_every_arm_and_tail": True,
        },
    }
    study["cohort_sha256"] = hashlib.sha256(json.dumps(
        study, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    upload = args.upload_root.resolve()
    data_dir = upload / "dataset"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "study.json").write_text(json.dumps(study, indent=2) + "\n")
    for name in ("scorer.pkl", "golden.npz"):
        shutil.copy2(args.artifact_dir / name, data_dir / name)
    metadata = {
        "title": "Blue architecture long run 2026-10-12",
        "id": DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
        "keywords": ["cyber-range", "reinforcement-learning", "development"],
    }
    (data_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    template = (ROOT / "ops/kaggle/architecture/runner_template.py").read_text()
    kernel_root = upload / "kernels"
    for seed in TRAINING_RNG_SEEDS:
        for group, arms in ARM_GROUPS.items():
            folder = kernel_root / f"seed-{seed}-{group}"
            folder.mkdir(parents=True, exist_ok=True)
            script = template.replace("__SHARD_SEED__", str(seed))
            script = script.replace("__SHARD_ARMS__", repr(arms))
            (folder / "run.py").write_text(script)
            kernel_id = f"souravsreekumar02/blue-architecture-longrun-s{seed}-{group}"
            kernel = {
                "id": kernel_id,
                "title": f"Blue Architecture Longrun S{seed} {group.upper()}",
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [DATASET_ID],
                "kernel_sources": [],
                "competition_sources": [],
                "model_sources": [],
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
            }
            (folder / "kernel-metadata.json").write_text(json.dumps(kernel, indent=2) + "\n")

    print(json.dumps({
        "source_commit": source_commit,
        "cohort_sha256": study["cohort_sha256"],
        "dataset_dir": str(data_dir),
        "kernel_dirs": [str(kernel_root / f"seed-{s}-{g}")
                        for s in TRAINING_RNG_SEEDS for g in ARM_GROUPS],
        "training_episodes_per_model": study["iterations"] * study["eps_per_iter"],
        "models": len(ARMS) * len(TRAINING_RNG_SEEDS),
        "training_episode_total": len(ARMS) * len(TRAINING_RNG_SEEDS) * study["iterations"] * study["eps_per_iter"],
        "full_episode_budget_including_evaluation_and_repeated_controls": (
            len(ARMS) * len(TRAINING_RNG_SEEDS) * study["iterations"] * study["eps_per_iter"]
            + len(ARMS) * len(TRAINING_RNG_SEEDS) * len(EVAL_SEEDS)
              * (1 + study["sample_eval_replicas"])
            + len(TRAINING_RNG_SEEDS) * len(ARM_GROUPS) * len(EVAL_SEEDS) * 6
        ),
        "kernel_count": len(TRAINING_RNG_SEEDS) * len(ARM_GROUPS),
        "eval_cells_per_model": len(EVAL_SEEDS) * (1 + study["sample_eval_replicas"]),
        "final_seed_block_touched": False,
    }, indent=2))


if __name__ == "__main__":
    main()
