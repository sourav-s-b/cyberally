"""One Kaggle CPU shard for the frozen Blue architecture study.

prepare.py substitutes the training RNG seed before uploading this script.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SHARD_SEED = __SHARD_SEED__
SHARD_ARMS = __SHARD_ARMS__

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")

studies = list(Path("/kaggle/input").rglob("study.json"))
if len(studies) != 1:
    raise RuntimeError("attach exactly the frozen architecture dataset")
data = studies[0].parent
cfg = json.loads(studies[0].read_text())
requirements = cfg["runtime_requirements"]
for name, digest in cfg["input_hashes"].items():
    if hashlib.sha256((data / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError("frozen input hash mismatch: " + name)
if SHARD_SEED not in cfg["training_rng_seeds"]:
    raise RuntimeError("unknown training RNG shard")
if not SHARD_ARMS or any(arm not in cfg["arms"] for arm in SHARD_ARMS):
    raise RuntimeError("unknown architecture shard")

root = Path("/tmp/blue-architecture-source")
subprocess.run(["git", "clone", "--no-checkout", cfg["source_repository"],
                str(root)], check=True)
subprocess.run(["git", "-C", str(root), "checkout", "--detach",
                cfg["source_commit"]], check=True)
actual = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
if actual != cfg["source_commit"]:
    raise RuntimeError("source commit mismatch")
lock = root / requirements
lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest()
if lock_hash != cfg["runtime_requirements_sha256"]:
    raise RuntimeError("runtime lock hash mismatch")

subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "uv==0.12.15"], check=True)
subprocess.run([sys.executable, "-m", "uv", "venv", "--python",
                cfg["runtime_versions"]["python"], "/tmp/blue-architecture-env"],
               check=True)
python = "/tmp/blue-architecture-env/bin/python"
subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--python",
                python, "--torch-backend", "cpu", "-r",
                str(lock)], check=True)
os.chdir(root)
sys.path.insert(0, str(root))

subprocess.run([python, "-m", "blue.training.architecture_preflight",
                "--input", str(data)], check=True)

output = Path("/kaggle/working/architecture_study")
output.mkdir(parents=True, exist_ok=True)
(output / "study.json").write_text(json.dumps(cfg, indent=2) + "\n")
cohort = cfg["training_rng_seeds"]
print(f"ARCHITECTURE SHARD START rng={SHARD_SEED} "
      f"cohort={cohort} cohort_sha256={cfg['cohort_sha256']}", flush=True)
command = [
    python, "-m", "blue.training.architecture_study",
    "--scorer", str(data / "scorer.pkl"),
    "--out", str(output),
    "--iters", str(cfg["iterations"]),
    "--eps-per-iter", str(cfg["eps_per_iter"]),
    "--steps", str(cfg["steps"]),
    "--training-rng", str(SHARD_SEED),
    "--arms", *SHARD_ARMS,
    "--cohort-seeds", *map(str, cohort),
    "--cohort-sha256", cfg["cohort_sha256"],
    "--train-seeds", *map(str, cfg["train_episode_seeds"]),
    "--eval-seeds", *map(str, cfg["evaluation_seeds"]),
    "--policy-replicas", str(cfg["sample_eval_replicas"]),
    "--red-agent", cfg["red_agent"],
]
subprocess.run(command, check=True)
print(f"ARCHITECTURE SHARD COMPLETE rng={SHARD_SEED}; "
      "all checkpoints and episode cells are in /kaggle/working/architecture_study",
      flush=True)
