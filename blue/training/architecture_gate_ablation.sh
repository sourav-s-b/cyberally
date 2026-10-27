#!/usr/bin/env bash
# Restartable, paired SetActor / fixed-blend / learned-gate comparison.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
train_python="${CYBERALLY_TRAIN_PYTHON:-/home/sourav/projects/cyberally/.venv-train/bin/python}"
base="blue/results/architecture_gate_ablation_v1"
eval_seeds=(8417 8418 8419 8420 8421 8422 8423 8424 \
            8425 8426 8427 8428 8429 8430 8431 8432)
mkdir -p "$base/evaluations" "$base/logs"

run_arch() {
  local arch="$1"
  for rng_seed in 0 1; do
    local out="$base/${arch}_s${rng_seed}"
    local report="$base/evaluations/${arch}_s${rng_seed}.json"
    local train_seeds
    if [[ "$rng_seed" == 0 ]]; then
      train_seeds=(8465 8466 8467 8468 8469 8470 8471 8472)
    else
      train_seeds=(8473 8474 8475 8476 8477 8478 8479 8480)
    fi
    mkdir -p "$out"
    if [[ -f "$out/manifest.json" ]]; then
      local status
      status="$("$train_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$out/manifest.json")"
      if [[ "$status" != complete ]]; then
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
          "$train_python" -u -m blue.training.architecture_compare train \
          --arch "$arch" --seed "$rng_seed" --out "$out" --resume \
          --train-seeds "${train_seeds[@]}" --eval-seeds "${eval_seeds[@]}" \
          --iters 6 --eps-per-iter 4 --steps 400 --threads 1
      fi
    else
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$train_python" -u -m blue.training.architecture_compare train \
        --arch "$arch" --seed "$rng_seed" --out "$out" \
        --train-seeds "${train_seeds[@]}" --eval-seeds "${eval_seeds[@]}" \
        --iters 6 --eps-per-iter 4 --steps 400 --threads 1
    fi
    if [[ ! -f "$report" ]]; then
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$train_python" -u -m blue.training.architecture_compare eval \
        --arch "$arch" --seed "$rng_seed" --model-dir "$out" \
        --out "$report" --train-seeds "${train_seeds[@]}" \
        --eval-seeds "${eval_seeds[@]}" --steps 400 --threads 1
    fi
  done
}

run_arch set >"$base/logs/set.log" 2>&1 & pid_set=$!
run_arch fixed >"$base/logs/fixed.log" 2>&1 & pid_fixed=$!
run_arch hybrid >"$base/logs/hybrid.log" 2>&1 & pid_hybrid=$!
status=0
wait "$pid_set" || status=1
wait "$pid_fixed" || status=1
wait "$pid_hybrid" || status=1
exit "$status"
