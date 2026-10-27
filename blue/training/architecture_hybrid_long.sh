#!/usr/bin/env bash
# Restartable, longer two-RNG run for the situation-routed hybrid actor.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
train_python="${CYBERALLY_TRAIN_PYTHON:-/home/sourav/projects/cyberally/.venv-train/bin/python}"
base="blue/results/architecture_hybrid_long_v1"
eval_seeds=(8401 8402 8403 8404 8405 8406 8407 8408 \
            8409 8410 8411 8412 8413 8414 8415 8416)
mkdir -p "$base/evaluations"

for rng_seed in 0 1; do
  out="$base/hybrid_s${rng_seed}"
  if [[ "$rng_seed" == 0 ]]; then
    train_seeds=(8253 8254 8255 8256 8257 8258 8259 8260 \
                 8261 8262 8263 8264 8265 8266 8267 8268 \
                 8269 8270 8271 8272 8273 8274 8275 8276)
  else
    train_seeds=(8277 8278 8279 8280 8281 8282 8283 8284 \
                 8285 8286 8287 8288 8289 8290 8291 8292 \
                 8293 8294 8295 8296 8297 8298 8299 8300)
  fi
  mkdir -p "$out"
  if [[ -f "$out/manifest.json" ]]; then
    status="$("$train_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$out/manifest.json")"
    if [[ "$status" != complete ]]; then
      "$train_python" -u -m blue.training.architecture_compare train \
        --arch hybrid --seed "$rng_seed" --out "$out" --resume \
        --train-seeds "${train_seeds[@]}" --eval-seeds "${eval_seeds[@]}" \
        --iters 6 --eps-per-iter 4 --steps 400
    fi
  else
    "$train_python" -u -m blue.training.architecture_compare train \
      --arch hybrid --seed "$rng_seed" --out "$out" \
      --train-seeds "${train_seeds[@]}" --eval-seeds "${eval_seeds[@]}" \
      --iters 6 --eps-per-iter 4 --steps 400
  fi
  report="$base/evaluations/hybrid_s${rng_seed}.json"
  if [[ ! -f "$report" ]]; then
    "$train_python" -u -m blue.training.architecture_compare eval \
      --arch hybrid --seed "$rng_seed" --model-dir "$out" --out "$report" \
      --train-seeds "${train_seeds[@]}" --eval-seeds "${eval_seeds[@]}" \
      --steps 400
  fi
done
