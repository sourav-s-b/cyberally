#!/usr/bin/env bash
# Restartable two-seed SetActor vs candidate-attention RL pilot.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
train_python="${CYBERALLY_TRAIN_PYTHON:-/home/sourav/projects/cyberally/.venv-train/bin/python}"
base="blue/results/architecture_cmp_v1"
mkdir -p "$base/evaluations"

for arch in set attention; do
  for rng_seed in 0 1; do
    out="$base/${arch}_s${rng_seed}"
    mkdir -p "$out"
    if [[ -f "$out/manifest.json" ]]; then
      status="$("$train_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$out/manifest.json")"
      if [[ "$status" == complete ]]; then
        echo "already complete: $out"
      else
        "$train_python" -u -m blue.training.architecture_compare train \
          --arch "$arch" --seed "$rng_seed" --out "$out" --resume \
          --train-seeds 7616 7617 7618 7619 --eval-seeds 7620 7621
      fi
    else
      "$train_python" -u -m blue.training.architecture_compare train \
        --arch "$arch" --seed "$rng_seed" --out "$out" \
        --train-seeds 7616 7617 7618 7619 --eval-seeds 7620 7621
    fi
    report="$base/evaluations/${arch}_s${rng_seed}.json"
    if [[ ! -f "$report" ]]; then
      "$train_python" -u -m blue.training.architecture_compare eval \
        --arch "$arch" --seed "$rng_seed" --model-dir "$out" \
        --out "$report" --train-seeds 7616 7617 7618 7619 \
        --eval-seeds 7620 7621
    fi
  done
done
