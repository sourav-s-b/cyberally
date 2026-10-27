#!/usr/bin/env bash
# Development-only Qwen evaluation of every pilot's saved second iteration.
# Each invocation persists whole seeds and can resume after a machine restart.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
train_python="${CYBERALLY_TRAIN_PYTHON:-/home/sourav/projects/cyberally/.venv-train/bin/python}"
base="blue/results/event_time_rl"

for arm in mc gae; do
  for rng_seed in 0 1; do
    model_dir="$base/pilot_${arm}_s${rng_seed}"
    out="$base/pilot_eval_${arm}_s${rng_seed}"
    mkdir -p "$out"
    if [[ -f "$out/report.json" ]]; then
      status="$($train_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$out/report.json")"
      if [[ "$status" == complete ]]; then
        echo "already complete: $out"
        continue
      fi
      resume=(--resume)
    else
      resume=()
    fi
    echo "evaluating: $model_dir -> $out"
    "$train_python" -u -m blue.training.event_time_rl eval \
      --model-dir "$model_dir" --checkpoint 2 \
      --scorer "$base/scorer.pkl" \
      --eval-seeds 7641 7642 7643 7644 \
      --steps 400 --policy-replicas 1 --out "$out" "${resume[@]}" \
      >> "$out/eval.log" 2>&1
    echo "finished: $out"
  done
done
