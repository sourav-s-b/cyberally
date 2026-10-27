#!/usr/bin/env bash
# Pre-registered Qwen event-time RL pilot: two training RNGs for each arm.
# Rerunning this file skips completed runs and resumes interrupted runs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
train_python="${CYBERALLY_TRAIN_PYTHON:-/home/sourav/projects/cyberally/.venv-train/bin/python}"
base="blue/results/event_time_rl"

for arm in mc gae; do
  if [[ "$arm" == mc ]]; then
    gae_lambda=1.0
  else
    gae_lambda=0.9
  fi
  for rng_seed in 0 1; do
    out="$base/pilot_${arm}_s${rng_seed}"
    mkdir -p "$out"
    if [[ -f "$out/manifest.json" ]]; then
      status="$($train_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$out/manifest.json")"
      if [[ "$status" == complete && -f "$out/checkpoint.pt" && -f "$out/actor.th" ]]; then
        echo "already complete: $out"
        continue
      fi
      if [[ -f "$out/checkpoint.pt" ]]; then
        resume=(--resume)
      else
        # A machine restart before iteration 1 leaves a manifest but no
        # resumable optimizer/RNG state. Preserve that trace and restart.
        archived="${out}_before_first_checkpoint_$(date +%Y%m%d_%H%M%S)"
        mv "$out" "$archived"
        mkdir -p "$out"
        echo "archived nonresumable start: $archived"
        resume=()
      fi
    else
      resume=()
    fi
    echo "starting: $out (lambda=$gae_lambda, rng=$rng_seed)"
    "$train_python" -u -m blue.training.event_time_rl train \
      --red-agent llm --model qwen2.5:3b \
      --scorer "$base/scorer.pkl" \
      --seed "$rng_seed" --train-seeds 7706 7707 7708 7709 \
      --steps 400 --iters 2 --eps-per-iter 2 \
      --gae-lambda "$gae_lambda" \
      --bonus 1.0 --explore 0.5 --lr 0.003 --critic-lr 0.0003 \
      --temp 0.5 --out "$out" "${resume[@]}" \
      >> "$out/pilot.log" 2>&1
    echo "finished: $out"
  done
done
