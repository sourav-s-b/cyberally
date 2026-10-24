#!/usr/bin/env bash
# Frozen equal-budget Qwen event-time RL main: 5 RNGs x 2 credit arms.
# Run only after the four-run pilot gate passes. Reruns resume by iteration.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
train_python="${CYBERALLY_TRAIN_PYTHON:-/home/sourav/projects/cyberally/.venv-train/bin/python}"
base="blue/results/event_time_rl"

"$train_python" -m blue.training.event_time_pilot_gate \
  --base "$base" --out "$base/pilot_gate.json"

for arm in mc gae; do
  if [[ "$arm" == mc ]]; then
    gae_lambda=1.0
  else
    gae_lambda=0.9
  fi
  for rng_seed in 0 1 2 3 4; do
    out="$base/main_${arm}_s${rng_seed}"
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
      --seed "$rng_seed" \
      --train-seeds 7706 7707 7708 7709 7710 7711 7712 7713 \
                    7714 7715 7716 7717 7718 7719 7720 7721 \
      --steps 400 --iters 12 --eps-per-iter 4 \
      --gae-lambda "$gae_lambda" \
      --bonus 1.0 --explore 0.5 --lr 0.003 --critic-lr 0.0003 \
      --temp 0.5 --out "$out" "${resume[@]}" \
      >> "$out/main.log" 2>&1
    echo "finished: $out"
  done
done
