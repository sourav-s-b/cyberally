#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CYBERALLY_TRAIN_PYTHON:-/home/sourav/projects/cyberally/.venv-train/bin/python}"
SCORER="/home/sourav/projects/cyberally/blue/results/gpt_harness/scorer.pkl"
OUT="$ROOT/blue/results/mixed_red_investigator_ppo_20261025"

cd "$ROOT"
for seed in 0 1 2 3 4; do
  run_dir="$OUT/s${seed}"
  mkdir -p "$run_dir"
  resume=()
  if [[ -f "$run_dir/manifest.json" ]]; then
    status="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", "unknown"))' "$run_dir/manifest.json")"
    if [[ "$status" == "complete" ]]; then
      echo "skip completed model seed=$seed"
      continue
    fi
    resume=(--resume)
  fi
  echo "train model seed=$seed (${resume[*]:-new})"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH="$ROOT" \
    "$PYTHON" -m blue.training.harness_rl \
      --out "$run_dir" \
      --scorer "$SCORER" \
      --seed "$seed" \
      --train-seeds 7600 7601 7602 7603 7604 7605 7606 7607 7608 7609 7610 7611 7612 \
      --red-agents finite discovery \
      --policy-version anchored-categorical-v1 \
      --iters 8 --eps-per-iter 4 --steps 400 --hidden 64 \
      --lr 0.0003 --bonus 0.25 --temp 0.5 --max-age 80 \
      --ml-inputs both --state-shaping off \
      --reference-mass 0.9 --reference-temp 0.5 \
      --reference-kl-coef 0.05 --max-update-kl 0.02 \
      "${resume[@]}"
done
