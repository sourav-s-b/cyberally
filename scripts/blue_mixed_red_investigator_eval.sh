#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CYBERALLY_TRAIN_PYTHON:-/home/sourav/projects/cyberally/.venv-train/bin/python}"
SCORER="/home/sourav/projects/cyberally/blue/results/gpt_harness/scorer.pkl"
OUT="$ROOT/blue/results/mixed_red_investigator_ppo_20261025"
SEEDS=(7641 7642 7643 7644 7645 7646 7647 7648 7649 7650 7651 7652 7653 7654 7655 7656)

cd "$ROOT"
for red in finite discovery; do
  baseline="$OUT/baseline_$red"
  cache="$OUT/lancer_$red.json"
  if [[ ! -f "$baseline/report.json" ]]; then
    mkdir -p "$baseline"
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH="$ROOT" \
      "$PYTHON" -m blue.training.harness_eval \
        --seeds "${SEEDS[@]}" --arms guard lancer \
        --red-agent "$red" --steps 400 --max-age 80 --scorer "$SCORER" \
        --lancer-cache "$cache" --out "$baseline"
  fi
  for seed in 0 1 2 3 4; do
    model="$OUT/s$seed"
    report="$OUT/eval_s${seed}_$red"
    if [[ -f "$report/report.json" ]]; then
      echo "skip completed evaluation model=$seed red=$red"
      continue
    fi
    mkdir -p "$report"
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH="$ROOT" \
      "$PYTHON" -m blue.training.harness_eval \
        --seeds "${SEEDS[@]}" --arms actor \
        --red-agent "$red" --steps 400 --max-age 80 --scorer "$SCORER" \
        --model-dir "$model" --lancer-cache "$cache" \
        --read-only-lancer-cache --out "$report"
  done
done

PYTHONPATH="$ROOT" "$PYTHON" -m blue.training.harness_mixed_red_analysis \
  --root "$OUT" --out "$OUT/analysis.json"
