#!/usr/bin/env bash
# Bounded finite-Red synchronous-A2C vs JSRL-style guide-prefix screen.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${BLUE_TRAIN_PYTHON:-/home/sourav/projects/cyberally/.venv-train/bin/python}
SCORER=${BLUE_HARNESS_SCORER:-/home/sourav/projects/cyberally/blue/results/gpt_harness/scorer.pkl}
OUT=${BLUE_JSRL_A2C_OUT:-$ROOT/blue/results/finite_red_jsrl_a2c_screen_v1}
TRAIN_ENVS="7706 7707 7708 7709"
RNG_SEEDS="0 1 2"
GUIDE="267 267 134 134 67 0 0 0"
EVAL_SEEDS=$(seq 9001 9032)

run_python() {
  ( cd "$ROOT" && env PYTHONPATH=/home/sourav/projects/cyberally/blue/results/harness_runtime:"$ROOT" \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$@" )
}

need_runtime() {
  [ -x "$PY" ] || { echo "training Python not found: $PY" >&2; return 1; }
  [ -f "$SCORER" ] || { echo "scorer not found: $SCORER" >&2; return 1; }
}

train_one() {
  local arm=$1 seed=$2 dir="$OUT/train/$1/s$2" resume=()
  local schedule="${GUIDE}"
  if [ "$arm" = a2c ]; then schedule="0"; fi
  if [ -f "$dir/manifest.json" ]; then
    local status
    status=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "$dir/manifest.json")
    if [ "$status" = complete ]; then echo "skip complete $arm s$seed"; return 0; fi
    [ -f "$dir/checkpoint.pt" ] || { echo "incomplete run has no checkpoint: $dir" >&2; return 1; }
    resume=(--resume)
  elif [ -e "$dir" ]; then
    echo "refusing unmanifested model directory: $dir" >&2
    return 1
  fi
  mkdir -p "$dir"
  run_python -m blue.training.harness_redesign train --algorithm a2c \
    --red-agent finite --seed "$seed" --train-seeds $TRAIN_ENVS \
    --iters 8 --eps-per-iter 4 --steps 400 --hidden 64 --lr 3e-4 \
    --critic-lr 3e-4 --bonus .25 --temp .5 --explore .2 --max-age 80 \
    --gamma .99 --gae-lambda 1 --ml-inputs risk \
    --guide-prefix-schedule $schedule --scorer "$SCORER" --out "$dir" \
    "${resume[@]}" > "$dir.log" 2>&1
  echo "complete $arm s$seed"
}

case "${1:-plan}" in
  plan)
    echo "finite Red; native return; A2C vs guide-prefix A2C; 3 training RNG seeds; 32 paired evaluation seeds; Lancer/guard/Sleep/masked-random controls"
    ;;
  setup)
    need_runtime
    # This block was searched across docs and saved JSON/log reports before freezing.
    if rg -n --glob '*.json' --glob '*.md' --glob '*.log' \
      '\b900[1-9]\b|\b901[0-9]\b|\b902[0-9]\b|\b903[0-2]\b' \
      "$ROOT/docs" "$ROOT/blue/results" 2>/dev/null | rg -v 'finite_red_jsrl_a2c_screen_v1' >/dev/null; then
      echo "evaluation seed block 9001-9032 already appears in saved artifacts/docs" >&2
      exit 1
    fi
    run_python -m blue.training.harness_jsrl_screen freeze --root "$OUT" --scorer "$SCORER"
    run_python -m blue.training.harness_jsrl_screen preflight --root "$OUT" --scorer "$SCORER"
    ;;
  train)
    need_runtime
    [ -f "$OUT/preflight.json" ] || { echo "setup/preflight required" >&2; exit 1; }
    tasks=()
    for arm in a2c jsrl_a2c; do for seed in $RNG_SEEDS; do tasks+=("$arm $seed"); done; done
    printf '%s\n' "${tasks[@]}" | "$ROOT/scripts/blue_xargs_pool.sh" "${BLUE_JSRL_WORKERS:-3}" 2 "$0" __train
    ;;
  controls)
    need_runtime
    [ -f "$OUT/frozen_plan.json" ] || { echo "setup required" >&2; exit 1; }
    run_python -m blue.training.harness_jsrl_screen controls --root "$OUT" --scorer "$SCORER"
    ;;
  eval)
    need_runtime
    [ -f "$OUT/controls.json" ] || { echo "shared controls required" >&2; exit 1; }
    tasks=()
    for arm in a2c jsrl_a2c; do for seed in $RNG_SEEDS; do tasks+=("$arm $seed"); done; done
    printf '%s\n' "${tasks[@]}" | "$ROOT/scripts/blue_xargs_pool.sh" "${BLUE_JSRL_WORKERS:-3}" 2 "$0" __eval
    ;;
  analyze)
    need_runtime
    run_python -m blue.training.harness_jsrl_screen analyze --root "$OUT" --scorer "$SCORER"
    ;;
  __train) shift; train_one "$1" "$2" ;;
  __eval)
    shift
    arm=$1 seed=$2
    run_python -m blue.training.harness_jsrl_screen eval-model --root "$OUT" \
      --scorer "$SCORER" --arm "$arm" --train-rng-seed "$seed" \
      --model-dir "$OUT/train/$arm/s$seed" > "$OUT/eval_${arm}_s${seed}.log" 2>&1
    ;;
  *) echo "usage: $0 {plan|setup|train|controls|eval|analyze}" >&2; exit 2 ;;
esac
