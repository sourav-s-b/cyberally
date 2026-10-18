#!/usr/bin/env bash
# Fixed-budget finite-Red PPO comparison. Every child stays in the foreground.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${BLUE_TRAIN_PYTHON:-/home/sourav/projects/cyberally/.venv-train/bin/python}
SCORER=${BLUE_HARNESS_SCORER:-/home/sourav/projects/cyberally/blue/results/gpt_harness/scorer.pkl}
OUT=${BLUE_ANCHOR_PILOT_OUT:-$ROOT/blue/results/finite_red_anchor_pilot_v1}
SEEDS=$(seq 7665 7688)

run_python() {
  ( cd "$ROOT" && env PYTHONPATH=/home/sourav/projects/cyberally/blue/results/harness_runtime:"$ROOT" \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$@" )
}

train_one() {
  local arm=$1 seed=$2 version dir status
  dir="$OUT/train/$arm/s$seed"
  version=current-residual-v1
  [ "$arm" != anchored ] || version=anchored-categorical-v1
  if [ -f "$dir/manifest.json" ]; then
    status=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$dir/manifest.json")
    [ "$status" != complete ] || { echo "complete train $arm s$seed"; return; }
    [ -f "$dir/checkpoint.pt" ] || { echo "incomplete checkpoint missing $dir" >&2; return 1; }
    resume=(--resume)
  else
    [ ! -e "$dir" ] || { echo "unmanifested directory $dir" >&2; return 1; }
    resume=()
  fi
  mkdir -p "$dir"
  run_python -m blue.training.harness_rl --out "$dir" --seed "$seed" \
    --iters 8 --eps-per-iter 4 --steps 400 --hidden 64 --lr 3e-4 \
    --bonus .25 --temp .5 --max-age 80 --ml-inputs both \
    --train-seeds 7706 7707 7708 7709 --red-agent finite \
    --state-shaping off --policy-version "$version" \
    --reference-mass .9 --reference-temp .5 --reference-kl-coef .05 \
    --max-update-kl .02 --scorer "$SCORER" "${resume[@]}" \
    > "$dir.log" 2>&1
  echo "complete train $arm s$seed"
}

control_one() {
  local arm=$1 dir="$OUT/eval/$1"
  [ ! -e "$dir" ] || { [ -f "$dir/report.json" ] && return; echo "partial control $dir" >&2; return 1; }
  mkdir -p "$OUT/eval"
  run_python -m blue.training.harness_eval --arms "$arm" --seeds $SEEDS \
    --steps 400 --max-age 80 --red-agent finite --scorer "$SCORER" \
    --out "$dir" > "$dir.log" 2>&1
  echo "complete control $arm"
}

eval_one() {
  local arm=$1 mode=$2 seed=$3 model="$OUT/train/$1/s$3" dir="$OUT/eval/${1}_${2}_s$3"
  [ -f "$model/manifest.json" ] || { echo "missing model $model" >&2; return 1; }
  [ ! -e "$dir" ] || { [ -f "$dir/report.json" ] && return; echo "partial evaluation $dir" >&2; return 1; }
  mkdir -p "$OUT/eval"
  run_python -m blue.training.harness_eval --arms actor --actor-mode "$mode" \
    --sample-seed-base 900000 --model-dir "$model" --seeds $SEEDS \
    --steps 400 --max-age 80 --red-agent finite --scorer "$SCORER" \
    --out "$dir" > "$dir.log" 2>&1
  echo "complete eval $arm $mode s$seed"
}

case "${1:-plan}" in
  plan) echo "finite Red anchored PPO; 3 seeds x 2 arms x 8 iterations x 4 episodes; 24 eval seeds" ;;
  setup) run_python -m blue.training.harness_anchor_pilot setup --root "$OUT" --scorer "$SCORER" ;;
  preflight) run_python -m blue.training.harness_anchor_pilot preflight --root "$OUT" --scorer "$SCORER" ;;
  train)
    [ -f "$OUT/preflight/report.json" ] || { echo "preflight required" >&2; exit 1; }
    for arm in current anchored; do
      for seed in 0 1 2; do
        train_one_args+=("$arm $seed")
      done
    done
    printf '%s\n' "${train_one_args[@]}" | "$ROOT/scripts/blue_xargs_pool.sh" 3 2 "$0" __train
    ;;
  controls)
    [ -f "$OUT/frozen_plan.json" ] || { echo "setup required" >&2; exit 1; }
    printf '%s\n' lancer guard | "$ROOT/scripts/blue_xargs_pool.sh" 2 1 "$0" __control
    ;;
  eval)
    [ -f "$OUT/eval/lancer/report.json" ] && [ -f "$OUT/eval/guard/report.json" ] || { echo "controls required" >&2; exit 1; }
    for arm in current anchored; do
      for mode in greedy sampled; do
        for seed in 0 1 2; do
          eval_one_args+=("$arm $mode $seed")
        done
      done
    done
    printf '%s\n' "${eval_one_args[@]}" | "$ROOT/scripts/blue_xargs_pool.sh" 3 3 "$0" __eval
    ;;
  analyze) run_python -m blue.training.harness_anchor_pilot analyze --root "$OUT" --scorer "$SCORER" ;;
  __train) shift; train_one "$1" "$2" ;;
  __control) shift; control_one "$1" ;;
  __eval) shift; eval_one "$1" "$2" "$3" ;;
  *) echo "usage: $0 {plan|setup|preflight|train|controls|eval|analyze}" >&2; exit 2 ;;
esac
