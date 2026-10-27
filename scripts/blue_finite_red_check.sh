#!/usr/bin/env bash
# Native-reward Blue learning check against finite Red.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${BLUE_TRAIN_PYTHON:-$ROOT/.venv-train/bin/python}
SCORER=${BLUE_HARNESS_SCORER:-$ROOT/blue/results/gpt_harness/scorer.pkl}
OUT=${BLUE_FINITE_RED_OUT:-$ROOT/blue/results/finite_red_native}
ITERS=32 EPS_PER_ITER=4 STEPS=400 MAX_AGE=80
TRAIN_SEEDS="7706 7707 7708 7709"
TRAIN_RNG_SEEDS=${BLUE_FINITE_RED_TRAIN_RNG_SEEDS:-"0 1 2"}
EVAL_SEEDS=$(seq 7665 7688)

need_runtime() {
  [ -x "$PY" ] || { echo "training Python not found: $PY" >&2; return 1; }
  [ -f "$SCORER" ] || { echo "scorer not found: $SCORER" >&2; return 1; }
}

train_one() {
  local seed=$1 dir="$OUT/train/s$1" status
  local resume=()
  if [ -f "$dir/manifest.json" ]; then
    status=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "$dir/manifest.json")
    if [ "$status" = complete ]; then echo "skip complete seed $seed"; return 0; fi
    [ -f "$dir/checkpoint.pt" ] || { echo "incomplete run has no checkpoint: $dir" >&2; return 1; }
    resume=(--resume)
  elif [ -e "$dir" ]; then
    echo "refusing directory without manifest: $dir" >&2
    return 1
  fi
  mkdir -p "$dir"
  echo "train seed=$seed resume=${resume[*]:-no}"
  ( cd "$ROOT" && env PYTHONPATH="$ROOT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$PY" -m blue.training.harness_rl --out "$dir" --seed "$seed" \
        --iters "$ITERS" --eps-per-iter "$EPS_PER_ITER" --steps "$STEPS" \
        --hidden 64 --lr 3e-4 --bonus .25 --temp .5 --max-age "$MAX_AGE" \
        --ml-inputs both --train-seeds $TRAIN_SEEDS --red-agent finite \
        --state-shaping off --scorer "$SCORER" "${resume[@]}" \
        > "$dir.log" 2>&1 )
}

eval_one() {
  local seed=$1 dir="$OUT/train/s$1" tag="$OUT/eval/actor_s$1"
  [ -f "$dir/actor.th" ] || { echo "missing actor: $dir" >&2; return 1; }
  if [ -f "$tag/report.json" ]; then echo "skip existing $tag"; return 0; fi
  mkdir -p "$OUT/eval"
  ( cd "$ROOT" && env PYTHONPATH="$ROOT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$PY" -m blue.training.harness_eval --model-dir "$dir" --arms actor \
        --seeds $EVAL_SEEDS --steps "$STEPS" --max-age "$MAX_AGE" \
        --red-agent finite --scorer "$SCORER" \
        --lancer-cache "$OUT/eval/lancer.json" --read-only-lancer-cache \
        --out "$tag" > "$tag.log" 2>&1 )
}

run_pool() {
  local workers=$1 task=$2; shift 2
  printf '%s\n' "$@" | "$ROOT/scripts/blue_xargs_pool.sh" "$workers" 1 "$0" "$task"
}

do_train() {
  need_runtime
  local workers=${1:-3} tasks=() seed
  for seed in $TRAIN_RNG_SEEDS; do tasks+=("$seed"); done
  echo "training native-reward finite Red, concurrency <= $workers"
  run_pool "$workers" __train "${tasks[@]}"
}

do_lancer() {
  need_runtime
  local cache="$OUT/eval/lancer.json" report="$OUT/eval/lancer"
  if [ -f "$cache" ] && "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); k={"red_agent":"finite","steps":400,"seeds":list(range(7665,7689))}; sys.exit(0 if d.get("key")==k and set(d.get("returns",{}))==set(map(str,k["seeds"])) else 1)' "$cache"; then
    echo "reusing complete finite-Red Lancer cache: $cache"
    return 0
  fi
  [ ! -e "$report" ] || report="$OUT/eval/lancer_$(date +%s)"
  ( cd "$ROOT" && env PYTHONPATH="$ROOT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$PY" -m blue.training.harness_eval --arms lancer --seeds $EVAL_SEEDS \
        --steps "$STEPS" --max-age "$MAX_AGE" --red-agent finite \
        --scorer "$SCORER" --lancer-cache "$cache" --out "$report" )
}

do_context() {
  need_runtime
  local report="$OUT/eval/context"
  if [ -f "$report/report.json" ]; then echo "context exists: $report"; return; fi
  [ ! -e "$report" ] || report="$OUT/eval/context_$(date +%s)"
  ( cd "$ROOT" && env PYTHONPATH="$ROOT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$PY" -m blue.training.harness_eval --arms sleep masked_random \
        --seeds $EVAL_SEEDS --steps "$STEPS" --max-age "$MAX_AGE" \
        --red-agent finite --scorer "$SCORER" --out "$report" )
}

do_eval() {
  need_runtime
  [ -f "$OUT/eval/lancer.json" ] || { echo "run lancer first" >&2; return 1; }
  local workers=${1:-3} tasks=() seed
  for seed in $TRAIN_RNG_SEEDS; do tasks+=("$seed"); done
  run_pool "$workers" __eval "${tasks[@]}"
}

do_analyze() {
  need_runtime
  local context models=() reports=() seeds=() seed context_report
  context_report=$(find "$OUT/eval" -maxdepth 2 -path "$OUT/eval/context*/report.json" -type f | sort | tail -n 1)
  [ -n "$context_report" ] || { echo "Sleep/masked-random report missing; run context" >&2; return 1; }
  for seed in $TRAIN_RNG_SEEDS; do
    models+=("$OUT/train/s$seed")
    reports+=("$OUT/eval/actor_s$seed/report.json")
    seeds+=("$seed")
  done
  ( cd "$ROOT" && env PYTHONPATH="$ROOT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$PY" -m blue.training.harness_finite_analysis --models "${models[@]}" \
        --evaluations "${reports[@]}" --train-seeds "${seeds[@]}" \
        --lancer "$OUT/eval/lancer.json" --context "$context_report" \
        --out "$OUT/analysis.json" )
}

case "${1:-plan}" in
  plan) echo "finite Red; native reward; training seeds: $TRAIN_RNG_SEEDS; eval seeds: 7665-7688; controls: Lancer, Sleep, masked-random" ;;
  pilot) need_runtime; (cd "$ROOT" && PYTHONPATH="$ROOT" "$PY" -m blue.training.harness_shape_pilot --seeds 7706 7707 --red-agent finite --state-shaping confirm --out "$OUT/pilot-confirm.json") ;;
  train) do_train "${2:-3}" ;;
  lancer) do_lancer ;;
  context) do_context ;;
  eval) do_eval "${2:-3}" ;;
  analyze) do_analyze ;;
  __train) shift; train_one "$1" ;;
  __eval) shift; eval_one "$1" ;;
  *) echo "usage: $0 {plan|pilot|train [workers]|lancer|context|eval [workers]|analyze}" >&2; exit 2 ;;
esac
