#!/usr/bin/env bash
# Cross-Red evaluation of frozen BC checkpoints; resumes episode artifacts.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${BLUE_TRAIN_PYTHON:-$ROOT/../../.venv-train/bin/python}
SCORER=${BLUE_HARNESS_SCORER:-$ROOT/../../blue/results/gpt_harness/scorer.pkl}
OUT=${BLUE_BC_REDSUITE_OUT:-$ROOT/blue/results/finite_red_bc_redsuite_20261018}
WORKERS=${BLUE_BC_REDSUITE_WORKERS:-3}
RED_MODES=${BLUE_BC_REDSUITE_RED_MODES:-"discovery random sleep"}

need_runtime() {
  [ -x "$PY" ] || { echo "training Python not found: $PY" >&2; return 1; }
  [ -f "$SCORER" ] || { echo "scorer not found: $SCORER" >&2; return 1; }
}

run_task() {
  local job=$1 red=${1%%/*} task=${1#*/}
  local log="$OUT/logs/$red-${task//:/_}.stderr.log"
  mkdir -p "$OUT/.locks" "$OUT/logs"
  if ! ( cd "$ROOT" && flock "$OUT/.locks/$red-$task.lock" \
      env PYTHONPATH="$ROOT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$PY" -m blue.training.harness_bc_redsuite --out "$OUT" \
        --scorer "$SCORER" --red "$red" --task "$task" ) > /dev/null 2> "$log"; then
    echo "FAILED $red/$task; stderr log: $log" >&2
    return 1
  fi
  echo "complete $red/$task"
}

if [ "${1:-}" = __task ]; then
  need_runtime
  run_task "$2"
  exit 0
fi
need_runtime
if [ "${1:-}" = --freeze ]; then
  ( cd "$ROOT" && env PYTHONPATH="$ROOT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$PY" -m blue.training.harness_bc_redsuite --out "$OUT" --scorer "$SCORER" --freeze )
  exit 0
fi
if [ "${1:-}" = --analyze ]; then
  ( cd "$ROOT" && env PYTHONPATH="$ROOT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$PY" -m blue.training.analyze_bc_redsuite --out "$OUT" --scorer "$SCORER" )
  exit 0
fi
if [ ! -f "$OUT/frozen_plan.json" ]; then
  ( cd "$ROOT" && env PYTHONPATH="$ROOT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$PY" -m blue.training.harness_bc_redsuite --out "$OUT" --scorer "$SCORER" --freeze )
fi
tasks=()
for red in $RED_MODES; do
  case "$red" in
    discovery|verbose|random|sleep) ;;
    *) echo "unknown Red mode: $red" >&2; exit 2 ;;
  esac
  for control in lancer guard sleep masked_random; do tasks+=("$red/$control"); done
  for model_seed in 0 1 2 3 4; do tasks+=("$red/bc:s$model_seed"); done
done
echo "Cross-Red BC evaluation; shared episode-worker limit=$WORKERS"
printf '%s\n' "${tasks[@]}" | "$ROOT/scripts/blue_xargs_pool.sh" \
  "$WORKERS" 1 "$0" __task
if [ "$RED_MODES" != "discovery random sleep" ]; then
  echo "Completed selected Red modes: $RED_MODES; full analysis waits for all modes."
  exit 0
fi
( cd "$ROOT" && env PYTHONPATH="$ROOT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    "$PY" -m blue.training.analyze_bc_redsuite --out "$OUT" --scorer "$SCORER" )
