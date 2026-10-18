#!/usr/bin/env bash
# Fresh-seed confirmation for the frozen finite-Red BC/DAgger models.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${BLUE_TRAIN_PYTHON:-$ROOT/../../.venv-train/bin/python}
SCORER=${BLUE_HARNESS_SCORER:-$ROOT/../../blue/results/gpt_harness/scorer.pkl}
OUT=${BLUE_DAGGER_CONFIRM_OUT:-$ROOT/blue/results/finite_red_dagger_bc_confirm_20261018}
WORKERS=${BLUE_DAGGER_CONFIRM_WORKERS:-3}

need_runtime() {
  [ -x "$PY" ] || { echo "training Python not found: $PY" >&2; return 1; }
  [ -f "$SCORER" ] || { echo "scorer not found: $SCORER" >&2; return 1; }
}

run_task() {
  local task=$1
  ( cd "$ROOT" && env PYTHONPATH="$ROOT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$PY" -m blue.training.harness_dagger_confirm --out "$OUT" \
        --scorer "$SCORER" --task "$task" )
}

if [ "${1:-}" = __task ]; then
  need_runtime
  run_task "$2"
  exit 0
fi

need_runtime
if [ "${1:-}" = --freeze ]; then
  ( cd "$ROOT" && env PYTHONPATH="$ROOT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$PY" -m blue.training.harness_dagger_confirm --out "$OUT" \
        --scorer "$SCORER" --freeze )
  exit 0
fi
if [ "${1:-}" = --analyze ]; then
  ( cd "$ROOT" && env PYTHONPATH="$ROOT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$PY" -m blue.training.harness_dagger_confirm --out "$OUT" \
        --scorer "$SCORER" --analyze )
  exit 0
fi
if [ ! -f "$OUT/frozen_plan.json" ]; then
  ( cd "$ROOT" && env PYTHONPATH="$ROOT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$PY" -m blue.training.harness_dagger_confirm --out "$OUT" \
        --scorer "$SCORER" --freeze )
fi

tasks=(lancer guard sleep masked_random)
for arm in bc dagger; do
  for model_seed in 0 1 2 3 4; do
    tasks+=("$arm:s$model_seed")
  done
done
echo "Confirming frozen BC/DAgger models; shared episode-worker limit=$WORKERS"
printf '%s\n' "${tasks[@]}" | "$ROOT/scripts/blue_xargs_pool.sh" \
  "$WORKERS" 1 "$0" __task
( cd "$ROOT" && env PYTHONPATH="$ROOT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    "$PY" -m blue.training.harness_dagger_confirm --out "$OUT" \
      --scorer "$SCORER" --analyze )
