#!/usr/bin/env bash
# OP2 prereg pipeline (run_id op2-jsrl-guide-8433-20261022, FROZEN):
# seq train -> dev eval -> frozen analysis vs the G1 guard re-measure.
# Training is SEQUENTIAL by design (prereg: one process at a time); launch
# the whole stage detached so it survives tool timeouts:
#   setsid nohup blue/training/op2_pipeline.sh all > <log> 2>&1 &
# Crash mid-training: re-runing the stage resumes the last dir via
# --resume-from with the same frozen config (G2: no config change).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY=/home/sourav/projects/cyberally/.venv-train/bin/python
RES=blue/results/op2_jsrl_guide_v1
TRAIN_SEEDS="8433 8434 8435 8436 8437 8438 8439 8440 8441 8442 8443 8444"
DEV_SEEDS="8445 8446 8447 8448 8449 8450 8451 8452 8453 8454 8455 8456 8457 8458 8459 8460 8461 8462 8463 8464"
GUIDE="--guide-h0 200 --guide-decay 0.6 --guide-floor 10 --guide-max-age 80.0"
NP_ITERS=12

train_one() {
  local arm=$1 s=$2 out="$RES/${1}_s${2}"
  local extra="" done_it=0
  if [[ -f "$out/pilot_manifest.json" ]]; then
    done_it=$("$PY" -c "import json;print(json.load(open('$out/pilot_manifest.json'))['iters_done'])")
    if [[ "$done_it" -ge $NP_ITERS ]]; then
      echo "skip train ${arm}_s${s}: ${done_it}/${NP_ITERS} iters done"
      return 0
    fi
    extra="--resume-from $out"
    echo "resuming ${arm}_s${s} from iter $((done_it + 1))"
  fi
  local flags=""
  if [[ $arm == jsrl ]]; then flags=$GUIDE; fi
  "$PY" -m blue.training.mappo_guide --mode train \
    --train-seeds $TRAIN_SEEDS \
    --steps 400 --iters $NP_ITERS --eps-per-iter 4 --hidden 64 \
    --lr 3e-4 --bonus 1.0 --temperature 1.0 --seed "$s" \
    --min-coverage 0.0 --min-agree 0.0 --max-kl 10.0 \
    --phase3-model blue/results/scorer_mlp.npz \
    $flags $extra --red-agent finite --out "$out"
}

eval_one() {
  local out=$1 json_path="$1/eval_dev.json"
  if [[ -f "$json_path" ]]; then
    echo "skip eval ${out}: eval_dev.json exists"
    return 0
  fi
  "$PY" -m blue.training.mappo_guide --mode eval \
    --model-dir "$out" \
    --eval-seeds $DEV_SEEDS --steps 400 --red-agent finite \
    --eval-arms learned --bonus 1.0 --hidden 64 \
    --out "$json_path"
}

analyze() {
  local args=()
  for d in "$RES"/noguide_s?; do
    args+=(--label noguide --eval-json "$d/eval_dev.json")
  done
  for d in "$RES"/jsrl_s?; do
    args+=(--label jsrl --eval-json "$d/eval_dev.json")
  done
  "$PY" -m blue.training.op2_dev_analysis \
    --g1-report "$RES/guard_remeasure/report.json" "${args[@]}" \
    --out "$RES/dev_analysis.json"
}

case "${1:-all}" in
  train)
    mkdir -p "$RES"
    for s in 0 1 2; do train_one jsrl "$s"; done
    for s in 0 1 2; do train_one noguide "$s"; done ;;
  eval)
    for d in "$RES"/jsrl_s? "$RES"/noguide_s?; do
      [[ -d $d ]] || { echo "missing $d - train first" >&2; exit 1; }
      eval_one "$d"
    done ;;
  analyze) analyze ;;
  all)
    "$0" train; "$0" eval; "$0" analyze ;;
  *)
    echo "usage: op2_pipeline.sh all|train|eval|analyze" >&2; exit 2 ;;
esac
