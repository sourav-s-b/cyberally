#!/usr/bin/env bash
# State-based shaping experiment: train and score under both attackers.
#
# THE QUESTION
#   Can a state-based training reward teach the harness an investigation
#   order that beats Lancer on NATIVE return?
#
#   The October 2026 shaping attempts scored actions (+1 per Analyse that
#   found something) and degraded native return at both learning rates: a
#   dense per-action term mixed with a sparse true signal biases the gradient
#   toward farming the term. These terms score state instead, which cannot be
#   farmed by acting more. See blue/harness/state_reward.py.
#
# THE MATRIX  (only state shaping differs between paired arms)
#   discovery Red : unshaped = main32 both_s{0,1,2}   (already exist)
#                   shaped   = 3 new runs
#   finite Red    : unshaped = 3 new runs
#                   shaped   = 3 new runs
#   => 9 new training models. The discovery/unshaped arm is REUSED, not
#      retrained, so the comparison sits at main32's exact 32-iteration
#      budget instead of a cheaper budget invented for this run. Comparing a
#      32-iteration control against a shorter treatment would confound
#      shaping with undertraining.
#
#   Evaluation applies NO shaping (harness_eval scores native only; pinned by
#   blue/tests_blue/test_harness_state_reward.py). Both arms are read out
#   under BOTH attackers, because the discovery regime cannot punish a broken
#   sweep and is not a sufficient discriminator on its own.
#
# SEEDS
#   train  7706-7709  frozen to match main32. Already spent; that is required
#                    for the match and they are never reported on.
#   eval   7665-7688  fresh, verified free by blue.training.seed_audit with
#                    the vendored-tree filter applied. Disjoint from
#                    6016-6039 (spent by the main-line guarded retrain).
#   7809-8200 stays reserved and is refused in code.
#
# USAGE
#   scripts/blue_state_shaping_run.sh plan
#   scripts/blue_state_shaping_run.sh train <shards> <shard-index>
#   scripts/blue_state_shaping_run.sh eval  <shards> <shard-index> [arm ...]
#
# Runs are detached with setsid; closing the terminal is safe. Each model
# writes its own log next to its output directory, and harness_rl refuses to
# overwrite an existing run directory, so re-running skips completed models.
set -euo pipefail

REPO=/home/sourav/projects/cyberally
WT=/home/sourav/projects/cyberally/worktrees/gpt-blue
PY="$REPO/.venv-train/bin/python"
SCORER="$REPO/blue/results/gpt_harness/scorer.pkl"
MAIN32="$REPO/blue/results/gpt_harness/main32_status"
OUT="$REPO/blue/results/state_shaping"

# Frozen to the main32 manifest (blue/results/gpt_harness/main32_status/
# s0/pilot/both_s0/manifest.json) so the shaped arm is comparable to the
# checkpoints it is judged against.
ITERS=32
EPS_PER_ITER=4
STEPS=400
LR=3e-4
BONUS=0.25
TEMP=0.5
MAX_AGE=80
ML_INPUTS=both
TRAIN_SEEDS="7706 7707 7708 7709"

EVAL_SEEDS=$(seq 7665 7688)
ATTACKERS="discovery finite"
ARMS="unshaped shaped"
TRAIN_SEED_LIST="0 1 2"

# ---------------------------------------------------------------- matrix ----

# Echo one line per model. REUSE lines are existing main32 checkpoints.
emit_models() {
  local a arm s
  for a in $ATTACKERS; do
    for arm in $ARMS; do
      for s in $TRAIN_SEED_LIST; do
        if [ "$a" = discovery ] && [ "$arm" = unshaped ]; then
          echo "REUSE $a $arm $s"
        else
          echo "TRAIN $a $arm $s"
        fi
      done
    done
  done
}

# Directory holding a trained model, whether reused or newly trained.
model_dir() {  # attacker arm train_seed
  if [ "$1" = discovery ] && [ "$2" = unshaped ]; then
    echo "$MAIN32/s$3/pilot/both_s$3"
  else
    echo "$OUT/$1/$2/s$3"
  fi
}

# ------------------------------------------------------------------ plan ----

do_plan() {
  local new=0 reused=0
  while read -r kind _; do
    case "$kind" in
      TRAIN) new=$((new + 1)) ;;
      REUSE) reused=$((reused + 1)) ;;
    esac
  done < <(emit_models)
  local per=$((ITERS * EPS_PER_ITER))
  local seeds; seeds=$(wc -w <<<"$EVAL_SEEDS")
  local nmodels=$((new + reused))
  cat <<EOF
train
  new models        : $new
  reused main32     : $reused
  episodes per model: $per (${ITERS} iters x ${EPS_PER_ITER} eps, ${STEPS} steps)
  rough cost        : ~25s/episode on this CPU
                     sequential ~$((new * per * 25 / 60)) min
                     3 shards       ~$((new * per * 25 / 60 / 3)) min
eval
  readouts          : $nmodels models, each paired against a shared Lancer
                      baseline, under BOTH attackers
  episodes          : $(( nmodels * seeds + 2 * seeds ))   # actors + 2 Lancer runs
                     (Lancer is computed once per attacker and cached, not
                      once per model)
  fresh eval seeds  : 7665-7688 ($seeds, disjoint from 6016-6039)
  reserved 7809-8200: untouched
EOF
}

# ----------------------------------------------------------------- train ----

# Start one training job. Returns immediately; the caller throttles with
# `throttle`, which keeps at most $WORKERS jobs alive. An earlier version
# launched every assigned model detached and immediately, so "3 shards"
# actually meant 9 concurrent jobs and the time estimate was fiction.
train_one() {  # attacker arm train_seed
  local attacker=$1 arm=$2 tseed=$3
  local dir; dir=$(model_dir "$attacker" "$arm" "$tseed")
  if [ -f "$dir/actor.th" ]; then
    echo "  skip (exists): $(basename "$dir")"
    return 0
  fi
  mkdir -p "$(dirname "$dir")"
  local shaping=off
  [ "$arm" = shaped ] && shaping=both
  echo "  launch: attacker=$attacker arm=$arm seed=$tseed shaping=$shaping"
  # Run Python in the FOREGROUND of this subshell. An earlier version used
  # `nohup ... &` *inside* the pooled job, so the job the pool tracked exited
  # immediately while python kept running detached -- throttle() counted
  # nothing and every model started at once. `setsid` alone still detaches
  # from the controlling terminal (new session, so closing the shell does not
  # SIGHUP it) while keeping the job alive for throttle() to count.
  # shellcheck disable=SC2086
  ( cd "$WT" && exec setsid env PYTHONPATH="$WT" \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$PY" -m blue.training.harness_rl \
        --out "$dir" --seed "$tseed" \
        --iters "$ITERS" --eps-per-iter "$EPS_PER_ITER" --steps "$STEPS" \
        --hidden 64 --lr "$LR" --bonus "$BONUS" --temp "$TEMP" \
        --max-age "$MAX_AGE" --ml-inputs "$ML_INPUTS" \
        --train-seeds $TRAIN_SEEDS \
        --red-agent "$attacker" \
        --state-shaping "$shaping" \
        --state-shaping-weight 0.05 --state-shaping-cap 20.0 \
        --scorer "$SCORER" \
        > "$dir.log" 2>&1 < /dev/null ) &
}

# Keep at most $1 detached training jobs alive at once.
throttle() {
  local limit=$1
  while [ "$(jobs -rp | wc -l)" -ge "$limit" ]; do
    wait -n 2>/dev/null || sleep 5
  done
}

do_train() {  # workers
  local workers=${1:-3}
  [ "$workers" -ge 1 ] || { echo "workers must be >= 1" >&2; exit 2; }
  echo "training with at most $workers concurrent job(s)"
  while read -r kind a arm s; do
    [ "$kind" = TRAIN ] || continue
    throttle "$workers"
    train_one "$a" "$arm" "$s" &
  done < <(emit_models)
  echo "waiting for training jobs to finish..."
  wait
  echo "training complete"
}

# ------------------------------------------------------------------ eval ----

eval_one() {  # readout_attacker arm train_seed
  local attacker=$1 arm=$2 tseed=$3
  local dir; dir=$(model_dir "$attacker" "$arm" "$tseed")
  if [ ! -f "$dir/actor.th" ]; then
    echo "  MISSING (not trained yet): $dir"; return 0
  fi
  local tag="$OUT/eval/$attacker/$arm/s$tseed"
  # The shell opens "$tag.log" BEFORE harness_eval can mkdir its own --out,
  # so the log's parent must exist here or the redirect fails outright.
  mkdir -p "$OUT/eval/$attacker/$arm"
  if [ -f "$tag/report.json" ]; then
    echo "  skip (exists): $attacker/$arm/s$tseed"; return 0
  fi
  echo "  launch: readout=$attacker arm=$arm seed=$tseed"
  # Foreground python in a setsid session; see train_one for why the trailing
  # `&` must live here (the pool) and not inside (the job).
  # shellcheck disable=SC2086
  ( cd "$WT" && exec setsid env PYTHONPATH="$WT" \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$PY" -m blue.training.harness_eval \
        --model-dir "$dir" --arms actor lancer \
        --seeds $EVAL_SEEDS --steps "$STEPS" --max-age "$MAX_AGE" \
        --red-agent "$attacker" --scorer "$SCORER" \
        --lancer-cache "$OUT/eval/$attacker/lancer.json" \
        --read-only-lancer-cache \
        --out "$tag" \
        > "$tag.log" 2>&1 < /dev/null ) &
}

do_eval() {  # workers
  local workers=${1:-3}
  # Baselines are precomputed, never written concurrently. See lancer_base.
  for a in $ATTACKERS; do
    if [ ! -f "$OUT/eval/$a/lancer.json" ]; then
      echo "ERROR: no precomputed Lancer baseline for $a."
      echo "       run:  $0 lancer $a"
      return 1
    fi
  done
  echo "evaluating with at most $workers concurrent job(s)"
  while read -r _ a arm s; do
    throttle "$workers"
    eval_one "$a" "$arm" "$s" &
  done < <(emit_models)
  echo "waiting for evaluation jobs to finish..."
  wait
  echo "evaluation complete"
}

# Precompute one Lancer baseline per attacker. Must run BEFORE `eval`, and
# one attacker at a time: concurrent writers to a shared cache file can
# interleave and produce a truncated or mixed baseline, which would then be
# silently paired against every model.
lancer_base() {  # attacker
  local attacker=$1
  local cache="$OUT/eval/$attacker/lancer.json"
  if [ -f "$cache" ]; then
    echo "  skip (exists): $attacker baseline"; return 0
  fi
  mkdir -p "$OUT/eval/$attacker"
  echo "  lancer baseline: attacker=$attacker ($(wc -w <<<"$EVAL_SEEDS") episodes)"
  ( cd "$WT" && env PYTHONPATH="$WT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$PY" -m blue.training.harness_eval \
        --arms lancer --seeds $EVAL_SEEDS --steps "$STEPS" \
        --max-age "$MAX_AGE" --red-agent "$attacker" --scorer "$SCORER" \
        --lancer-cache "$cache" --out "$OUT/eval/$attacker/lancer" )
}

# ------------------------------------------------------------------ main ----

do_pilot() {
  local attacker=${1:-finite}
  mkdir -p "$OUT"
  ( cd "$WT" && PYTHONPATH="$WT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$PY" -m blue.training.harness_shape_pilot \
        --seeds 7706 7707 --red-agent "$attacker" --state-shaping both \
        --out "$OUT/pilot_shape_$attacker.json" )
}

cmd=${1:-plan}
case "$cmd" in
  plan)   do_plan ;;
  pilot)  do_pilot "${2:-finite}" ;;
  train)  mkdir -p "$OUT"; do_train "${2:-3}" ;;
  lancer) mkdir -p "$OUT/eval"; lancer_base "${2:?attacker required}" ;;
  eval)   mkdir -p "$OUT/eval"; do_eval "${2:-3}" ;;
  models) emit_models ;;
  *) echo "usage: $0 {plan|pilot|train <workers>|lancer <attacker>|eval <workers>}" >&2
     exit 2 ;;
esac