#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "${ROOT}"
PY="${ROOT}/../../.venv-train/bin/python"
BASE="${ROOT}/blue/results/policy_reference_kl_v1"
EVAL_SEEDS=(7746 7747 7748 7749 7750 7751 7752 7753)

run_cell() {
  local arm="$1" rng="$2" coef="$3"
  shift 3
  local train_seeds=("$@")
  local model_dir="${BASE}/models/${arm}_s${rng}"
  local eval_json="${BASE}/evaluations/${arm}_s${rng}.json"
  mkdir -p "${BASE}/models" "${BASE}/evaluations"

  local resume_flag=""
  if [[ -f "${model_dir}/manifest.json" && -f "${model_dir}/checkpoint.pt" ]]; then
    resume_flag="--resume"
  fi
  if [[ -n "${resume_flag}" ]]; then
    "${PY}" -m blue.training.architecture_compare train \
      --arch fixed --out "${model_dir}" --seed "${rng}" \
      --train-seeds "${train_seeds[@]}" --eval-seeds "${EVAL_SEEDS[@]}" \
      --iters 6 --eps-per-iter 4 --steps 400 --epochs 4 \
      --critic-epochs 4 --minibatch 256 --threads 1 \
      --lr 0.003 --critic-lr 0.0003 --bonus 1.0 --temp 0.5 \
      --gamma 0.99 --gae-lambda 1.0 --explore 0.5 --clip 0.2 \
      --entropy-coef 0.0 --target-kl 0.02 \
      --reference-kl-coef "${coef}" --experiment-tag \
      "blue-reference-kl-policy-screen-20261025-${arm}" --resume
  else
    "${PY}" -m blue.training.architecture_compare train \
    --arch fixed --out "${model_dir}" --seed "${rng}" \
    --train-seeds "${train_seeds[@]}" --eval-seeds "${EVAL_SEEDS[@]}" \
    --iters 6 --eps-per-iter 4 --steps 400 --epochs 4 \
    --critic-epochs 4 --minibatch 256 --threads 1 \
    --lr 0.003 --critic-lr 0.0003 --bonus 1.0 --temp 0.5 \
    --gamma 0.99 --gae-lambda 1.0 --explore 0.5 --clip 0.2 \
    --entropy-coef 0.0 --target-kl 0.02 \
    --reference-kl-coef "${coef}" --experiment-tag \
    "blue-reference-kl-policy-screen-20261025-${arm}"
  fi

  if [[ ! -f "${eval_json}" ]]; then
    "${PY}" -m blue.training.architecture_compare eval \
      --arch fixed --out "${eval_json}" --model-dir "${model_dir}" \
      --seed "${rng}" --train-seeds "${train_seeds[@]}" \
      --eval-seeds "${EVAL_SEEDS[@]}" --steps 400 --threads 1 \
      --reference-kl-coef "${coef}" --experiment-tag \
      "blue-reference-kl-policy-screen-20261025-${arm}"
  fi
}

run_cell ppo 0 0.0 7730 7731 7732 7733 7734 7735 7736 7737
run_cell ppo 1 0.0 7738 7739 7740 7741 7742 7743 7744 7745
run_cell reference 0 0.1 7730 7731 7732 7733 7734 7735 7736 7737
run_cell reference 1 0.1 7738 7739 7740 7741 7742 7743 7744 7745

"${PY}" -m blue.training.policy_reference_analysis \
  --base "${BASE}" --eval-seeds "${EVAL_SEEDS[@]}"
