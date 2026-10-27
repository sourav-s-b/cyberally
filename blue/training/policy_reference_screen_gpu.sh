#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "${ROOT}"
PY="/home/sourav/projects/cyberally/.venv-train-cuda/bin/python"
BASE="${ROOT}/blue/results/policy_reference_kl_v2_gpu"
EVAL_SEEDS=(7770 7771 7772 7773 7774 7775 7776 7777)

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
      --arch fixed --out "${model_dir}" --seed "${rng}" --device cuda \
      --train-seeds "${train_seeds[@]}" --eval-seeds "${EVAL_SEEDS[@]}" \
      --iters 6 --eps-per-iter 4 --steps 400 --epochs 4 \
      --critic-epochs 4 --minibatch 256 --threads 1 \
      --lr 0.003 --critic-lr 0.0003 --bonus 1.0 --temp 0.5 \
      --gamma 0.99 --gae-lambda 1.0 --explore 0.5 --clip 0.2 \
      --entropy-coef 0.0 --target-kl 0.02 \
      --reference-kl-coef "${coef}" --experiment-tag \
      "blue-reference-kl-policy-screen-20261025-${arm}-cuda-v2" --resume
  else
    "${PY}" -m blue.training.architecture_compare train \
      --arch fixed --out "${model_dir}" --seed "${rng}" --device cuda \
      --train-seeds "${train_seeds[@]}" --eval-seeds "${EVAL_SEEDS[@]}" \
      --iters 6 --eps-per-iter 4 --steps 400 --epochs 4 \
      --critic-epochs 4 --minibatch 256 --threads 1 \
      --lr 0.003 --critic-lr 0.0003 --bonus 1.0 --temp 0.5 \
      --gamma 0.99 --gae-lambda 1.0 --explore 0.5 --clip 0.2 \
      --entropy-coef 0.0 --target-kl 0.02 \
      --reference-kl-coef "${coef}" --experiment-tag \
      "blue-reference-kl-policy-screen-20261025-${arm}-cuda-v2"
  fi

  if [[ ! -f "${eval_json}" ]]; then
    "${PY}" -m blue.training.architecture_compare eval \
      --arch fixed --out "${eval_json}" --model-dir "${model_dir}" \
      --seed "${rng}" --device cuda --train-seeds "${train_seeds[@]}" \
      --eval-seeds "${EVAL_SEEDS[@]}" --steps 400 --threads 1
  fi
}

run_cell ppo 0 0.0 7754 7755 7756 7757 7758 7759 7760 7761
run_cell ppo 1 0.0 7762 7763 7764 7765 7766 7767 7768 7769
run_cell reference 0 0.1 7754 7755 7756 7757 7758 7759 7760 7761
run_cell reference 1 0.1 7762 7763 7764 7765 7766 7767 7768 7769

"${PY}" -m blue.training.policy_reference_analysis \
  --base "${BASE}" --eval-seeds "${EVAL_SEEDS[@]}"
