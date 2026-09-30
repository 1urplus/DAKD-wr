#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

: "${GPU:?Set GPU to the physical GPU index}"
: "${JOBS:?Set comma-separated jobs as method:student:seed}"

if [[ -n "${WAIT_SESSION:-}" ]]; then
  echo "Waiting for tmux session ${WAIT_SESSION} before starting KL multi-seed jobs on GPU ${GPU}."
  while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
    sleep 30
  done
fi

run_ours_kl() {
  local student_model="$1"
  local seed="$2"

  env \
    PYTHON=/data/miniconda3/envs/clip/bin/python \
    NVRTC_LIB_DIR=/tmp/dckd_clip_no_nvrtc_override \
    NVRTC_TARGET_LIB_DIR=/tmp/dckd_clip_no_nvrtc_override \
    DATA_ROOT=/data2/dingyu/datasets/tiny-imagenet-200 \
    DATASET=tinyimagenet \
    GPU="${GPU}" \
    T_MODEL=resnet32x4 \
    ARCH="${student_model}" \
    TEACHER_LIST="tinyA0 tinyA1 tinyA2 tinyA3" \
    DISTILLER=kl \
    EPOCHS=240 \
    TRAIN_PASSES=1 \
    MODEL_EMA_DECAY=0.9999 \
    SEED="${seed}" \
    BATCH_SIZE=64 \
    VAL_SIZE=0 \
    AUG_AGENT_LR=0.0001 \
    AUG_ENTROPY_COEF=0.001 \
    AUG_BASELINE_MOMENTUM=0.9 \
    LATE_TEST_START=210 \
    INIT_LR=0.05 \
    LR_DECAY=0.1 \
    LR_TYPE=cosine \
    MILESTONES_OVERRIDE="150 180 210" \
    CE_WEIGHT=1 \
    KD_WEIGHT=1 \
    FEAT_WEIGHT=3 \
    LOSS_ANNEAL_START=-1 \
    KD_WEIGHT_END=1 \
    FEAT_WEIGHT_END=3 \
    KD_T=4 \
    WEIGHT_DECAY=0.0001 \
    RUN_LABEL="tinyimagenet-resnet32x4-${student_model}-kl-cosine-multiseed" \
    RUN_ID="${SUITE_RUN_ID}" \
    FOREGROUND=1 \
    ./run_ours.sh
}

run_avg_kl() {
  local student_model="$1"
  local seed="$2"

  env \
    PYTHON=/data/miniconda3/envs/clip/bin/python \
    DATA_ROOT=/data2/dingyu/datasets/tiny-imagenet-200 \
    DATASET=tinyimagenet \
    GPU="${GPU}" \
    T_MODEL=resnet32x4 \
    ARCH="${student_model}" \
    TEACHER_LIST="tinyA0 tinyA1 tinyA2 tinyA3" \
    DISTILLER=kl \
    EPOCHS=240 \
    SEED="${seed}" \
    BATCH_SIZE=64 \
    INIT_LR=0.05 \
    LR_DECAY=0.01 \
    LR_TYPE=multistep \
    MILESTONES_OVERRIDE="150 180 210" \
    CE_WEIGHT=1 \
    KD_WEIGHT=1 \
    FEAT_WEIGHT=5 \
    KD_T=4 \
    WEIGHT_DECAY=0.0001 \
    MOMENTUM=0.9 \
    RUN_LABEL="tinyimagenet-resnet32x4-${student_model}-kl-avg-original-multiseed" \
    RUN_ID="${SUITE_RUN_ID}" \
    FOREGROUND=1 \
    ./run_avg.sh
}

SUITE_RUN_ID="${SUITE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
IFS=',' read -r -a JOB_LIST <<< "${JOBS}"

for job in "${JOB_LIST[@]}"; do
  IFS=':' read -r method student_model seed <<< "${job}"
  case "${student_model}" in
    MobileNetV2|ShuffleV2) ;;
    *)
      echo "Unsupported student model: ${student_model}" >&2
      exit 2
      ;;
  esac
  case "${seed}" in
    1|3407) ;;
    *)
      echo "Unsupported multi-seed value: ${seed}" >&2
      exit 2
      ;;
  esac
  case "${method}" in
    ours) run_ours_kl "${student_model}" "${seed}" ;;
    avg) run_avg_kl "${student_model}" "${seed}" ;;
    *)
      echo "Unsupported method: ${method}" >&2
      exit 2
      ;;
  esac
done
