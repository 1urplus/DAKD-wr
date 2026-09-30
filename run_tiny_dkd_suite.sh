#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

: "${GPU:?Set GPU to the physical GPU index}"
: "${JOB_GROUP:?Set JOB_GROUP to vgg_shuffle, wrn, or mobile}"

if [[ -n "${WAIT_SESSION:-}" ]]; then
  echo "Waiting for tmux session ${WAIT_SESSION} before starting DKD jobs on GPU ${GPU}."
  while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
    sleep 30
  done
fi

run_ours_dkd() {
  local teacher_model="$1"
  local student_model="$2"
  local teacher_list="$3"
  local run_id="$4"

  env \
    PYTHON=/data/miniconda3/envs/clip/bin/python \
    NVRTC_LIB_DIR=/tmp/dckd_clip_no_nvrtc_override \
    NVRTC_TARGET_LIB_DIR=/tmp/dckd_clip_no_nvrtc_override \
    DATA_ROOT=/data2/dingyu/datasets/tiny-imagenet-200 \
    DATASET=tinyimagenet \
    GPU="${GPU}" \
    T_MODEL="${teacher_model}" \
    ARCH="${student_model}" \
    TEACHER_LIST="${teacher_list}" \
    DISTILLER=dkd \
    EPOCHS=240 \
    TRAIN_PASSES=1 \
    MODEL_EMA_DECAY=0.9999 \
    SEED=42 \
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
    DKD_ALPHA=1.0 \
    DKD_BETA=2.0 \
    WEIGHT_DECAY=0.0001 \
    RUN_LABEL="tinyimagenet-${teacher_model}-${student_model}-dkd-cosine" \
    RUN_ID="${run_id}" \
    FOREGROUND=1 \
    ./run_ours.sh
}

run_avg_dkd() {
  local teacher_model="$1"
  local student_model="$2"
  local teacher_list="$3"
  local run_id="$4"

  env \
    PYTHON=/data/miniconda3/envs/clip/bin/python \
    DATA_ROOT=/data2/dingyu/datasets/tiny-imagenet-200 \
    DATASET=tinyimagenet \
    GPU="${GPU}" \
    T_MODEL="${teacher_model}" \
    ARCH="${student_model}" \
    TEACHER_LIST="${teacher_list}" \
    DISTILLER=dkd \
    EPOCHS=240 \
    SEED=42 \
    BATCH_SIZE=64 \
    INIT_LR=0.05 \
    LR_DECAY=0.01 \
    LR_TYPE=multistep \
    MILESTONES_OVERRIDE="150 180 210" \
    CE_WEIGHT=1 \
    KD_WEIGHT=1 \
    FEAT_WEIGHT=5 \
    KD_T=4 \
    DKD_ALPHA=1.0 \
    DKD_BETA=2.0 \
    WEIGHT_DECAY=0.0001 \
    MOMENTUM=0.9 \
    RUN_LABEL="tinyimagenet-${teacher_model}-${student_model}-dkd-avg-original" \
    RUN_ID="${run_id}" \
    FOREGROUND=1 \
    ./run_avg.sh
}

run_pair() {
  local teacher_model="$1"
  local student_model="$2"
  local teacher_list="$3"
  local run_id="$4"

  run_ours_dkd "${teacher_model}" "${student_model}" "${teacher_list}" "${run_id}"
  run_avg_dkd "${teacher_model}" "${student_model}" "${teacher_list}" "${run_id}"
}

SUITE_RUN_ID="${SUITE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

case "${JOB_GROUP}" in
  vgg_shuffle)
    run_pair vgg13 vgg8 \
      "tinyvggA0 tinyvggA1 tinyvggA2 tinyvggA3" \
      "${SUITE_RUN_ID}"
    run_pair resnet32x4 ShuffleV2 \
      "tinyA0 tinyA1 tinyA2 tinyA3" \
      "${SUITE_RUN_ID}"
    ;;
  wrn)
    run_pair resnet32x4 wrn_16_2 \
      "tinyA0 tinyA1 tinyA2 tinyA3" \
      "${SUITE_RUN_ID}"
    ;;
  mobile)
    run_pair resnet32x4 MobileNetV2 \
      "tinyA0 tinyA1 tinyA2 tinyA3" \
      "${SUITE_RUN_ID}"
    ;;
  *)
    echo "Unknown JOB_GROUP: ${JOB_GROUP}" >&2
    exit 2
    ;;
esac
