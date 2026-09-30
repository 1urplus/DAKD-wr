#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

: "${GPU:?Set GPU to the physical GPU index}"
: "${QUEUE:?Set QUEUE to seed1 or seed3407}"
SUITE_RUN_ID="${SUITE_RUN_ID:-$(date +%Y%m%d_%H%M%S)_restart}"

run_avg_dkd() {
  local student_model="$1"

  env \
    PYTHON=/data/miniconda3/envs/clip/bin/python \
    DATA_ROOT=/data2/dingyu/datasets/tiny-imagenet-200 \
    DATASET=tinyimagenet \
    GPU="${GPU}" \
    T_MODEL=resnet32x4 \
    ARCH="${student_model}" \
    TEACHER_LIST="tinyA0 tinyA1 tinyA2 tinyA3" \
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
    RUN_LABEL="tinyimagenet-resnet32x4-${student_model}-dkd-avg-original-restart" \
    RUN_ID="${SUITE_RUN_ID}" \
    FOREGROUND=1 \
    ./run_avg.sh
}

case "${QUEUE}" in
  seed1)
    run_avg_dkd MobileNetV2
    env GPU="${GPU}" \
      JOBS="ours:ShuffleV2:1,avg:ShuffleV2:1,ours:MobileNetV2:1,avg:MobileNetV2:1" \
      SUITE_RUN_ID="${SUITE_RUN_ID}" \
      ./run_tiny_kl_multiseed.sh
    ;;
  seed3407)
    run_avg_dkd ShuffleV2
    env GPU="${GPU}" \
      JOBS="ours:ShuffleV2:3407,avg:ShuffleV2:3407,ours:MobileNetV2:3407,avg:MobileNetV2:3407" \
      SUITE_RUN_ID="${SUITE_RUN_ID}" \
      ./run_tiny_kl_multiseed.sh
    ;;
  *)
    echo "Unknown QUEUE: ${QUEUE}" >&2
    exit 2
    ;;
esac
