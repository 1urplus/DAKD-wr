#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

: "${GPU:?Set GPU to the physical GPU index}"
: "${ARCH:?Set ARCH to ShuffleV2 or MobileNetV2}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

case "${ARCH}" in
  ShuffleV2|MobileNetV2) ;;
  *) echo "Unsupported student: ${ARCH}" >&2; exit 2 ;;
esac

exec env \
  PYTHON=/data/miniconda3/envs/clip/bin/python \
  DATA_ROOT=/data2/dingyu/datasets/tiny-imagenet-200 \
  DATASET=tinyimagenet \
  GPU="${GPU}" \
  T_MODEL=resnet32x4 \
  ARCH="${ARCH}" \
  TEACHER_LIST="tinyA0 tinyA1 tinyA2 tinyA3" \
  DISTILLER=sp \
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
  SP_WEIGHT=3000 \
  LOSS_ANNEAL_START=-1 \
  KD_WEIGHT_END=1 \
  FEAT_WEIGHT_END=3 \
  KD_T=4 \
  WEIGHT_DECAY=0.0001 \
  RUN_ID="${RUN_ID}" \
  RUN_LABEL="tinyimagenet-resnet32x4-${ARCH}-sp-optimized-gamma3000-cosine" \
  FOREGROUND=1 \
  NVRTC_LIB_DIR=/tmp/dckd_clip_no_nvrtc_override \
  NVRTC_TARGET_LIB_DIR=/tmp/dckd_clip_no_nvrtc_override \
  ./run_ours.sh
