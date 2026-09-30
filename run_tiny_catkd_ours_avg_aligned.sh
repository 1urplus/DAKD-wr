#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

: "${GPU:?Set GPU to the physical GPU index}"
: "${ARCH:?Set ARCH to ShuffleV2 or MobileNetV2}"

case "${ARCH}" in
  ShuffleV2|MobileNetV2) ;;
  *) echo "Unsupported student: ${ARCH}" >&2; exit 2 ;;
esac

if [[ -n "${WAIT_SESSION:-}" ]]; then
  echo "Waiting for tmux session ${WAIT_SESSION} before starting aligned CATKD on GPU ${GPU}."
  while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
    sleep 30
  done
fi

env \
  PYTHON=/data/miniconda3/envs/clip/bin/python \
  NVRTC_LIB_DIR=/tmp/dckd_clip_no_nvrtc_override \
  NVRTC_TARGET_LIB_DIR=/tmp/dckd_clip_no_nvrtc_override \
  DATA_ROOT=/data2/dingyu/datasets/tiny-imagenet-200 \
  DATASET=tinyimagenet \
  GPU="${GPU}" \
  T_MODEL=resnet32x4 \
  ARCH="${ARCH}" \
  TEACHER_LIST="tinyA0 tinyA1 tinyA2 tinyA3" \
  DISTILLER=catkd \
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
  LR_DECAY=0.01 \
  LR_TYPE=multistep \
  MILESTONES_OVERRIDE="150 180 210" \
  CE_WEIGHT=1 \
  KD_WEIGHT=1 \
  FEAT_WEIGHT=5 \
  LOSS_ANNEAL_START=-1 \
  KD_WEIGHT_END=1 \
  FEAT_WEIGHT_END=5 \
  KD_T=4 \
  CATKD_T=4 \
  CATKD_ALPHA=1 \
  CATKD_BETA=1 \
  CATKD_GAMMA=1 \
  WEIGHT_DECAY=0.0001 \
  RUN_LABEL="tinyimagenet-resnet32x4-${ARCH}-catkd-ours-avg-aligned" \
  RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}" \
  FOREGROUND=1 \
  ./run_ours.sh
