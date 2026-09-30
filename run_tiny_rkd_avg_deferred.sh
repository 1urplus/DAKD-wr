#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

: "${GPU:?Set GPU to the physical GPU index}"
: "${ARCH:?Set ARCH to ShuffleV2 or MobileNetV2}"
: "${CURRENT_RKD_SESSION:?Set the current RKD-OURS tmux session}"
: "${PRIORITY_SESSION:?Set the priority queue tmux session}"
SUITE_RUN_ID="${SUITE_RUN_ID:-20260914_122516}"

case "${ARCH}" in
  ShuffleV2|MobileNetV2) ;;
  *) echo "Unsupported student: ${ARCH}" >&2; exit 2 ;;
esac

echo "Deferring RKD-AVG ${ARCH} until ${CURRENT_RKD_SESSION} and ${PRIORITY_SESSION} finish."
while tmux has-session -t "${CURRENT_RKD_SESSION}" 2>/dev/null || \
      tmux has-session -t "${PRIORITY_SESSION}" 2>/dev/null; do
  sleep 60
done

exec env \
  PYTHON=/data/miniconda3/envs/clip/bin/python \
  DATA_ROOT=/data2/dingyu/datasets/tiny-imagenet-200 \
  DATASET=tinyimagenet \
  GPU="${GPU}" \
  T_MODEL=resnet32x4 \
  ARCH="${ARCH}" \
  TEACHER_LIST="tinyA0 tinyA1 tinyA2 tinyA3" \
  DISTILLER=rkd \
  EPOCHS=240 \
  SEED=42 \
  BATCH_SIZE=64 \
  MILESTONES_OVERRIDE="150 180 210" \
  CE_WEIGHT=1 \
  KD_WEIGHT=1 \
  KD_T=4 \
  RKD_DIST_W=25 \
  RKD_ANGLE_W=50 \
  WEIGHT_DECAY=0.0001 \
  RUN_ID="${SUITE_RUN_ID}" \
  FOREGROUND=1 \
  IGNORE_QUEUE_HOLD=1 \
  INIT_LR=0.05 \
  LR_DECAY=0.01 \
  LR_TYPE=multistep \
  FEAT_WEIGHT=5 \
  MOMENTUM=0.9 \
  RUN_LABEL="tinyimagenet-resnet32x4-${ARCH}-rkd-avg-original" \
  ./run_avg.sh
