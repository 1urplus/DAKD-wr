#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

: "${GPU:?Set GPU to the physical GPU index}"
: "${JOB_GROUP:?Set JOB_GROUP to vgg_wrn or mobile_shuffle}"

if [[ -n "${WAIT_SESSION:-}" ]]; then
  echo "Waiting for tmux session ${WAIT_SESSION} before starting AVG jobs on GPU ${GPU}."
  while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
    sleep 30
  done
fi

run_avg_job() {
  local teacher_model="$1"
  local student_model="$2"
  local teacher_list="$3"
  local run_label="$4"
  local run_id="$5"

  env \
    PYTHON=/data/miniconda3/envs/clip/bin/python \
    DATA_ROOT=/data2/dingyu/datasets/tiny-imagenet-200 \
    DATASET=tinyimagenet \
    GPU="${GPU}" \
    T_MODEL="${teacher_model}" \
    ARCH="${student_model}" \
    TEACHER_LIST="${teacher_list}" \
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
    WEIGHT_DECAY=0.0001 \
    MOMENTUM=0.9 \
    RUN_LABEL="${run_label}" \
    RUN_ID="${run_id}" \
    FOREGROUND=1 \
    ./run_avg.sh
}

SUITE_RUN_ID="${SUITE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

case "${JOB_GROUP}" in
  vgg_wrn)
    run_avg_job vgg13 vgg8 \
      "tinyvggA0 tinyvggA1 tinyvggA2 tinyvggA3" \
      tinyimagenet-vgg13-vgg8-avg-original \
      "${SUITE_RUN_ID}"
    run_avg_job resnet32x4 wrn_16_2 \
      "tinyA0 tinyA1 tinyA2 tinyA3" \
      tinyimagenet-resnet32x4-wrn_16_2-avg-original \
      "${SUITE_RUN_ID}"
    ;;
  mobile_shuffle)
    run_avg_job resnet32x4 MobileNetV2 \
      "tinyA0 tinyA1 tinyA2 tinyA3" \
      tinyimagenet-resnet32x4-MobileNetV2-avg-original \
      "${SUITE_RUN_ID}"
    run_avg_job resnet32x4 ShuffleV2 \
      "tinyA0 tinyA1 tinyA2 tinyA3" \
      tinyimagenet-resnet32x4-ShuffleV2-avg-original \
      "${SUITE_RUN_ID}"
    ;;
  *)
    echo "Unknown JOB_GROUP: ${JOB_GROUP}" >&2
    exit 2
    ;;
esac
