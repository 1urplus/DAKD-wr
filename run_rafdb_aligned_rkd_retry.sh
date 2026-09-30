#!/usr/bin/env bash
# Retry one failed RAF-DB AVG-aligned OURS-RKD architecture.
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 PHYSICAL_GPU STUDENT_MODEL" >&2
  exit 2
fi

gpu="$1"
student="$2"
case "$student" in
  MobileNetV2|ShuffleV2) ;;
  *) echo "unsupported student: $student" >&2; exit 2 ;;
esac

python_bin="${PYTHON_BIN:-/data/miniconda3/envs/clip/bin/python}"
data_root="${RAFDB_DATA_ROOT:-/data2/xujianyang/rafdb_3modals}"

env PYTHON="$python_bin" GPU="$gpu" DATASET=rafdb DATA_ROOT="$data_root" \
  T_MODEL=resnet32x4 ARCH="$student" \
  TEACHER_LIST="rafA0 rafA1 rafA2 rafA3" \
  DISTILLER=rkd EPOCHS=240 BATCH_SIZE=64 SEED=42 \
  INIT_LR=0.05 LR_DECAY=0.01 LR_TYPE=multistep \
  MILESTONES_OVERRIDE="150 180 210" \
  CE_WEIGHT=1 KD_WEIGHT=1 FEAT_WEIGHT=5 FEAT_WEIGHT_END=5 \
  SP_WEIGHT=1 TRAIN_PASSES=1 NORMALIZE_FIRST_EPOCH_ACTIONS=1 \
  MODEL_EMA_DECAY=0 VAL_SIZE=0 LATE_TEST_START=210 \
  WEIGHT_DECAY=0.0001 RUN_LABEL=rafdb-avg-aligned-rkd-retry-noema \
  FOREGROUND=1 ./run_ours.sh
