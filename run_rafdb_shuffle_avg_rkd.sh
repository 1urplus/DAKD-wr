#!/usr/bin/env bash
# Missing RAF-DB ResNet32x4 -> ShuffleV2 AVG-RKD baseline.
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 PHYSICAL_GPU" >&2
  exit 2
fi

gpu="$1"
python_bin="${PYTHON_BIN:-/data/miniconda3/envs/clip/bin/python}"
data_root="${RAFDB_DATA_ROOT:-/data2/xujianyang/rafdb_3modals}"

env PYTHON="$python_bin" GPU="$gpu" DATASET=rafdb DATA_ROOT="$data_root" \
  T_MODEL=resnet32x4 ARCH=ShuffleV2 \
  TEACHER_LIST="rafA0 rafA1 rafA2 rafA3" \
  DISTILLER=rkd EPOCHS=240 BATCH_SIZE=64 SEED=42 \
  INIT_LR=0.05 LR_DECAY=0.01 LR_TYPE=multistep \
  MILESTONES_OVERRIDE="150 180 210" \
  CE_WEIGHT=1 KD_WEIGHT=1 FEAT_WEIGHT=5 SP_WEIGHT=1 FEAT_KD=none \
  WEIGHT_DECAY=0.0001 LATE_TEST_START=210 \
  RUN_LABEL=rafdb-standard-rkd-missing FOREGROUND=1 ./run_avg.sh
