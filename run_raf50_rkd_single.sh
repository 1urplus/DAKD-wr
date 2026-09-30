#!/usr/bin/env bash
# Run one RAF-DB ResNet50 -> ResNet32 RKD method on a selected GPU.
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 {ours|avg} PHYSICAL_GPU" >&2
  exit 2
fi

method="$1"
gpu="$2"
python_bin="${PYTHON_BIN:-/data/miniconda3/envs/clip/bin/python}"
data_root="${RAFDB_DATA_ROOT:-/data2/xujianyang/rafdb_3modals}"
teachers="raf50A0 raf50A1 raf50A2 raf50A3"

if [[ "$method" == ours ]]; then
  env PYTHON="$python_bin" GPU="$gpu" DATASET=rafdb DATA_ROOT="$data_root" \
    T_MODEL=ResNet50 ARCH=resnet32 TEACHER_LIST="$teachers" \
    DISTILLER=rkd EPOCHS=240 BATCH_SIZE=64 SEED=42 \
    INIT_LR=0.05 LR_DECAY=0.01 LR_TYPE=multistep \
    MILESTONES_OVERRIDE="150 180 210" \
    CE_WEIGHT=1 KD_WEIGHT=1 FEAT_WEIGHT=5 FEAT_WEIGHT_END=5 \
    SP_WEIGHT=1 TRAIN_PASSES=1 NORMALIZE_FIRST_EPOCH_ACTIONS=1 \
    MODEL_EMA_DECAY=0 VAL_SIZE=0 LATE_TEST_START=210 \
    WEIGHT_DECAY=0.0001 RUN_LABEL="rafdb-ResNet50-resnet32-rkd-retry-gpu${gpu}-noema" \
    FOREGROUND=1 ./run_ours.sh
elif [[ "$method" == avg ]]; then
  env PYTHON="$python_bin" GPU="$gpu" DATASET=rafdb DATA_ROOT="$data_root" \
    T_MODEL=ResNet50 ARCH=resnet32 TEACHER_LIST="$teachers" \
    DISTILLER=rkd EPOCHS=240 BATCH_SIZE=64 SEED=42 \
    INIT_LR=0.05 LR_DECAY=0.01 LR_TYPE=multistep \
    MILESTONES_OVERRIDE="150 180 210" \
    CE_WEIGHT=1 KD_WEIGHT=1 FEAT_WEIGHT=5 SP_WEIGHT=1 FEAT_KD=none \
    WEIGHT_DECAY=0.0001 LATE_TEST_START=210 \
    RUN_LABEL="rafdb-ResNet50-resnet32-rkd-retry-gpu${gpu}" FOREGROUND=1 ./run_avg.sh
else
  echo "unsupported method: $method" >&2
  exit 2
fi
