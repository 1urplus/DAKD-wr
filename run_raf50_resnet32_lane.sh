#!/usr/bin/env bash
# RAF-DB ResNet50 (A0-A3) -> ResNet32: corrected OURS and original AVG.
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 PHYSICAL_GPU" >&2
  exit 2
fi

gpu="$1"
python_bin="${PYTHON_BIN:-/data/miniconda3/envs/clip/bin/python}"
data_root="${RAFDB_DATA_ROOT:-/data2/xujianyang/rafdb_3modals}"
teachers="raf50A0 raf50A1 raf50A2 raf50A3"

for distiller in kl dkd sp catkd rkd; do
  sp_weight=1
  avg_feat_kd=mse
  if [[ "$distiller" == sp ]]; then
    sp_weight=3000
    avg_feat_kd=none
  elif [[ "$distiller" == rkd ]]; then
    avg_feat_kd=none
  fi

  echo "Starting RAF-DB ResNet50 -> ResNet32: corrected OURS ${distiller} on GPU ${gpu}"
  env PYTHON="$python_bin" GPU="$gpu" DATASET=rafdb DATA_ROOT="$data_root" \
    T_MODEL=ResNet50 ARCH=resnet32 TEACHER_LIST="$teachers" \
    DISTILLER="$distiller" EPOCHS=240 BATCH_SIZE=64 SEED=42 \
    INIT_LR=0.05 LR_DECAY=0.01 LR_TYPE=multistep \
    MILESTONES_OVERRIDE="150 180 210" \
    CE_WEIGHT=1 KD_WEIGHT=1 FEAT_WEIGHT=5 FEAT_WEIGHT_END=5 \
    SP_WEIGHT="$sp_weight" TRAIN_PASSES=1 \
    NORMALIZE_FIRST_EPOCH_ACTIONS=1 \
    MODEL_EMA_DECAY=0 VAL_SIZE=0 LATE_TEST_START=210 \
    WEIGHT_DECAY=0.0001 RUN_LABEL=rafdb-ResNet50-resnet32-avg-aligned-noema-late210 \
    FOREGROUND=1 ./run_ours.sh

  echo "Starting RAF-DB ResNet50 -> ResNet32: AVG ${distiller} on GPU ${gpu}"
  env PYTHON="$python_bin" GPU="$gpu" DATASET=rafdb DATA_ROOT="$data_root" \
    T_MODEL=ResNet50 ARCH=resnet32 TEACHER_LIST="$teachers" \
    DISTILLER="$distiller" EPOCHS=240 BATCH_SIZE=64 SEED=42 \
    INIT_LR=0.05 LR_DECAY=0.01 LR_TYPE=multistep \
    MILESTONES_OVERRIDE="150 180 210" \
    CE_WEIGHT=1 KD_WEIGHT=1 FEAT_WEIGHT=5 SP_WEIGHT="$sp_weight" \
    FEAT_KD="$avg_feat_kd" WEIGHT_DECAY=0.0001 LATE_TEST_START=210 \
    RUN_LABEL=rafdb-ResNet50-resnet32-standard FOREGROUND=1 ./run_avg.sh
done
