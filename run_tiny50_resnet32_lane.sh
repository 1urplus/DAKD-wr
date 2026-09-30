#!/usr/bin/env bash
# Tiny-ImageNet ResNet50 (A0-A3) -> ResNet32 standard distillation matrix.
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 PHYSICAL_GPU" >&2
  exit 2
fi

gpu="$1"
python_bin="${PYTHON_BIN:-/data/miniconda3/envs/clip/bin/python}"
data_root="${TINY_DATA_ROOT:-/data2/dingyu/datasets/tiny-imagenet-200}"
teachers="tiny50A0 tiny50A1 tiny50A2 tiny50A3"

for distiller in kl dkd sp catkd rkd; do
  sp_weight=1
  avg_feat_kd=mse
  if [[ "$distiller" == sp ]]; then
    sp_weight=3000
    avg_feat_kd=none
  elif [[ "$distiller" == rkd ]]; then
    avg_feat_kd=none
  fi

  for method in ours avg; do
    echo "Starting Tiny-ImageNet ResNet50 -> ResNet32: ${method} ${distiller} on GPU ${gpu}"
    if [[ "$method" == ours ]]; then
      env PYTHON="$python_bin" GPU="$gpu" DATASET=tinyimagenet DATA_ROOT="$data_root" \
        T_MODEL=ResNet50 ARCH=resnet32 TEACHER_LIST="$teachers" \
        DISTILLER="$distiller" EPOCHS=240 BATCH_SIZE=64 SEED=42 \
        INIT_LR=0.05 LR_DECAY=0.1 LR_TYPE=cosine \
        CE_WEIGHT=1 KD_WEIGHT=1 FEAT_WEIGHT=3 SP_WEIGHT="$sp_weight" \
        TRAIN_PASSES=1 MODEL_EMA_DECAY=0.9999 VAL_SIZE=0 LATE_TEST_START=210 \
        WEIGHT_DECAY=0.0001 RUN_LABEL=tinyimagenet-ResNet50-resnet32-standard \
        FOREGROUND=1 ./run_ours.sh
    else
      env PYTHON="$python_bin" GPU="$gpu" DATASET=tinyimagenet DATA_ROOT="$data_root" \
        T_MODEL=ResNet50 ARCH=resnet32 TEACHER_LIST="$teachers" \
        DISTILLER="$distiller" EPOCHS=240 BATCH_SIZE=64 SEED=42 \
        INIT_LR=0.05 LR_DECAY=0.01 LR_TYPE=multistep \
        CE_WEIGHT=1 KD_WEIGHT=1 FEAT_WEIGHT=5 SP_WEIGHT="$sp_weight" \
        FEAT_KD="$avg_feat_kd" WEIGHT_DECAY=0.0001 \
        RUN_LABEL=tinyimagenet-ResNet50-resnet32-standard FOREGROUND=1 ./run_avg.sh
    fi
  done
done
