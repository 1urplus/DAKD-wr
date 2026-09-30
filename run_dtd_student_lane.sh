#!/usr/bin/env bash
# Run one serial lane of the DTD split-1 distillation matrix.
set -euo pipefail

if (( $# < 3 || ($# - 1) % 2 != 0 )); then
  echo "usage: $0 PHYSICAL_GPU TEACHER_MODEL STUDENT [TEACHER_MODEL STUDENT ...]" >&2
  exit 2
fi

gpu="$1"
shift
data_root="${DTD_DATA_ROOT:-/data2/xujianyang/dtd_3modals}"
python_bin="${PYTHON_BIN:-/data/miniconda3/envs/clip/bin/python}"

while (( $# > 0 )); do
  teacher_model="$1"
  student="$2"
  shift 2

  case "$teacher_model" in
    vgg13) teachers="dtdvggA0 dtdvggA1 dtdvggA2 dtdvggA3" ;;
    resnet32x4) teachers="dtdA0 dtdA1 dtdA2 dtdA3" ;;
    *) echo "unsupported teacher: $teacher_model" >&2; exit 2 ;;
  esac

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
      echo "Starting DTD split 1: ${method} ${distiller} ${teacher_model} -> ${student} on GPU ${gpu}"
      if [[ "$method" == ours ]]; then
        env PYTHON="$python_bin" GPU="$gpu" DATASET=dtd DATA_ROOT="$data_root" \
          T_MODEL="$teacher_model" ARCH="$student" TEACHER_LIST="$teachers" \
          DISTILLER="$distiller" EPOCHS=240 BATCH_SIZE=64 SEED=42 \
          INIT_LR=0.05 LR_DECAY=0.1 LR_TYPE=cosine \
          CE_WEIGHT=1 KD_WEIGHT=1 FEAT_WEIGHT=3 SP_WEIGHT="$sp_weight" \
          TRAIN_PASSES=1 MODEL_EMA_DECAY=0.9999 VAL_SIZE=0 LATE_TEST_START=210 \
          WEIGHT_DECAY=0.0001 RUN_LABEL=dtd-split1-standard \
          FOREGROUND=1 ./run_ours.sh
      else
        env PYTHON="$python_bin" GPU="$gpu" DATASET=dtd DATA_ROOT="$data_root" \
          T_MODEL="$teacher_model" ARCH="$student" TEACHER_LIST="$teachers" \
          DISTILLER="$distiller" EPOCHS=240 BATCH_SIZE=64 SEED=42 \
          INIT_LR=0.05 LR_DECAY=0.01 LR_TYPE=multistep \
          CE_WEIGHT=1 KD_WEIGHT=1 FEAT_WEIGHT=5 SP_WEIGHT="$sp_weight" \
          FEAT_KD="$avg_feat_kd" WEIGHT_DECAY=0.0001 \
          RUN_LABEL=dtd-split1-standard FOREGROUND=1 ./run_avg.sh
      fi
    done
  done
done
