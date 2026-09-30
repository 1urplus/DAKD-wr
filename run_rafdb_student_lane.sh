#!/usr/bin/env bash
# One RAF-DB teacher/student pair: KL, DKD, SP, CATKD, RKD, OURS and AVG.
set -euo pipefail

if (( $# != 3 )); then
  echo "usage: $0 PHYSICAL_GPU TEACHER_MODEL STUDENT_MODEL" >&2
  exit 2
fi

gpu="$1"
teacher_model="$2"
student="$3"
python_bin="${PYTHON_BIN:-/data/miniconda3/envs/clip/bin/python}"
data_root="${RAFDB_DATA_ROOT:-/data2/xujianyang/rafdb_3modals}"

case "$teacher_model" in
  vgg13) teachers="rafvggA0 rafvggA1 rafvggA2 rafvggA3" ;;
  resnet32x4) teachers="rafA0 rafA1 rafA2 rafA3" ;;
  *) echo "unsupported RAF-DB teacher: $teacher_model" >&2; exit 2 ;;
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
    echo "Starting RAF-DB: ${method} ${distiller} ${teacher_model} -> ${student} on GPU ${gpu}"
    if [[ "$method" == ours ]]; then
      env PYTHON="$python_bin" GPU="$gpu" DATASET=rafdb DATA_ROOT="$data_root" \
        T_MODEL="$teacher_model" ARCH="$student" TEACHER_LIST="$teachers" \
        DISTILLER="$distiller" EPOCHS=240 BATCH_SIZE=64 SEED=42 \
        INIT_LR=0.05 LR_DECAY=0.1 LR_TYPE=cosine \
        CE_WEIGHT=1 KD_WEIGHT=1 FEAT_WEIGHT=3 SP_WEIGHT="$sp_weight" \
        TRAIN_PASSES=1 MODEL_EMA_DECAY=0.9999 VAL_SIZE=0 LATE_TEST_START=210 \
        WEIGHT_DECAY=0.0001 RUN_LABEL=rafdb-standard \
        FOREGROUND=1 ./run_ours.sh
    else
      env PYTHON="$python_bin" GPU="$gpu" DATASET=rafdb DATA_ROOT="$data_root" \
        T_MODEL="$teacher_model" ARCH="$student" TEACHER_LIST="$teachers" \
        DISTILLER="$distiller" EPOCHS=240 BATCH_SIZE=64 SEED=42 \
        INIT_LR=0.05 LR_DECAY=0.01 LR_TYPE=multistep \
        CE_WEIGHT=1 KD_WEIGHT=1 FEAT_WEIGHT=5 SP_WEIGHT="$sp_weight" \
        FEAT_KD="$avg_feat_kd" WEIGHT_DECAY=0.0001 \
        RUN_LABEL=rafdb-standard FOREGROUND=1 ./run_avg.sh
    fi
  done
done
