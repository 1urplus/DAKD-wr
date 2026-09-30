#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/data2/dingyu/datasets/tiny-imagenet-200}"
MODEL="${MODEL:-resnet32x4}"
AUG_GROUP="${AUG_GROUP:?AUG_GROUP must be A0, A1, A2, or A3}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-240}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-8}"
RESUME="${RESUME:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoint}"
LOG_DIR="${PROJECT_DIR}/logs/tinyimagenet_teachers"
LOG_PATH="${LOG_DIR}/${MODEL}_${AUG_GROUP}_resume.out"
mkdir -p "${LOG_DIR}"

command=(
    env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u train_teacher_tinyimagenet.py
    --gpu 0
    --data "${DATA_ROOT}"
    --model "${MODEL}"
    --aug-group "${AUG_GROUP}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --workers "${WORKERS}"
    --learning-rate 0.05
    --lr-decay-epochs 150 180 210
    --lr-decay-rate 0.1
    --momentum 0.9
    --weight-decay 0.0005
    --seed 42
    --checkpoint-dir "${CHECKPOINT_DIR}"
)
if [[ -n "${RESUME}" ]]; then
    command+=(--resume "${RESUME}")
fi

echo "Starting ${MODEL}/${AUG_GROUP} on physical GPU ${GPU}; resume=${RESUME:-none}" >> "${LOG_PATH}"
exec "${command[@]}" >> "${LOG_PATH}" 2>&1
