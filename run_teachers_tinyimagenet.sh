#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/data2/dingyu/datasets/tiny-imagenet-200}"
MODEL="${MODEL:-resnet32x4}"
EPOCHS="${EPOCHS:-240}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
LR_DECAY_RATE="${LR_DECAY_RATE:-0.1}"
SEED="${SEED:-42}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoint}"
WAIT_FOR_TMUX_SESSION="${WAIT_FOR_TMUX_SESSION:-}"

LOG_DIR="${PROJECT_DIR}/logs/tinyimagenet_teachers"
mkdir -p "${LOG_DIR}"

if [[ -n "${WAIT_FOR_TMUX_SESSION}" ]]; then
    echo "Waiting for tmux session ${WAIT_FOR_TMUX_SESSION} to finish."
    while tmux has-session -t "${WAIT_FOR_TMUX_SESSION}" 2>/dev/null; do
        sleep 60
    done
fi

run_teacher() {
    local aug_group="$1"
    local physical_gpu="$2"
    local log_path="${LOG_DIR}/${MODEL}_${aug_group}.out"
    echo "Starting ${MODEL}/${aug_group} on GPU ${physical_gpu}; log=${log_path}"
    env CUDA_VISIBLE_DEVICES="${physical_gpu}" "${PYTHON}" -u train_teacher_tinyimagenet.py \
        --gpu 0 \
        --data "${DATA_ROOT}" \
        --model "${MODEL}" \
        --aug-group "${aug_group}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --workers "${WORKERS}" \
        --learning-rate "${LEARNING_RATE}" \
        --lr-decay-epochs 150 180 210 \
        --lr-decay-rate "${LR_DECAY_RATE}" \
        --momentum "${MOMENTUM}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --seed "${SEED}" \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        > "${log_path}" 2>&1
}

# Three GPUs are available. A3 is queued behind A0 instead of sharing a GPU.
run_teacher A0 0 &
pid_a0=$!
run_teacher A1 1 &
pid_a1=$!
run_teacher A2 2 &
pid_a2=$!

wait "${pid_a0}"
run_teacher A3 0 &
pid_a3=$!

wait "${pid_a1}"
wait "${pid_a2}"
wait "${pid_a3}"
echo "All Tiny-ImageNet teachers finished."
