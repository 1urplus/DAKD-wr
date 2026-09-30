#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

WAIT_FOR_TMUX_SESSION="${WAIT_FOR_TMUX_SESSION:-}"
if [[ -n "${WAIT_FOR_TMUX_SESSION}" ]]; then
    echo "Waiting for tmux session ${WAIT_FOR_TMUX_SESSION} to release GPU 1."
    while tmux has-session -t "${WAIT_FOR_TMUX_SESSION}" 2>/dev/null; do
        sleep 60
    done
fi

exec env \
    GPU="${GPU:-1}" \
    DATA_ROOT=/data1/xujianyang/MTKD-RL-two/data \
    T_MODEL=vgg13 \
    ARCH=vgg8 \
    TEACHER_LIST="vggA0 vggA1 vggA2 vggA3" \
    EPOCHS=240 \
    SEED=42 \
    BATCH_SIZE=64 \
    INIT_LR=0.05 \
    LR_DECAY=0.01 \
    MILESTONES_OVERRIDE="150 180 210" \
    CE_WEIGHT=1 \
    KD_WEIGHT=1 \
    FEAT_WEIGHT=5 \
    KD_T=4 \
    RUN_LABEL=vgg13-vgg8-decay001-fw5 \
    FOREGROUND=1 \
    ./run_avg.sh
