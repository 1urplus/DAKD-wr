#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

: "${GPU:?Set GPU to the physical GPU index}"
: "${ARCH:?Set the student architecture}"
: "${T_MODEL:?Set the teacher architecture}"
: "${TEACHER_LIST:?Set the four-teacher list}"
: "${DISTILLER:?Set DISTILLER to sp, catkd, or rkd}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

case "${DISTILLER}" in
  sp|catkd|rkd) ;;
  *) echo "Unsupported distiller: ${DISTILLER}" >&2; exit 2 ;;
esac

# A method pair may be pulled forward into a parallel GPU lane while the
# original architecture lane is still running an earlier method. Serialize
# identical pairs and leave per-stage markers so the original lane skips work
# that has already completed instead of launching a duplicate.
LOCK_DIR="${PROJECT_DIR}/.standard_queue_locks"
mkdir -p "${LOCK_DIR}"
PAIR_KEY="${T_MODEL}_${ARCH}_${DISTILLER}_${RUN_ID}"
PAIR_LOCK="${LOCK_DIR}/${PAIR_KEY}.lock"
OURS_DONE="${LOCK_DIR}/${PAIR_KEY}.ours.done"
AVG_DONE="${LOCK_DIR}/${PAIR_KEY}.avg.done"
exec 9>"${PAIR_LOCK}"
flock 9

COMMON_ENV=(
  PYTHON=/data/miniconda3/envs/clip/bin/python
  DATA_ROOT=/data2/dingyu/datasets/tiny-imagenet-200
  DATASET=tinyimagenet
  GPU="${GPU}"
  T_MODEL="${T_MODEL}"
  ARCH="${ARCH}"
  TEACHER_LIST="${TEACHER_LIST}"
  DISTILLER="${DISTILLER}"
  EPOCHS=240
  SEED=42
  BATCH_SIZE=64
  MILESTONES_OVERRIDE="150 180 210"
  CE_WEIGHT=1
  KD_WEIGHT=1
  KD_T=4
  DKD_ALPHA=1
  DKD_BETA=2
  RKD_DIST_W=25
  RKD_ANGLE_W=50
  CATKD_T=4
  CATKD_ALPHA=1
  CATKD_BETA=1
  CATKD_GAMMA=1
  WEIGHT_DECAY=0.0001
  RUN_ID="${RUN_ID}"
  FOREGROUND=1
)

OURS_SP_WEIGHT=1
AVG_SP_WEIGHT=1
if [[ "${DISTILLER}" == "sp" ]]; then
  OURS_SP_WEIGHT=3000
  AVG_SP_WEIGHT=3000
fi

if [[ -f "${OURS_DONE}" ]]; then
  echo "Standard OURS ${DISTILLER} for ${T_MODEL} -> ${ARCH} already completed; skipping."
else
  echo "Starting standard OURS ${DISTILLER} for ${T_MODEL} -> ${ARCH} on GPU ${GPU}."
  env "${COMMON_ENV[@]}" \
    NVRTC_LIB_DIR=/tmp/dckd_clip_no_nvrtc_override \
    NVRTC_TARGET_LIB_DIR=/tmp/dckd_clip_no_nvrtc_override \
    TRAIN_PASSES=1 \
    MODEL_EMA_DECAY=0.9999 \
    VAL_SIZE=0 \
    AUG_AGENT_LR=0.0001 \
    AUG_ENTROPY_COEF=0.001 \
    AUG_BASELINE_MOMENTUM=0.9 \
    LATE_TEST_START=210 \
    INIT_LR=0.05 \
    LR_DECAY=0.1 \
    LR_TYPE=cosine \
    FEAT_WEIGHT=3 \
    SP_WEIGHT="${OURS_SP_WEIGHT}" \
    LOSS_ANNEAL_START=-1 \
    KD_WEIGHT_END=1 \
    FEAT_WEIGHT_END=3 \
    RUN_LABEL="tinyimagenet-${T_MODEL}-${ARCH}-${DISTILLER}-standard-cosine" \
    ./run_ours.sh
  touch "${OURS_DONE}"
fi

if [[ -f "${AVG_DONE}" ]]; then
  echo "Standard AVG ${DISTILLER} for ${T_MODEL} -> ${ARCH} already completed; skipping."
else
  echo "Starting standard AVG ${DISTILLER} for ${T_MODEL} -> ${ARCH} on GPU ${GPU}."
  env "${COMMON_ENV[@]}" \
    INIT_LR=0.05 \
    LR_DECAY=0.01 \
    LR_TYPE=multistep \
    FEAT_WEIGHT=5 \
    SP_WEIGHT="${AVG_SP_WEIGHT}" \
    FEAT_KD=none \
    MOMENTUM=0.9 \
    RUN_LABEL="tinyimagenet-${T_MODEL}-${ARCH}-${DISTILLER}-avg-standard-no-mse" \
    ./run_avg.sh
  touch "${AVG_DONE}"
fi
