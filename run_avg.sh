#!/bin/bash
set -euo pipefail

# Fixed average multi-teacher distillation launcher, seed 42.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
GPU="${GPU:-1}"
T_MODEL="${T_MODEL:-resnet32x4}"
ARCH="${ARCH:-ShuffleV2}"
DISTILLER="${DISTILLER:-kl}"
DATASET="${DATASET:-cifar100}"
TEACHER_LIST_TEXT="${TEACHER_LIST:-A0 A1 A2 A3}"
read -r -a TEACHER_LIST <<< "${TEACHER_LIST_TEXT}"
EPOCHS="${EPOCHS:-240}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LATE_TEST_START="${LATE_TEST_START:-210}"

# utils.adjust_lr uses INIT_LR as the initial LR and LR_DECAY as the
# multiplicative decay at epochs 150/180/210.
INIT_LR="${INIT_LR:-0.05}"
LR_DECAY="${LR_DECAY:-0.1}"
LR_TYPE="${LR_TYPE:-multistep}"
MILESTONES_TEXT="${MILESTONES_OVERRIDE:-150 180 210}"
read -r -a MILESTONES <<< "${MILESTONES_TEXT}"
CE_WEIGHT="${CE_WEIGHT:-1}"
KD_WEIGHT="${KD_WEIGHT:-1}"
FEAT_WEIGHT="${FEAT_WEIGHT:-1}"
SP_WEIGHT="${SP_WEIGHT:-1.0}"
FEAT_KD="${FEAT_KD:-mse}"
KD_T="${KD_T:-4}"
DKD_ALPHA="${DKD_ALPHA:-1.0}"
DKD_BETA="${DKD_BETA:-2.0}"
DKD_WARMUP="${DKD_WARMUP:-0}"
RKD_DIST_W="${RKD_DIST_W:-25.0}"
RKD_ANGLE_W="${RKD_ANGLE_W:-50.0}"
CATKD_T="${CATKD_T:-4.0}"
CATKD_ALPHA="${CATKD_ALPHA:-1.0}"
CATKD_BETA="${CATKD_BETA:-1.0}"
CATKD_GAMMA="${CATKD_GAMMA:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
MOMENTUM="${MOMENTUM:-0.9}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_LABEL="${RUN_LABEL:-manual}"
QUEUE_HOLD_FILE="${PROJECT_DIR}/.queue_holds/${DISTILLER}_${ARCH}_${RUN_ID}"
if [[ "${IGNORE_QUEUE_HOLD:-0}" != "1" && -f "${QUEUE_HOLD_FILE}" ]]; then
  echo "AVG run deferred by queue hold: ${QUEUE_HOLD_FILE}"
  exit 0
fi
RUN_NAME="avg_${T_MODEL}_to_${ARCH}_${DISTILLER}_seed${SEED}_ilr${INIT_LR}_decay${LR_DECAY}_fw${FEAT_WEIGHT}_T${KD_T}_${RUN_LABEL}_${RUN_ID}"
LOG_DIR="logs/retune_20260902"
CHECKPOINT_DIR="checkpoint/retune_20260902/${RUN_NAME}"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.out"
PARAM_PATH="${LOG_DIR}/${RUN_NAME}.params"
PATCH_PATH="${LOG_DIR}/${RUN_NAME}.patch"
mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_COMMIT="$(git rev-parse HEAD)"
  GIT_STATUS="$(git status --short)"
  git diff -- . > "${PATCH_PATH}"
else
  GIT_COMMIT="not-a-git-repository"
  GIT_STATUS="not-a-git-repository"
  : > "${PATCH_PATH}"
fi

cat > "${PARAM_PATH}" <<EOF
method=avg
run_label=${RUN_LABEL}
python=${PYTHON}
physical_gpu=${GPU}
teacher_model=${T_MODEL}
student_model=${ARCH}
distiller=${DISTILLER}
dataset=${DATASET}
data_root=${DATA_ROOT}
teachers=${TEACHER_LIST[*]}
epochs=${EPOCHS}
seed=${SEED}
batch_size=${BATCH_SIZE}
late_test_start=${LATE_TEST_START}
init_lr=${INIT_LR}
lr_decay=${LR_DECAY}
lr_type=${LR_TYPE}
milestones=${MILESTONES[*]}
ce_weight=${CE_WEIGHT}
kd_weight=${KD_WEIGHT}
feat_weight=${FEAT_WEIGHT}
sp_weight=${SP_WEIGHT}
feat_kd=${FEAT_KD}
kd_temperature=${KD_T}
dkd_alpha=${DKD_ALPHA}
dkd_beta=${DKD_BETA}
dkd_warmup=${DKD_WARMUP}
rkd_distance_weight=${RKD_DIST_W}
rkd_angle_weight=${RKD_ANGLE_W}
catkd_temperature=${CATKD_T}
catkd_alpha=${CATKD_ALPHA}
catkd_beta=${CATKD_BETA}
catkd_gamma=${CATKD_GAMMA}
weight_decay=${WEIGHT_DECAY}
momentum=${MOMENTUM}
git_commit=${GIT_COMMIT}
worktree_status=${GIT_STATUS//$'\n'/; }
log=${LOG_PATH}
checkpoint_dir=${CHECKPOINT_DIR}
code_patch=${PATCH_PATH}
EOF

echo "========== 启动 avg-KD 训练 =========="
cat "${PARAM_PATH}"
echo "======================================"

TRAIN_CMD=(env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" train_avg.py \
  --gpu 0 \
  --data "${DATA_ROOT}" \
  --arch "${ARCH}" \
  --t_model "${T_MODEL}" \
  --distiller "${DISTILLER}" \
  --dataset "${DATASET}" \
  --teacher-name-list "${TEACHER_LIST[@]}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --late-test-start "${LATE_TEST_START}" \
  --seed "${SEED}" \
  --init-lr "${INIT_LR}" \
  --lr "${LR_DECAY}" \
  --lr-type "${LR_TYPE}" \
  --milestones "${MILESTONES[@]}" \
  --momentum "${MOMENTUM}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --ce-weight "${CE_WEIGHT}" \
  --kd-weight "${KD_WEIGHT}" \
  --feat-weight "${FEAT_WEIGHT}" \
  --sp-weight "${SP_WEIGHT}" \
  --feat-kd "${FEAT_KD}" \
  --kd-T "${KD_T}" \
  --dkd-alpha "${DKD_ALPHA}" \
  --dkd-beta "${DKD_BETA}" \
  --dkd-warmup "${DKD_WARMUP}" \
  --rkd-dist-w "${RKD_DIST_W}" \
  --rkd-angle-w "${RKD_ANGLE_W}" \
  --catkd-T "${CATKD_T}" \
  --catkd-alpha "${CATKD_ALPHA}" \
  --catkd-beta "${CATKD_BETA}" \
  --catkd-gamma "${CATKD_GAMMA}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --trial "${RUN_NAME}" \
  --dist-backend nccl \
  --world-size 1 \
  --rank 0)

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  echo "pid=$$" >> "${PARAM_PATH}"
  echo "训练由当前会话托管，PID: $$"
  exec "${TRAIN_CMD[@]}" > "${LOG_PATH}" 2>&1
fi

nohup "${TRAIN_CMD[@]}" > "${LOG_PATH}" 2>&1 &
PID=$!
echo "pid=${PID}" >> "${PARAM_PATH}"
echo "训练已启动，PID: ${PID}"
echo "日志: ${LOG_PATH}"
echo "参数: ${PARAM_PATH}"
