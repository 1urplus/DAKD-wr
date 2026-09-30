#!/bin/bash
set -euo pipefail

# Proposed RL multi-teacher distillation launcher, seed 42.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
GPU="${GPU:-2}"
T_MODEL="${T_MODEL:-resnet32x4}"
ARCH="${ARCH:-ShuffleV2}"
DISTILLER="${DISTILLER:-kl}"
DATASET="${DATASET:-cifar100}"
TEACHER_LIST_TEXT="${TEACHER_LIST:-A0 A1 A2 A3}"
read -r -a TEACHER_LIST <<< "${TEACHER_LIST_TEXT}"
EPOCHS="${EPOCHS:-240}"
TRAIN_PASSES="${TRAIN_PASSES:-1}"
NORMALIZE_FIRST_EPOCH_ACTIONS="${NORMALIZE_FIRST_EPOCH_ACTIONS:-0}"
MODEL_EMA_DECAY="${MODEL_EMA_DECAY:-0.9999}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VAL_SIZE="${VAL_SIZE:-0}"
AUG_AGENT_LR="${AUG_AGENT_LR:-0.0001}"
AUG_ENTROPY_COEF="${AUG_ENTROPY_COEF:-0.001}"
AUG_BASELINE_MOMENTUM="${AUG_BASELINE_MOMENTUM:-0.9}"
LATE_TEST_START="${LATE_TEST_START:-210}"

# The machine's base PyTorch is built with CUDA 13, while the NVRTC runtime is
# installed in Conda's package cache rather than the default linker path.
NVRTC_LIB_DIR="${NVRTC_LIB_DIR:-/data/miniconda3/pkgs/cuda-nvrtc-13.0.88-hb0a779f_0/lib}"
NVRTC_TARGET_LIB_DIR="${NVRTC_TARGET_LIB_DIR:-/data/miniconda3/pkgs/cuda-nvrtc-13.0.88-hb0a779f_0/targets/x86_64-linux/lib}"
if [[ -f "${NVRTC_LIB_DIR}/libnvrtc-builtins.so.13.0" ]]; then
  export LD_LIBRARY_PATH="${NVRTC_LIB_DIR}:${NVRTC_TARGET_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

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
KD_T="${KD_T:-4}"
DKD_ALPHA="${DKD_ALPHA:-1.0}"
DKD_BETA="${DKD_BETA:-2.0}"
RKD_DIST_W="${RKD_DIST_W:-25.0}"
RKD_ANGLE_W="${RKD_ANGLE_W:-50.0}"
CATKD_T="${CATKD_T:-4.0}"
CATKD_ALPHA="${CATKD_ALPHA:-1.0}"
CATKD_BETA="${CATKD_BETA:-1.0}"
CATKD_GAMMA="${CATKD_GAMMA:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
LOSS_ANNEAL_START="${LOSS_ANNEAL_START:--1}"
KD_WEIGHT_END="${KD_WEIGHT_END:-${KD_WEIGHT}}"
FEAT_WEIGHT_END="${FEAT_WEIGHT_END:-${FEAT_WEIGHT}}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_LABEL="${RUN_LABEL:-manual}"
RUN_NAME="ours_${T_MODEL}_to_${ARCH}_${DISTILLER}_seed${SEED}_ilr${INIT_LR}_decay${LR_DECAY}_fw${FEAT_WEIGHT}_T${KD_T}_${RUN_LABEL}_${RUN_ID}"
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
method=ours
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
train_passes=${TRAIN_PASSES}
normalize_first_epoch_actions=${NORMALIZE_FIRST_EPOCH_ACTIONS}
model_ema_decay=${MODEL_EMA_DECAY}
seed=${SEED}
batch_size=${BATCH_SIZE}
val_size=${VAL_SIZE}
aug_agent_lr=${AUG_AGENT_LR}
aug_entropy_coef=${AUG_ENTROPY_COEF}
aug_baseline_momentum=${AUG_BASELINE_MOMENTUM}
late_test_start=${LATE_TEST_START}
init_lr=${INIT_LR}
lr_decay=${LR_DECAY}
lr_type=${LR_TYPE}
milestones=${MILESTONES[*]}
ce_weight=${CE_WEIGHT}
kd_weight=${KD_WEIGHT}
feat_weight=${FEAT_WEIGHT}
sp_weight=${SP_WEIGHT}
loss_anneal_start=${LOSS_ANNEAL_START}
kd_weight_end=${KD_WEIGHT_END}
feat_weight_end=${FEAT_WEIGHT_END}
kd_temperature=${KD_T}
dkd_alpha=${DKD_ALPHA}
dkd_beta=${DKD_BETA}
rkd_distance_weight=${RKD_DIST_W}
rkd_angle_weight=${RKD_ANGLE_W}
catkd_temperature=${CATKD_T}
catkd_alpha=${CATKD_ALPHA}
catkd_beta=${CATKD_BETA}
catkd_gamma=${CATKD_GAMMA}
weight_decay=${WEIGHT_DECAY}
momentum=0.9
git_commit=${GIT_COMMIT}
worktree_status=${GIT_STATUS//$'\n'/; }
log=${LOG_PATH}
checkpoint_dir=${CHECKPOINT_DIR}
code_patch=${PATCH_PATH}
EOF

echo "========== 启动 ours-KD 训练 =========="
cat "${PARAM_PATH}"
echo "======================================="

TRAIN_CMD=(env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" train_ours.py \
  --gpu 0 \
  --data "${DATA_ROOT}" \
  --arch "${ARCH}" \
  --t_model "${T_MODEL}" \
  --distiller "${DISTILLER}" \
  --dataset "${DATASET}" \
  --teacher-name-list "${TEACHER_LIST[@]}" \
  --epochs "${EPOCHS}" \
  --train-passes "${TRAIN_PASSES}" \
  --normalize-first-epoch-actions "${NORMALIZE_FIRST_EPOCH_ACTIONS}" \
  --model-ema-decay "${MODEL_EMA_DECAY}" \
  --batch-size "${BATCH_SIZE}" \
  --val-size "${VAL_SIZE}" \
  --aug-agent-lr "${AUG_AGENT_LR}" \
  --aug-entropy-coef "${AUG_ENTROPY_COEF}" \
  --aug-baseline-momentum "${AUG_BASELINE_MOMENTUM}" \
  --late-test-start "${LATE_TEST_START}" \
  --seed "${SEED}" \
  --init-lr "${INIT_LR}" \
  --lr "${LR_DECAY}" \
  --lr-type "${LR_TYPE}" \
  --milestones "${MILESTONES[@]}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --ce-weight "${CE_WEIGHT}" \
  --kd-weight "${KD_WEIGHT}" \
  --feat-weight "${FEAT_WEIGHT}" \
  --sp-weight "${SP_WEIGHT}" \
  --loss-anneal-start "${LOSS_ANNEAL_START}" \
  --kd-weight-end "${KD_WEIGHT_END}" \
  --feat-weight-end "${FEAT_WEIGHT_END}" \
  --kd-T "${KD_T}" \
  --dkd-alpha "${DKD_ALPHA}" \
  --dkd-beta "${DKD_BETA}" \
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
