#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || ($# - 1) % 2 != 0 )); then
    echo "usage: $0 GPU MODEL AUG_GROUP [MODEL AUG_GROUP ...]" >&2
    exit 2
fi

gpu="$1"
shift
python_bin="${PYTHON_BIN:-/data/miniconda3/envs/clip/bin/python}"
data_root="${DTD_DATA_ROOT:-/data2/xujianyang/dtd_3modals}"

while (( $# > 0 )); do
    model="$1"
    aug_group="$2"
    shift 2
    echo "Starting DTD teacher: model=${model} aug=${aug_group} gpu=${gpu}"
    "$python_bin" train_teacher_dtd.py \
        --data "$data_root" \
        --model "$model" \
        --aug-group "$aug_group" \
        --split 1 \
        --epochs 240 \
        --batch-size 32 \
        --workers 4 \
        --learning-rate 0.05 \
        --lr-decay-epochs 150 180 210 \
        --lr-decay-rate 0.1 \
        --momentum 0.9 \
        --weight-decay 0.0005 \
        --seed 42 \
        --gpu "$gpu" \
        --checkpoint-dir ./checkpoint
done
