#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

: "${GPU:?Set GPU to the physical GPU index}"
: "${ARCH:?Set ARCH to vgg8 or wrn_16_2}"
LANE_RUN_ID="${LANE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

case "${ARCH}" in
  vgg8)
    T_MODEL=vgg13
    TEACHER_LIST="tinyvggA0 tinyvggA1 tinyvggA2 tinyvggA3"
    ;;
  wrn_16_2)
    T_MODEL=resnet32x4
    TEACHER_LIST="tinyA0 tinyA1 tinyA2 tinyA3"
    ;;
  *)
    echo "Unsupported student: ${ARCH}" >&2
    exit 2
    ;;
esac

for distiller in sp catkd rkd; do
  env \
    GPU="${GPU}" \
    ARCH="${ARCH}" \
    T_MODEL="${T_MODEL}" \
    TEACHER_LIST="${TEACHER_LIST}" \
    DISTILLER="${distiller}" \
    RUN_ID="${LANE_RUN_ID}" \
    ./run_tiny_standard_method_pair.sh
done

