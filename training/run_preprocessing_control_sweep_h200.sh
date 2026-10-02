#!/usr/bin/env bash
# 전처리 대조 분류 실험 (H200 1장)
#
# 새로 학습하는 입력 5개:
#   medsam3_center, medsam3_margin0, medsam3_margin10, medsam3_margin20, medsam3_soft
# 원본 / 본 실험 크롭 / hard mask 는 results_pneumonia (학습 seed 42)를 재사용합니다.
#
# 고정:
#   환자 분할 seed 42, splits/ 의 AP·PA CSV
#   학습 seed 42 (기존 70-run과 같은 시작값)
#   대표 모델: resnet152, eva_x_base
#
# 기본 범위: AP, PA × 5 controls × 2 models = 20 runs
#
# 명령만 확인:
#   DRY_RUN=1 bash run_preprocessing_control_sweep_h200.sh
#
# 한 조합만 1 fold 점검:
#   MAX_FOLDS=1 VIEW=PA ARCHS="resnet152" DATA_MODES="medsam3_center" \
#     bash run_preprocessing_control_sweep_h200.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SWEEP_SCRIPT="${SCRIPT_DIR}/run_all_base_backbones_sweep_h200.sh"

export SPLIT_SEED=42
export TRAINING_SEED=42
export VIEW="${VIEW:-both}"
export ARCHS="${ARCHS:-${ARCH:-resnet152 eva_x_base}}"
export DATA_MODES="${DATA_MODES:-${DATA_MODE:-medsam3_center medsam3_margin0 medsam3_margin10 medsam3_margin20 medsam3_soft}}"
export RESULTS_ROOT="${RESULTS_ROOT:-${STUDY_DIR}/results_pneumonia_controls}"
export LOG_DIR="${LOG_DIR:-${RESULTS_ROOT}/logs}"

echo ""
echo "전처리 대조 분류 실험"
echo "  split seed         : $SPLIT_SEED"
echo "  training seed      : $TRAINING_SEED"
echo "  views              : $VIEW"
echo "  data modes         : $DATA_MODES"
echo "  backbones          : $ARCHS"
echo "  results            : $RESULTS_ROOT"
echo "  재사용              : results_pneumonia 의 raw / medsam3_crop / medsam3_seg (seed 42)"
echo ""

bash "$SWEEP_SCRIPT"
