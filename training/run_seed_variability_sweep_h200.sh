#!/usr/bin/env bash
# 학습 seed 변동성 실험 (H200 1장)
#
# 목적:
#   환자 split은 기존 70-run과 완전히 같게 두고,
#   학습 난수(초기화, 셔플)만 바꿔서 전처리 차이가
#   학습 흔들림보다 큰지 측정합니다.
#
# 고정:
#   --split-seed 42
#   splits/ 의 AP·PA outer-test / fold CSV (있으면 그대로 사용)
#
# 학습 seed:
#   42   = 기존 results_pneumonia 70-run (이 스크립트는 다시 안 돌림)
#   123  = 추가 seed 1
#   2026 = 추가 seed 2
#
# 기본 범위 (리뷰어 "주요 실험"):
#   AP, PA × 5 data modes × resnet152 + eva_x_base
#   = 20 configs × 2 new seeds = 40 runs
#
# 기본 실행:
#   bash run_seed_variability_sweep_h200.sh
#
# 1 fold 점검:
#   MAX_FOLDS=1 VIEW=AP ARCHS="resnet152" DATA_MODES="raw" \
#     SEEDS="123" bash run_seed_variability_sweep_h200.sh
#
# 7개 백본 전부:
#   ARCHS="resnet152 densenet convnextv2_base swint_base dinov3_base eva_x_base rad_dino" \
#     bash run_seed_variability_sweep_h200.sh
#
# 명령만 확인:
#   DRY_RUN=1 bash run_seed_variability_sweep_h200.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SWEEP_SCRIPT="${SCRIPT_DIR}/run_all_base_backbones_sweep_h200.sh"

if [[ ! -f "$SWEEP_SCRIPT" ]]; then
  echo "오류: 스윕 스크립트를 찾을 수 없습니다: $SWEEP_SCRIPT" >&2
  exit 1
fi

# 환자 구성은 seed마다 바꾸지 않습니다.
export SPLIT_SEED=42

# 기존 70-run(seed=42)은 다시 학습하지 않습니다.
read -r -a SEED_LIST <<< "${SEEDS:-123 2026}"

if (( ${#SEED_LIST[@]} == 0 )); then
  echo "오류: SEEDS가 비어 있습니다." >&2
  exit 1
fi

for seed in "${SEED_LIST[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "오류: 학습 seed는 정수여야 합니다: $seed" >&2
    exit 1
  fi
  if [[ "$seed" == "42" ]]; then
    echo "오류: training seed 42는 기존 70-run입니다. 이 스크립트는 123/2026만 돌립니다." >&2
    echo "기존 결과는 RESULTS_ROOT=${STUDY_DIR}/results_pneumonia 를 사용하세요." >&2
    exit 1
  fi
done

export VIEW="${VIEW:-both}"
export DATA_MODES="${DATA_MODES:-${DATA_MODE:-raw medsam3_seg medsam3_crop chexmask_seg chexmask_crop}}"
export ARCHS="${ARCHS:-${ARCH:-resnet152 eva_x_base}}"

RESULTS_ROOT_BASE="${RESULTS_ROOT_BASE:-${STUDY_DIR}/results_pneumonia}"

echo ""
echo "학습 seed 변동성 스윕"
echo "  split seed (고정)  : $SPLIT_SEED"
echo "  training seeds     : ${SEED_LIST[*]}  (42는 기존 70-run 재사용)"
echo "  views              : $VIEW"
echo "  data modes         : $DATA_MODES"
echo "  backbones          : $ARCHS"
echo "  결과 루트 패턴     : ${RESULTS_ROOT_BASE}_seed<SEED>"
echo ""

for seed in "${SEED_LIST[@]}"; do
  export TRAINING_SEED="$seed"
  export RESULTS_ROOT="${RESULTS_ROOT_BASE}_seed${seed}"
  export LOG_DIR="${RESULTS_ROOT}/logs"

  echo "============================================================"
  echo "시작: split_seed=${SPLIT_SEED}  training_seed=${TRAINING_SEED}"
  echo "결과: $RESULTS_ROOT"
  echo "============================================================"

  bash "$SWEEP_SCRIPT"
done

echo "학습 seed 변동성 스윕 종료: seeds=${SEED_LIST[*]}"
