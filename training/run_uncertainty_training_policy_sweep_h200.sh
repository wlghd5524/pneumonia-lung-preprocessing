#!/usr/bin/env bash
# Training-policy sensitivity (CheXpert U-Zero / U-One 재학습)
#
# 기존 evaluation-only U-Zero/U-One(라벨만 다시 코딩)과 별개입니다.
# 환자 분할, training seed, epoch, learning rate, augmentation은 70-run과 같게 두고
# 학습 라벨 정책만 바꿉니다. 검증과 outer test는 explicit 0/1 그대로입니다.
#
# Explicit-only는 다시 학습하지 않습니다.
# 기준선은 results_pneumonia 의 seed 42 run입니다.
#
# 기본 범위:
#   ResNet-152, EVA-X × Raw, MedSAM3 crop × AP, PA × U-Zero, U-One = 16 runs
#
# 실행 전 데이터 준비:
#   python data_preparation/build_uncertain_endpoint_labels.py --scope trainval
#   SCOPE=trainval SKIP_CHEXMASK=1 bash preprocessing/run_uncertain_endpoint_preprocess.sh
#   python preprocessing/finalize_uncertain_trainval_cohort.py --modes raw medsam3_crop
#
# 명령만 확인:
#   DRY_RUN=1 bash training/run_uncertainty_training_policy_sweep_h200.sh
#
# 1 fold만:
#   MAX_FOLDS=1 VIEW=AP ARCHS="resnet152" DATA_MODES="raw" POLICIES="u_zero" \
#     bash training/run_uncertainty_training_policy_sweep_h200.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SWEEP_SCRIPT="${SCRIPT_DIR}/run_all_base_backbones_sweep_h200.sh"

if [[ ! -f "$SWEEP_SCRIPT" ]]; then
  echo "오류: 스윕 스크립트를 찾을 수 없습니다: $SWEEP_SCRIPT" >&2
  exit 1
fi

export SPLIT_SEED=42
export TRAINING_SEED=42
export VIEW="${VIEW:-both}"
export DATA_MODES="${DATA_MODES:-${DATA_MODE:-raw medsam3_crop}}"
export ARCHS="${ARCHS:-${ARCH:-resnet152 eva_x_base}}"
export SPLIT_DIR="${SPLIT_DIR:-${STUDY_DIR}/splits}"

RESULTS_ROOT_BASE="${RESULTS_ROOT_BASE:-${STUDY_DIR}/results_pneumonia_trainpolicy}"
BASELINE_ROOT="${BASELINE_ROOT:-${STUDY_DIR}/results_pneumonia}"

read -r -a POLICY_LIST <<< "${POLICIES:-u_zero u_one}"
if (( ${#POLICY_LIST[@]} == 0 )); then
  echo "오류: POLICIES가 비어 있습니다." >&2
  exit 1
fi

for policy in "${POLICY_LIST[@]}"; do
  case "$policy" in
    u_zero|u_one) ;;
    explicit_only)
      echo "오류: explicit_only는 기존 70-run을 재사용합니다. POLICIES에서 빼세요." >&2
      echo "기준선: $BASELINE_ROOT" >&2
      exit 1
      ;;
    *)
      echo "오류: policy는 u_zero 또는 u_one 이어야 합니다: $policy" >&2
      exit 1
      ;;
  esac
done

_view_input="${VIEW^^}"
case "$_view_input" in
  AP) CHECK_VIEWS=(AP) ;;
  PA) CHECK_VIEWS=(PA) ;;
  BOTH|ALL|"") CHECK_VIEWS=(AP PA) ;;
  *)
    echo "오류: VIEW는 AP, PA, both 중 하나여야 합니다: $VIEW" >&2
    exit 1
    ;;
esac

missing=0
for view in "${CHECK_VIEWS[@]}"; do
  cohort="${SPLIT_DIR}/${view}_uncertain_trainval.csv"
  if [[ ! -f "$cohort" ]]; then
    echo "오류: uncertainty cohort CSV가 없습니다: $cohort" >&2
    missing=1
  fi
done
if (( missing == 1 )); then
  echo "먼저 데이터 준비를 실행하세요." >&2
  echo "  python data_preparation/build_uncertain_endpoint_labels.py --scope trainval" >&2
  echo "  SCOPE=trainval SKIP_CHEXMASK=1 bash preprocessing/run_uncertain_endpoint_preprocess.sh" >&2
  echo "  python preprocessing/finalize_uncertain_trainval_cohort.py --modes raw medsam3_crop" >&2
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    exit 1
  fi
  echo "DRY_RUN=1 이라 명령만 이어서 출력합니다. 실제 학습은 cohort CSV 없이 실패합니다." >&2
fi

echo ""
echo "Training-policy sensitivity 스윕"
echo "  이것은 evaluation-only U-Zero/U-One 분석이 아닙니다. 모델을 다시 학습합니다."
echo "  split seed (고정)     : $SPLIT_SEED"
echo "  training seed (고정)  : $TRAINING_SEED"
echo "  policies              : ${POLICY_LIST[*]}"
echo "  views                 : $VIEW"
echo "  data modes            : $DATA_MODES"
echo "  backbones             : $ARCHS"
echo "  explicit-only 기준선  : $BASELINE_ROOT (재학습 안 함)"
echo "  결과                  : $RESULTS_ROOT_BASE"
echo ""

for policy in "${POLICY_LIST[@]}"; do
  export RESULTS_ROOT="${RESULTS_ROOT_BASE}"
  export LOG_DIR="${RESULTS_ROOT_BASE}/logs"
  export RUN_ID="$(date +%Y%m%d_%H%M%S)_${policy}"
  export EXTRA_ARGS="--uncertainty-policy ${policy}"
  echo "------------------------------------------------------------"
  echo "policy=${policy}  RUN_ID=${RUN_ID}"
  echo "------------------------------------------------------------"
  bash "$SWEEP_SCRIPT"
done

echo "Training-policy 스윕 종료"
echo "평가는 evaluation/evaluate_training_policy_sensitivity.py 를 사용하세요."
echo "evaluation/evaluate_uncertainty_endpoint_sensitivity.py 와 섞지 마세요."
