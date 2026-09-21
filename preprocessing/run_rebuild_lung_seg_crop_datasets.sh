#!/usr/bin/env bash
# 폐 세그멘테이션 + 크롭 데이터셋 8종 전체 재생성
#
# 대상 출력 (IEEE_ICCBE/ 아래):
#   cxr_chexmask_lung_seg_pa / _ap
#   cxr_chexmask_lung_seg_cropped_pa / _ap
#   cxr_medsam3_lung_seg_pa / _ap
#   cxr_medsam3_lung_seg_cropped_pa / _ap
#
# 순서: seg 4개 → crop 4개 (crop은 seg manifest 필요)
#
# 사용:
#   python -m pip install -r requirements.txt
#   cd /path/to/pneumonia-lung-preprocessing
#   bash preprocessing/run_rebuild_lung_seg_crop_datasets.sh
#
# 옵션 (환경변수):
#   CLEAN=1              기존 8개 디렉터리 삭제 후 처음부터 생성 (기본: 0, 유지)
#   NO_RESUME=1          --no-resume 로 전체 재처리 (기본: 1)
#   SKIP_SEG=1           seg 단계 건너뜀 (이미 seg만 완료된 경우 crop만)
#   SKIP_CROP=1          crop 단계 건너뜀
#   ONLY=chexmask_pa     특정 조합만 실행 (아래 ONLY 목록 참고)
#   LIMIT=100            디버그용 샘플 수 제한 (0=전체, 기본: 0)
#   DEVICE=cuda          MedSAM3 디바이스 (기본: cuda)
#
# ONLY 값:
#   chexmask_pa, chexmask_ap, medsam3_pa, medsam3_ap
#   chexmask_cropped_pa, chexmask_cropped_ap, medsam3_cropped_pa, medsam3_cropped_ap
#
# 예:
#   CLEAN=1 bash run_rebuild_lung_seg_crop_datasets.sh
#   SKIP_SEG=1 bash run_rebuild_lung_seg_crop_datasets.sh
#   ONLY=medsam3_pa bash run_rebuild_lung_seg_crop_datasets.sh
#   LIMIT=50 bash run_rebuild_lung_seg_crop_datasets.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_BASE_DIR="${DATA_BASE_DIR:-$PACKAGE_ROOT}"
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  PYTHON="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
  echo "ERROR: Python 실행 파일을 찾지 못했습니다. PYTHON=/path/to/python으로 지정하세요." >&2
  exit 1
fi
SEG_PY="${SCRIPT_DIR}/build_cxr_lung_seg_dataset.py"
CROP_PY="${SCRIPT_DIR}/build_cxr_lung_crop_dataset.py"
LABELS_JSON="${LABELS_JSON:-${PACKAGE_ROOT}/pneumonia_labels.json}"
CHEXMASK_CSV="${CHEXMASK_CSV:-${PACKAGE_ROOT}/data/ChexMask_MIMIC-CXR-JPG.csv}"
MIMIC_CXR_ROOT="${MIMIC_CXR_ROOT:-}"

CLEAN="${CLEAN:-0}"
NO_RESUME="${NO_RESUME:-1}"
SKIP_SEG="${SKIP_SEG:-0}"
SKIP_CROP="${SKIP_CROP:-0}"
ONLY="${ONLY:-}"
LIMIT="${LIMIT:-0}"
DEVICE="${DEVICE:-cuda}"

# 출력 디렉터리 (사용자 지정 8종)
CHEXMASK_SEG_PA="${DATA_BASE_DIR}/cxr_chexmask_lung_seg_pa"
CHEXMASK_SEG_AP="${DATA_BASE_DIR}/cxr_chexmask_lung_seg_ap"
MEDSAM3_SEG_PA="${DATA_BASE_DIR}/cxr_medsam3_lung_seg_pa"
MEDSAM3_SEG_AP="${DATA_BASE_DIR}/cxr_medsam3_lung_seg_ap"

CHEXMASK_CROP_PA="${DATA_BASE_DIR}/cxr_chexmask_lung_seg_cropped_pa"
CHEXMASK_CROP_AP="${DATA_BASE_DIR}/cxr_chexmask_lung_seg_cropped_ap"
MEDSAM3_CROP_PA="${DATA_BASE_DIR}/cxr_medsam3_lung_seg_cropped_pa"
MEDSAM3_CROP_AP="${DATA_BASE_DIR}/cxr_medsam3_lung_seg_cropped_ap"

ALL_OUT_DIRS=(
  "$CHEXMASK_SEG_PA"
  "$CHEXMASK_SEG_AP"
  "$MEDSAM3_SEG_PA"
  "$MEDSAM3_SEG_AP"
  "$CHEXMASK_CROP_PA"
  "$CHEXMASK_CROP_AP"
  "$MEDSAM3_CROP_PA"
  "$MEDSAM3_CROP_AP"
)

RESUME_FLAG=()
if [[ "$NO_RESUME" == "1" ]]; then
  RESUME_FLAG=(--no-resume)
fi

LIMIT_FLAG=()
if [[ "$LIMIT" =~ ^[0-9]+$ ]] && [[ "$LIMIT" -gt 0 ]]; then
  LIMIT_FLAG=(--limit "$LIMIT")
fi

should_run () {
  local key="$1"
  [[ -z "$ONLY" || "$ONLY" == "$key" ]]
}

dirs_to_clean=()
if [[ "$SKIP_SEG" != "1" ]]; then
  if should_run chexmask_pa; then dirs_to_clean+=("$CHEXMASK_SEG_PA"); fi
  if should_run chexmask_ap; then dirs_to_clean+=("$CHEXMASK_SEG_AP"); fi
  if should_run medsam3_pa; then dirs_to_clean+=("$MEDSAM3_SEG_PA"); fi
  if should_run medsam3_ap; then dirs_to_clean+=("$MEDSAM3_SEG_AP"); fi
fi
if [[ "$SKIP_CROP" != "1" ]]; then
  if should_run chexmask_cropped_pa; then dirs_to_clean+=("$CHEXMASK_CROP_PA"); fi
  if should_run chexmask_cropped_ap; then dirs_to_clean+=("$CHEXMASK_CROP_AP"); fi
  if should_run medsam3_cropped_pa; then dirs_to_clean+=("$MEDSAM3_CROP_PA"); fi
  if should_run medsam3_cropped_ap; then dirs_to_clean+=("$MEDSAM3_CROP_AP"); fi
fi

log_step () {
  echo ""
  echo "============================================================"
  echo "  $*"
  echo "  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"
}

run_seg () {
  local seg_mode="$1"
  local view="$2"
  local out_root="$3"

  log_step "SEG  seg-mode=${seg_mode}  view=${view}  out=${out_root}"

  local extra=()
  if [[ "$seg_mode" == "chexmask" ]]; then
    extra+=(--chexmask-csv "$CHEXMASK_CSV")
  else
    extra+=(--device "$DEVICE")
  fi
  if [[ -n "$MIMIC_CXR_ROOT" ]]; then
    extra+=(--source-image-root "$MIMIC_CXR_ROOT")
  fi

  "$PYTHON" -u "$SEG_PY" \
    --seg-mode "$seg_mode" \
    --view "$view" \
    --labels-json "$LABELS_JSON" \
    --out-root "$out_root" \
    "${RESUME_FLAG[@]}" \
    "${LIMIT_FLAG[@]}" \
    "${extra[@]}"
}

run_crop () {
  local seg_mode="$1"
  local view="$2"
  local seg_root="$3"
  local out_root="$4"

  local view_tag="pa"
  local seg_tag="medsam3"
  if [[ "$view" == "AP" ]]; then
    view_tag="ap"
  fi
  if [[ "$seg_mode" == "chexmask" ]]; then
    seg_tag="chexmask"
  fi

  local manifest="${seg_root}/manifest_${view_tag}_${seg_tag}_ok.json"
  if [[ ! -f "$manifest" ]]; then
    echo "ERROR: manifest not found: $manifest"
    echo "       seg 단계를 먼저 실행하세요."
    exit 1
  fi

  log_step "CROP seg-mode=${seg_mode}  view=${view}  manifest=${manifest}  out=${out_root}"

  local extra=()
  if [[ "$seg_mode" == "chexmask" ]]; then
    extra+=(--chexmask-csv "$CHEXMASK_CSV")
  fi

  "$PYTHON" -u "$CROP_PY" \
    --seg-mode "$seg_mode" \
    --view "$view" \
    --manifest "$manifest" \
    --out-root "$out_root" \
    "${RESUME_FLAG[@]}" \
    "${LIMIT_FLAG[@]}" \
    "${extra[@]}"
}

echo "Python      : $PYTHON"
echo "Labels JSON : $LABELS_JSON"
echo "ChexMask CSV: $CHEXMASK_CSV"
echo "CLEAN=$CLEAN  NO_RESUME=$NO_RESUME  SKIP_SEG=$SKIP_SEG  SKIP_CROP=$SKIP_CROP  ONLY=${ONLY:-<all>}  LIMIT=$LIMIT"

if [[ ! -f "$LABELS_JSON" ]]; then
  echo "ERROR: labels JSON not found: $LABELS_JSON"
  exit 1
fi

if [[ "$CLEAN" == "1" ]]; then
  log_step "CLEAN: 선택된 출력 디렉터리 삭제"
  for d in "${dirs_to_clean[@]}"; do
    if [[ -d "$d" ]]; then
      echo "  rm -rf $d"
      rm -rf "$d"
    fi
  done
fi

if [[ "$SKIP_SEG" != "1" ]]; then
  if should_run chexmask_pa; then
    run_seg chexmask PA "$CHEXMASK_SEG_PA"
  fi
  if should_run chexmask_ap; then
    run_seg chexmask AP "$CHEXMASK_SEG_AP"
  fi
  if should_run medsam3_pa; then
    run_seg medsam3 PA "$MEDSAM3_SEG_PA"
  fi
  if should_run medsam3_ap; then
    run_seg medsam3 AP "$MEDSAM3_SEG_AP"
  fi
else
  echo ""
  echo "SKIP_SEG=1 → seg 단계 건너뜀"
fi

if [[ "$SKIP_CROP" != "1" ]]; then
  if should_run chexmask_cropped_pa; then
    run_crop chexmask PA "$CHEXMASK_SEG_PA" "$CHEXMASK_CROP_PA"
  fi
  if should_run chexmask_cropped_ap; then
    run_crop chexmask AP "$CHEXMASK_SEG_AP" "$CHEXMASK_CROP_AP"
  fi
  if should_run medsam3_cropped_pa; then
    run_crop medsam3 PA "$MEDSAM3_SEG_PA" "$MEDSAM3_CROP_PA"
  fi
  if should_run medsam3_cropped_ap; then
    run_crop medsam3 AP "$MEDSAM3_SEG_AP" "$MEDSAM3_CROP_AP"
  fi
else
  echo ""
  echo "SKIP_CROP=1 → crop 단계 건너뜀"
fi

echo ""
echo "Done. Rebuild script finished."
echo "Outputs:"
for d in "${ALL_OUT_DIRS[@]}"; do
  if [[ -d "$d" ]]; then
    echo "  [OK] $d"
  else
    echo "  [--] $d  (not created; ONLY/SKIP 설정 확인)"
  fi
done
