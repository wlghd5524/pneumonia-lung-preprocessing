#!/usr/bin/env bash
# Build lung masks/crops for Pneumonia=-1 images of one fixed-split scope.
#   SCOPE=outer_test (default): evaluation-endpoint set -> cxr_*_uncertain_{ap,pa}
#   SCOPE=trainval            : training-policy set    -> cxr_*_uncertain_trainval_{ap,pa}
# Writes to *uncertain* directories so the original 0/1 cohort is not overwritten.
#
# Training-policy analysis (Raw + MedSAM3 crop only):
#   SCOPE=trainval SKIP_CHEXMASK=1 bash preprocessing/run_uncertain_endpoint_preprocess.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$STUDY_DIR"

PYTHON="${PYTHON:-/usr/bin/python3}"
SCOPE="${SCOPE:-outer_test}"
case "$SCOPE" in
  outer_test) DIR_TAG="uncertain" ;;
  trainval)   DIR_TAG="uncertain_trainval" ;;
  *) echo "ERROR: SCOPE must be outer_test or trainval: $SCOPE"; exit 1 ;;
esac
LABELS_JSON="${LABELS_JSON:-${STUDY_DIR}/pneumonia_labels_uncertain_${SCOPE}.json}"
CHEXMASK_CSV="${CHEXMASK_CSV:-${STUDY_DIR}/ChexMask_MIMIC-CXR-JPG.csv}"
SOURCE_ROOT="${SOURCE_ROOT:-${MIMIC_CXR_ROOT:-${STUDY_DIR}/data/mimic-cxr-jpg/2.1.0}}"
DEVICE="${DEVICE:-cuda}"
ONLY="${ONLY:-}"
SKIP_CHEXMASK="${SKIP_CHEXMASK:-0}"
SKIP_MEDSAM3="${SKIP_MEDSAM3:-0}"
SKIP_CROP="${SKIP_CROP:-0}"

SEG_PY="${SCRIPT_DIR}/build_cxr_lung_seg_dataset.py"
CROP_PY="${SCRIPT_DIR}/build_cxr_lung_crop_dataset.py"

should_run () {
  local key="$1"
  [[ -z "$ONLY" || "$ONLY" == "$key" ]]
}

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
  log_step "SEG  ${seg_mode} ${view} -> ${out_root}"
  local extra=()
  if [[ "$seg_mode" == "chexmask" ]]; then
    extra+=(--chexmask-csv "$CHEXMASK_CSV")
  else
    extra+=(--device "$DEVICE")
  fi
  "$PYTHON" -u "$SEG_PY" \
    --seg-mode "$seg_mode" \
    --view "$view" \
    --labels-json "$LABELS_JSON" \
    --out-root "$out_root" \
    --source-image-root "$SOURCE_ROOT" \
    --keep-uncertain \
    --resume \
    "${extra[@]}"
}

run_crop () {
  local seg_mode="$1"
  local view="$2"
  local seg_root="$3"
  local out_root="$4"
  local view_tag="pa"
  local seg_tag="medsam3"
  if [[ "$view" == "AP" ]]; then view_tag="ap"; fi
  if [[ "$seg_mode" == "chexmask" ]]; then seg_tag="chexmask"; fi
  local manifest="${seg_root}/manifest_${view_tag}_${seg_tag}_ok.json"
  log_step "CROP ${seg_mode} ${view} <- ${manifest}"
  local extra=()
  if [[ "$seg_mode" == "chexmask" ]]; then
    extra+=(--chexmask-csv "$CHEXMASK_CSV")
  fi
  "$PYTHON" -u "$CROP_PY" \
    --seg-mode "$seg_mode" \
    --view "$view" \
    --manifest "$manifest" \
    --out-root "$out_root" \
    --resume \
    "${extra[@]}"
}

echo "Python      : $PYTHON"
echo "Labels JSON : $LABELS_JSON"
echo "ONLY=${ONLY:-<all>} SKIP_CHEXMASK=$SKIP_CHEXMASK SKIP_MEDSAM3=$SKIP_MEDSAM3 SKIP_CROP=$SKIP_CROP"

if [[ ! -f "$LABELS_JSON" ]]; then
  echo "ERROR: labels JSON not found: $LABELS_JSON"
  exit 1
fi
case "$(basename "$LABELS_JSON")" in
  pneumonia_labels_uncertain_trainval.json)
    if [[ "$SCOPE" != "trainval" ]]; then
      echo "ERROR: trainval 라벨은 SCOPE=trainval 로만 처리해야 합니다. 지금 SCOPE=$SCOPE 이면 평가용 uncertain 폴더를 덮습니다." >&2
      exit 1
    fi
    ;;
  pneumonia_labels_uncertain_outer_test.json)
    if [[ "$SCOPE" != "outer_test" ]]; then
      echo "ERROR: outer-test 라벨은 SCOPE=outer_test 로만 처리해야 합니다. 지금 SCOPE=$SCOPE" >&2
      exit 1
    fi
    ;;
esac

if [[ "$SKIP_CHEXMASK" != "1" ]]; then
  if should_run chexmask_ap; then
    run_seg chexmask AP "${STUDY_DIR}/cxr_chexmask_lung_seg_uncertain_ap"
  fi
  if should_run chexmask_pa; then
    run_seg chexmask PA "${STUDY_DIR}/cxr_chexmask_lung_seg_uncertain_pa"
  fi
  if [[ "$SKIP_CROP" != "1" ]]; then
    if should_run chexmask_ap; then
      run_crop chexmask AP "${STUDY_DIR}/cxr_chexmask_lung_seg_uncertain_ap" \
        "${STUDY_DIR}/cxr_chexmask_lung_seg_cropped_uncertain_ap"
    fi
    if should_run chexmask_pa; then
      run_crop chexmask PA "${STUDY_DIR}/cxr_chexmask_lung_seg_uncertain_pa" \
        "${STUDY_DIR}/cxr_chexmask_lung_seg_cropped_uncertain_pa"
    fi
  fi
fi

if [[ "$SKIP_MEDSAM3" != "1" ]]; then
  if should_run medsam3_ap; then
    run_seg medsam3 AP "${STUDY_DIR}/cxr_medsam3_lung_seg_uncertain_ap"
  fi
  if should_run medsam3_pa; then
    run_seg medsam3 PA "${STUDY_DIR}/cxr_medsam3_lung_seg_uncertain_pa"
  fi
  if [[ "$SKIP_CROP" != "1" ]]; then
    if should_run medsam3_ap; then
      run_crop medsam3 AP "${STUDY_DIR}/cxr_medsam3_lung_seg_uncertain_ap" \
        "${STUDY_DIR}/cxr_medsam3_lung_seg_cropped_uncertain_ap"
    fi
    if should_run medsam3_pa; then
      run_crop medsam3 PA "${STUDY_DIR}/cxr_medsam3_lung_seg_uncertain_pa" \
        "${STUDY_DIR}/cxr_medsam3_lung_seg_cropped_uncertain_pa"
    fi
  fi
fi

echo ""
echo "DONE $(date '+%Y-%m-%d %H:%M:%S')"
