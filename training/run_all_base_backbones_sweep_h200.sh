#!/usr/bin/env bash
# H200 1장용 폐렴 분류 base-scale 백본 스윕
#
# H200 기본 최적화:
#   - batch size 64 (모든 백본 공통; BATCH_SIZE로 변경 가능)
#   - learning rate 2e-5 (모든 백본 공통; LEARNING_RATE로 변경 가능)
#   - num_workers 16
#   - torch.compile OFF (H200/Triton 안정성; COMPILE=1 로만 활성화)
#   - BF16 + TF32
#   - CPU 수에 따라 eval/preload worker 자동 설정
#   - 결과 폴더: pneumonia_train.py 자동 naming
#     {timestamp}_{model_name}_{view}_{data_mode}_cv_{N}fold_holdout15pct
#   - RESULTS_ROOT 환경변수로 저장 루트 지정 (기본: ./results_pneumonia)
#   - 모든 전처리에서 같은 환자 split을 쓰도록 raw manifest를 reference로 사용
#
# 기본 범위: AP, PA × 5 data modes × 7 backbones = 70 runs
#
# 기본 실행:
#   bash run_all_base_backbones_sweep_h200.sh
#
# 먼저 1 fold만 점검:
#   MAX_FOLDS=1 bash run_all_base_backbones_sweep_h200.sh
#
# 일부만 실행:
#   VIEW=AP ARCHS="swint_base" DATA_MODES="raw" MAX_FOLDS=1 \
#     bash run_all_base_backbones_sweep_h200.sh
#
# 특정 run부터 시작:
#   START_FROM="AP:chexmask_crop:swint_base" \
#     bash run_all_base_backbones_sweep_h200.sh
#
# 클라우드의 데이터/출력/cache 위치 지정:
#   DATA_BASE_DIR=/data/IEEE_ICCBE \
#   MIMIC_CXR_ROOT=/data/mimic-cxr-jpg/2.1.0 \
#   RESULTS_ROOT=/data/results_pneumonia \
#   NPY_CACHE_DIR=/local_nvme/preload_cache_npy \
#     bash run_all_base_backbones_sweep_h200.sh
#
# CPU worker 직접 지정:
#   NUM_WORKERS=8 EVAL_NUM_WORKERS=4 PRELOAD_WORKERS=24 \
#     bash run_all_base_backbones_sweep_h200.sh
#
# batch size / compile 끄기:
#   BATCH_SIZE=32 COMPILE=0 bash run_all_base_backbones_sweep_h200.sh
#
# Python 해석:
#   1) PYTHON 환경변수 (직접 지정)
#   2) 활성 venv / .venv / PATH의 python3·python
#   PYTHON=/usr/bin/python3 bash run_all_base_backbones_sweep_h200.sh
#
# 실제 학습 없이 명령만 확인:
#   DRY_RUN=1 bash run_all_base_backbones_sweep_h200.sh
#
# 학습 seed 변동성 (split=42 고정, training seed만 123/2026):
#   bash run_seed_variability_sweep_h200.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$STUDY_DIR"

TRAIN_PY="${SCRIPT_DIR}/pneumonia_train.py"
DATA_BASE_DIR="${DATA_BASE_DIR:-$STUDY_DIR}"
MIMIC_CXR_ROOT="${MIMIC_CXR_ROOT:-}"
RESULTS_ROOT="${RESULTS_ROOT:-${STUDY_DIR}/results_pneumonia}"
export RESULTS_ROOT
NPY_CACHE_DIR="${NPY_CACHE_DIR:-${STUDY_DIR}/preload_cache_npy_h200}"
LOG_DIR="${LOG_DIR:-${RESULTS_ROOT}/logs}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ ! -f "$TRAIN_PY" ]]; then
  echo "오류: 학습 파일을 찾을 수 없습니다: $TRAIN_PY" >&2
  exit 1
fi

# MIMIC-CXR 2.1.0 루트 자동 탐색 (raw data-mode용; manifest 절대경로가 다른 머신일 때 재매핑)
resolve_mimic_cxr_root() {
  local candidate seen=""
  local -a candidates=()
  if [[ -n "${MIMIC_CXR_ROOT:-}" ]]; then
    candidates+=("$MIMIC_CXR_ROOT")
  else
    candidates+=(
      "${STUDY_DIR}/../mimic-cxr-jpg/2.1.0"
      "${DATA_BASE_DIR}/../mimic-cxr-jpg/2.1.0"
      "/data/mimic-cxr-jpg/2.1.0"
    )
  fi
  for candidate in "${candidates[@]}"; do
    [[ -z "$candidate" ]] && continue
    candidate="$(cd "$(dirname "$candidate")" 2>/dev/null && pwd)/$(basename "$candidate")" || candidate="$candidate"
    if [[ " $seen " == *" $candidate "* ]]; then
      continue
    fi
    seen="${seen} ${candidate}"
    if [[ -d "$candidate/files" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

# Python: PYTHON / venv / 시스템 python3 순으로 탐색
resolve_python() {
  if [[ -n "${PYTHON:-}" && -x "$PYTHON" ]]; then
    echo "$PYTHON"
    return 0
  fi

  local -a py_candidates=()
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    py_candidates+=("${VIRTUAL_ENV}/bin/python")
  fi
  py_candidates+=(
    "${STUDY_DIR}/.venv/bin/python"
    "${STUDY_DIR}/../.venv/bin/python"
    "${HOME}/.local/bin/python3"
    "/usr/bin/python3"
  )
  if command -v python3 >/dev/null 2>&1; then
    py_candidates+=("$(command -v python3)")
  fi
  if command -v python >/dev/null 2>&1; then
    py_candidates+=("$(command -v python)")
  fi

  local seen="" py
  for py in "${py_candidates[@]}"; do
    [[ -z "$py" || ! -x "$py" ]] && continue
    if [[ " $seen " == *" $py "* ]]; then
      continue
    fi
    seen="${seen} ${py}"
    echo "$py"
    return 0
  done

  echo "오류: Python 실행 파일을 찾지 못했습니다." >&2
  echo "PYTHON=/path/to/python 으로 지정하거나, python3/venv를 PATH에 추가하세요." >&2
  return 1
}

PYTHON="$(resolve_python)" || exit 1
export PYTHON

CPU_COUNT="${CPU_COUNT:-$(nproc)}"
if [[ ! "$CPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "오류: CPU_COUNT는 1 이상의 정수여야 합니다: $CPU_COUNT" >&2
  exit 1
fi

# eval/preload worker는 CPU 수에 따라 자동 설정합니다.
_eval_workers=$((CPU_COUNT / 4))
(( _eval_workers < 2 )) && _eval_workers="$CPU_COUNT"
(( _eval_workers > 8 )) && _eval_workers=8

_preload_workers="$CPU_COUNT"
(( _preload_workers > 32 )) && _preload_workers=32

NUM_WORKERS="${NUM_WORKERS:-16}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-$_eval_workers}"
PRELOAD_WORKERS="${PRELOAD_WORKERS:-$_preload_workers}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

for worker_setting in \
  "NUM_WORKERS:$NUM_WORKERS" \
  "EVAL_NUM_WORKERS:$EVAL_NUM_WORKERS" \
  "PRELOAD_WORKERS:$PRELOAD_WORKERS" \
  "PREFETCH_FACTOR:$PREFETCH_FACTOR"
do
  _name="${worker_setting%%:*}"
  _value="${worker_setting#*:}"
  if [[ ! "$_value" =~ ^[0-9]+$ ]]; then
    echo "오류: ${_name}은 0 이상의 정수여야 합니다: $_value" >&2
    exit 1
  fi
done

MAX_FOLDS="${MAX_FOLDS:-5}"
OUTER_TEST_RATIO="${OUTER_TEST_RATIO:-0.15}"
USE_SPLIT_REFERENCE="${USE_SPLIT_REFERENCE:-1}"
SPLIT_DIR="${SPLIT_DIR:-${STUDY_DIR}/splits}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TRAINING_SEED="${TRAINING_SEED:-42}"

USE_FIXED_SPLITS=0
if [[ -d "$SPLIT_DIR" \
      && -f "$SPLIT_DIR/AP_outer_test.csv" \
      && -f "$SPLIT_DIR/AP_fold_assignment.csv" \
      && -f "$SPLIT_DIR/PA_outer_test.csv" \
      && -f "$SPLIT_DIR/PA_fold_assignment.csv" ]]; then
  USE_FIXED_SPLITS=1
fi
PRELOAD_IMAGES="${PRELOAD_IMAGES:-1}"
COMPILE="${COMPILE:-0}"
DRY_RUN="${DRY_RUN:-0}"
CHECK_GPU="${CHECK_GPU:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
START_FROM="${START_FROM:-}"

if [[ ! "$MAX_FOLDS" =~ ^[1-5]$ ]]; then
  echo "오류: MAX_FOLDS는 1~5여야 합니다: $MAX_FOLDS" >&2
  exit 1
fi
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "오류: BATCH_SIZE는 1 이상의 정수여야 합니다: $BATCH_SIZE" >&2
  exit 1
fi
if (( PREFETCH_FACTOR < 1 )); then
  echo "오류: PREFETCH_FACTOR는 1 이상이어야 합니다: $PREFETCH_FACTOR" >&2
  exit 1
fi
if [[ "$PRELOAD_IMAGES" == "1" && "$PRELOAD_WORKERS" == "0" ]]; then
  echo "오류: PRELOAD_IMAGES=1일 때 PRELOAD_WORKERS는 1 이상이어야 합니다." >&2
  exit 1
fi

_view_input="${VIEW:-both}"
_view_input="${_view_input^^}"
case "$_view_input" in
  AP) VIEWS=(AP) ;;
  PA) VIEWS=(PA) ;;
  BOTH|ALL|"") VIEWS=(AP PA) ;;
  *)
    echo "오류: VIEW는 AP, PA, both 중 하나여야 합니다: ${VIEW:-}" >&2
    exit 1
    ;;
esac

read -r -a DATA_MODE_LIST <<< \
  "${DATA_MODES:-${DATA_MODE:-raw medsam3_seg medsam3_crop chexmask_seg chexmask_crop}}"
read -r -a ARCH_LIST <<< \
  "${ARCHS:-${ARCH:-resnet152 densenet convnextv2_base swint_base dinov3_base eva_x_base rad_dino}}"
read -r -a EXTRA_ARGS_ARRAY <<< "$EXTRA_ARGS"

if (( ${#DATA_MODE_LIST[@]} == 0 || ${#ARCH_LIST[@]} == 0 )); then
  echo "오류: DATA_MODES와 ARCHS는 비어 있을 수 없습니다." >&2
  exit 1
fi

_needs_raw_root=0
for _mode in "${DATA_MODE_LIST[@]}"; do
  if [[ "$_mode" == "raw" ]]; then
    _needs_raw_root=1
    break
  fi
done

if (( _needs_raw_root == 1 )); then
  if _resolved_mimic="$(resolve_mimic_cxr_root)"; then
    if [[ -z "${MIMIC_CXR_ROOT:-}" ]]; then
      MIMIC_CXR_ROOT="$_resolved_mimic"
      echo "ℹ️ MIMIC_CXR_ROOT 자동 탐지: $MIMIC_CXR_ROOT"
    elif [[ ! -d "$MIMIC_CXR_ROOT/files" ]]; then
      echo "오류: MIMIC_CXR_ROOT에 files/ 폴더가 없습니다: $MIMIC_CXR_ROOT" >&2
      echo "예: MIMIC_CXR_ROOT=/path/to/mimic-cxr-jpg/2.1.0" >&2
      exit 1
    fi
  else
    echo "오류: data mode 'raw'에는 MIMIC-CXR 원본 경로가 필요합니다." >&2
    echo "manifest의 source_image_abs_path가 이 서버에 없습니다." >&2
    echo "MIMIC_CXR_ROOT=/path/to/mimic-cxr-jpg/2.1.0 bash run_all_base_backbones_sweep_h200.sh" >&2
    exit 1
  fi
fi

data_root_for() {
  local view="$1"
  local mode="$2"
  local suffix
  suffix="${view,,}"

  case "$mode" in
    raw|medsam3_seg)
      printf '%s/cxr_medsam3_lung_seg_%s\n' "$DATA_BASE_DIR" "$suffix"
      ;;
    medsam3_crop)
      printf '%s/cxr_medsam3_lung_seg_cropped_%s\n' "$DATA_BASE_DIR" "$suffix"
      ;;
    chexmask_seg)
      printf '%s/cxr_chexmask_lung_seg_%s\n' "$DATA_BASE_DIR" "$suffix"
      ;;
    chexmask_crop)
      printf '%s/cxr_chexmask_lung_seg_cropped_%s\n' "$DATA_BASE_DIR" "$suffix"
      ;;
    medsam3_center|medsam3_margin0|medsam3_margin10|medsam3_margin20|medsam3_soft)
      printf '%s/cxr_medsam3_control_%s_%s\n' "$DATA_BASE_DIR" "${mode#medsam3_}" "$suffix"
      ;;
    *)
      echo "오류: 지원하지 않는 data mode입니다: $mode" >&2
      return 1
      ;;
  esac
}

manifest_for() {
  local root="$1"
  local view="$2"
  local mode="$3"
  local seg_tag="medsam3"
  [[ "$mode" == chexmask_* ]] && seg_tag="chexmask"
  printf '%s/manifest_%s_%s_ok.json\n' "$root" "${view,,}" "$seg_tag"
}

# 시작 전에 폴더와 manifest가 복사되었는지 확인합니다.
for view in "${VIEWS[@]}"; do
  for mode in "${DATA_MODE_LIST[@]}"; do
    root="$(data_root_for "$view" "$mode")"
    manifest="$(manifest_for "$root" "$view" "$mode")"
    if [[ ! -d "$root" ]]; then
      echo "오류: 데이터 폴더를 찾을 수 없습니다: $root" >&2
      echo "클라우드 경로가 다르면 DATA_BASE_DIR을 지정하세요." >&2
      exit 1
    fi
    if [[ ! -f "$manifest" ]]; then
      echo "오류: manifest를 찾을 수 없습니다: $manifest" >&2
      exit 1
    fi
  done
done

if [[ "$CHECK_GPU" == "1" && "$DRY_RUN" != "1" ]]; then
  "$PYTHON" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("오류: CUDA GPU를 찾지 못했습니다.")
count = torch.cuda.device_count()
if count != 1:
    raise SystemExit(
        f"오류: 이 스크립트는 GPU 1장용입니다. 현재 보이는 GPU 수: {count}. "
        "CUDA_VISIBLE_DEVICES를 확인하세요."
    )
name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
print(f"GPU 확인: {name}, VRAM={props.total_memory / 1024**3:.1f} GiB")
if "H200" not in name.upper():
    print(f"주의: H200이 아닌 GPU로 보입니다: {name}", file=sys.stderr)
if not torch.cuda.is_bf16_supported():
    raise SystemExit("오류: 현재 GPU/PyTorch 조합에서 BF16을 지원하지 않습니다.")
PY
fi

mkdir -p "$RESULTS_ROOT" "$LOG_DIR"
if [[ "$PRELOAD_IMAGES" == "1" ]]; then
  mkdir -p "$NPY_CACHE_DIR"
fi

run_count=$((${#VIEWS[@]} * ${#DATA_MODE_LIST[@]} * ${#ARCH_LIST[@]}))
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

echo ""
echo "H200 1-GPU 스윕 설정"
echo "  계획된 runs       : $run_count"
echo "  views              : ${VIEWS[*]}"
echo "  data modes         : ${DATA_MODE_LIST[*]}"
echo "  backbones          : ${ARCH_LIST[*]}"
echo "  folds              : $MAX_FOLDS / 5"
echo "  precision          : BF16 (TF32 허용)"
echo "  batch size         : $BATCH_SIZE (모든 백본 공통)"
echo "  learning rate      : $LEARNING_RATE (모든 백본 공통)"
echo "  Python             : $PYTHON"
echo "  CPU cores          : $CPU_COUNT"
echo "  workers            : train=$NUM_WORKERS, eval=$EVAL_NUM_WORKERS, preload=$PRELOAD_WORKERS"
echo "  prefetch factor    : $PREFETCH_FACTOR"
echo "  preload cache      : $NPY_CACHE_DIR"
echo "  data base          : $DATA_BASE_DIR"
echo "  raw CXR root       : ${MIMIC_CXR_ROOT:-(raw 미사용 — medsam3/chexmask manifest 경로 재매핑)}"
echo "  results            : $RESULTS_ROOT (train.py auto naming)"
echo "  split reference    : $USE_SPLIT_REFERENCE"
echo "  fixed split dir    : ${SPLIT_DIR} (enabled=$USE_FIXED_SPLITS)"
echo "  split seed         : $SPLIT_SEED"
echo "  training seed      : $TRAINING_SEED"
echo "  torch.compile      : $COMPILE"
[[ -n "$START_FROM" ]] && echo "  start from         : $START_FROM"
[[ "$DRY_RUN" == "1" ]] && echo "  mode               : DRY RUN"
echo ""

start_reached=1
if [[ -n "$START_FROM" ]]; then
  start_reached=0
fi

completed=0
for view in "${VIEWS[@]}"; do
  reference_root="$(data_root_for "$view" raw)"

  for mode in "${DATA_MODE_LIST[@]}"; do
    data_root="$(data_root_for "$view" "$mode")"

    for arch in "${ARCH_LIST[@]}"; do
      run_key="${view}:${mode}:${arch}"

      if (( start_reached == 0 )); then
        if [[ "$run_key" == "$START_FROM" ]]; then
          start_reached=1
        else
          echo "건너뜀: $run_key"
          continue
        fi
      fi

      log_file="${LOG_DIR}/${run_id}_${view}_${mode}_${arch}.log"

      cmd=(
        "$PYTHON" "$TRAIN_PY"
        --arch "$arch"
        --view "$view"
        --data-mode "$mode"
        --data-root "$data_root"
        --split-mode cv5
        --n-folds 5
        --max-folds "$MAX_FOLDS"
        --outer-test-ratio "$OUTER_TEST_RATIO"
        --num-workers "$NUM_WORKERS"
        --eval-num-workers "$EVAL_NUM_WORKERS"
        --prefetch-factor "$PREFETCH_FACTOR"
        --preload-workers "$PRELOAD_WORKERS"
        --batch-size "$BATCH_SIZE"
        --learning-rate "$LEARNING_RATE"
        --bf16
        --split-seed "$SPLIT_SEED"
        --seed "$TRAINING_SEED"
      )

      if [[ "$USE_FIXED_SPLITS" == "1" ]]; then
        cmd+=(--split-dir "$SPLIT_DIR")
      elif [[ "$USE_SPLIT_REFERENCE" == "1" ]]; then
        cmd+=(
          --split-reference-data-root "$reference_root"
          --split-reference-data-mode raw
        )
      fi
      if [[ -n "$MIMIC_CXR_ROOT" ]]; then
        cmd+=(--source-image-root "$MIMIC_CXR_ROOT")
      fi
      if [[ "$PRELOAD_IMAGES" == "1" ]]; then
        cmd+=(--npy-cache-dir "$NPY_CACHE_DIR")
      else
        cmd+=(--no-preload-images)
      fi
      if [[ "$COMPILE" == "1" ]]; then
        cmd+=(--compile)
      else
        cmd+=(--no-compile)
      fi
      if (( ${#EXTRA_ARGS_ARRAY[@]} > 0 )); then
        cmd+=("${EXTRA_ARGS_ARRAY[@]}")
      fi

      echo "[$((completed + 1))/$run_count] 시작: $run_key"
      printf '명령:'
      printf ' %q' "${cmd[@]}"
      printf '\n'

      if [[ "$DRY_RUN" != "1" ]]; then
        "${cmd[@]}" 2>&1 | tee "$log_file"
      fi

      completed=$((completed + 1))
      echo "완료: $run_key"
      echo ""
    done
  done
done

if (( start_reached == 0 )); then
  echo "오류: START_FROM에 해당하는 run을 찾지 못했습니다: $START_FROM" >&2
  exit 1
fi

echo "전체 스윕 종료: 실행한 run 수=$completed"
