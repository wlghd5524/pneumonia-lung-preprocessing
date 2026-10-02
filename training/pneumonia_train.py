"""
단일 CXR(흉부 X선 한 장) 기반 폐렴(Pneumonia) 이진 분류 학습
(ResNet / DenseNet / ConvNeXtV2 / SwinV2 / DINOv3 / EVA-X /
 RAD-DINO / CheXficient), 5-fold CV
- 환자(subject) 단위 그룹으로 나누어 데이터 누수를 막음 (manifest의 subject_id)
- MedSAM3로 폐 영역만 세그멘테이션한 마스크드 CXR 이미지 사용
- Weighted BCE (pos_weight = N_negative / N_positive, fold별) + validation AUROC 기준 best checkpoint
- 최종 evaluation: AUROC + AUPRC 모두 보고
- **기본 분할**: 15% holdout outer test + 나머지 85% 로 5-fold CV (IEEE/RSNA 권장)
    OOF 에서 threshold 를 결정해 독립 holdout 에 단 한 번 앙상블 평가 →
    threshold 순환 편향 제거, bootstrap 95% CI 포함한 최종 수치를 ``outer_test/`` 에 저장.
    순수 5-fold 만 원하면 ``--no-outer-test`` (또는 ``--outer-test-ratio 0.0``).
    holdout 사용 시 결과 폴더명에 ``…_cv_Nfold_holdout15pct`` 처럼 비율이 붙음.
- 설정: JSON config와 CLI 인자 (명시한 CLI가 우선)
- 기본 데이터: ``IEEE_ICCBE/cxr_medsam3_lung_seg`` (manifest_pa_medsam3_ok.json + masked_cxr_jpg 경로)
- --view AP/both 로 AP·PA+AP 영상도 지원 (cxr_*_lung_seg_ap / cxr_*_lung_seg_pa_ap 폴더 자동 선택)
- 라벨: ``IEEE_ICCBE/pneumonia_labels.json`` 에서 pneumonia 필드 (0=정상, 1=폐렴)
- 멀티 GPU DDP 자동 재실행: **기본 꺼짐** (단일 ``python`` 프로세스, 보통 첫 GPU만 사용). 켜려면 ``--auto-ddp`` 또는 ``CXR_SINGLE_AUTO_DDP=1``. 수동으로는 ``torchrun --nproc_per_node=N ...`` 사용 가능.
"""

import os

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import timm
import numpy as np
import random
import argparse
import sys
import hashlib
import subprocess
import time
from tqdm import tqdm
# matplotlib 백엔드를 'Agg'로 설정 (GUI 없이 파일 저장만 가능, 멀티스레딩 환경에서 안전)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
    matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, GroupShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta
import csv
import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', message='.*is a low-contrast image.*')

# 이 스크립트 파일의 위치를 기준으로 프로젝트 루트를 계산합니다.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.dirname(SCRIPT_DIR)          # 저장소 루트 (코드, splits, configs, eva-x)
PROJECT_ROOT = STUDY_DIR                           # EVA-X 가중치/로더는 저장소 안의 eva-x/
# 전처리된 데이터셋(cxr_*), 라벨 JSON의 위치. 환경변수 DATA_BASE_DIR로 바꿀 수 있습니다.
DATA_BASE_DIR = os.path.abspath(os.environ.get("DATA_BASE_DIR", STUDY_DIR))
def default_eva_x_ckpt_path(model_name: str) -> str:
    """레포 루트 ``eva-x/<이름>.pt`` 가중치 경로. ``get_evax_backbone``은 파일 경로를 그대로 받음."""
    _fname = {
        'eva_x_tiny': 'eva_x_tiny_patch16_merged520k_mim.pt',
        'eva_x_small': 'eva_x_small_patch16_merged520k_mim.pt',
        'eva_x_base': 'eva_x_base_patch16_merged520k_mim.pt',
    }.get(str(model_name), 'eva_x_small_patch16_merged520k_mim.pt')
    return os.path.normpath(os.path.join(PROJECT_ROOT, 'eva-x', _fname))


# EVA-X: Python 로더와 가중치 .pt 모두 ``eva-x/``
get_evax_backbone = None  # type: ignore
EVA_X_KNOWN_NAMES = ('eva_x_tiny', 'eva_x_small', 'eva_x_base')
EVA_X_NAMES = EVA_X_KNOWN_NAMES
for _eva_dir in (
    os.path.join(PROJECT_ROOT, 'eva-x'),
):
    if not os.path.isdir(_eva_dir):
        continue
    if _eva_dir not in sys.path:
        sys.path.insert(0, _eva_dir)
    try:
        from eva_x import get_evax_backbone, EVA_X_REGISTRY  # noqa: WPS433
        EVA_X_NAMES = tuple(EVA_X_REGISTRY.keys())
        break
    except Exception:
        get_evax_backbone = None
        EVA_X_NAMES = EVA_X_KNOWN_NAMES
        continue

RAD_DINO_HF_ID = 'microsoft/rad-dino'
RAD_DINO_INPUT_SIZE = 512
# RAD-DINO: 다른 backbone과 동일한 512px로 맞춤 (HF native 518 → resize).
# config에서 norm_mean/norm_std를 명시하면 그 값이 우선합니다.
RAD_DINO_NORM_MEAN = [0.485, 0.456, 0.406]
RAD_DINO_NORM_STD = [0.229, 0.224, 0.225]
RAD_DINO_ALIASES = {'rad_dino', 'rad-dino', RAD_DINO_HF_ID}

CHEXFICIENT_HF_ID = 'StanfordAIMI/CheXficient'
CHEXFICIENT_INPUT_SIZE = 378
CHEXFICIENT_NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
CHEXFICIENT_NORM_STD = [0.26862954, 0.26130258, 0.27577711]
CHEXFICIENT_ALIASES = {
    'chexficient',
    'stanfordaimi/chexficient',
    CHEXFICIENT_HF_ID.lower(),
}


def _is_rad_dino_model(model_name: str) -> bool:
    return str(model_name or '').strip().lower() in RAD_DINO_ALIASES


def _canonical_rad_dino_model_name(model_name: str) -> str:
    name = str(model_name or '').strip()
    return RAD_DINO_HF_ID if name.lower() in {'rad_dino', 'rad-dino'} else name


def _is_chexficient_model(model_name: str) -> bool:
    return str(model_name or '').strip().lower() in CHEXFICIENT_ALIASES


def _canonical_chexficient_model_name(model_name: str) -> str:
    name = str(model_name or '').strip()
    if name.lower() in {'chexficient', 'stanfordaimi/chexficient'}:
        return CHEXFICIENT_HF_ID
    return name

# Optional Weights & Biases
try:
    import wandb
    _WANDB_OK = True
except Exception:
    wandb = None
    _WANDB_OK = False


# ----------------------
# Dataset: 단일 CXR (MedSAM3 폐 세그멘테이션 마스크드 이미지)
# ----------------------


def load_all_cxr_samples_with_groups(
    data_root: str,
    strict_grouping: bool = True,
    labels_json_path: Optional[str] = None,
    data_mode: str = "medsam3_seg",
    view_tag: str = "pa",
    source_image_root: Optional[str] = None,
    require_files: bool = True,
    return_metadata: bool = False,
):
    """
    manifest JSON에서 CXR 경로·폐렴 라벨·그룹을 읽습니다.

    data_mode에 따라 사용할 이미지 경로 필드가 결정되고,
    view_tag(pa / ap / pa_ap)와 data_mode로 manifest 파일명이 결정됩니다:

    ┌──────────────┬───────────────────────────────────┬──────────────────────────────────────────────────┐
    │ data_mode    │ 이미지 경로 필드                  │ manifest 파일명 (view_tag=pa 기준 예시)          │
    ├──────────────┼───────────────────────────────────┼──────────────────────────────────────────────────┤
    │ raw          │ source_image_abs_path (원본 CXR)  │ manifest_{view_tag}_medsam3_ok.json              │
    │ medsam3_seg  │ masked_cxr_jpg  (MedSAM3 폐마스크)│ manifest_{view_tag}_medsam3_ok.json              │
    │ medsam3_crop │ cropped_cxr_jpg (MedSAM3 크롭)   │ manifest_{view_tag}_medsam3_ok.json              │
    │ chexmask_seg │ masked_cxr_jpg  (ChexMask 폐마스크│ manifest_{view_tag}_chexmask_ok.json             │
    │ chexmask_crop│ cropped_cxr_jpg (ChexMask 크롭)  │ manifest_{view_tag}_chexmask_ok.json             │
    │ medsam3_center / medsam3_margin0 / medsam3_margin10 / medsam3_margin20                        │
    │              │ cropped_cxr_jpg                  │ manifest_{view_tag}_medsam3_ok.json              │
    │ medsam3_soft │ masked_cxr_jpg                   │ manifest_{view_tag}_medsam3_ok.json              │
    └──────────────┴───────────────────────────────────┴──────────────────────────────────────────────────┘

    view_tag 값 예:
      "pa"    → --view PA (기본)
      "ap"    → --view AP
      "pa_ap" → --view both

    (하위 호환: ``masked`` → ``medsam3_seg``, ``cropped`` → ``medsam3_crop`` 자동 변환)

    - ``pneumonia``: 0(정상) 또는 1(폐렴)
    - ``subject_id``: 환자 단위 그룹 (데이터 누수 방지)
    - ``labels_json_path``: pneumonia_labels.json 경로. manifest에 라벨이 없는 항목의 보완용.
    - ``source_image_root``: 클라우드 이전 후 raw 원본의 새 루트. manifest의
      source_image_abs_path가 더 이상 유효하지 않을 때 source_image_rel_path와 결합합니다.
    - ``require_files``: False이면 이미지 파일 존재 여부를 검사하지 않습니다.
      split reference 계산처럼 manifest의 subject_id·라벨만 필요할 때 사용합니다.
    반환 ``all_items`` 원소: ``(image_path, label, ref)`` — ref는 배치/시각화용(보통 image_path).
    """
    # 하위 호환 alias 정규화
    _ALIAS = {"masked": "medsam3_seg", "cropped": "medsam3_crop"}
    data_mode = _ALIAS.get(data_mode, data_mode)

    _MODE_FIELD = {
        "raw":          "source_image_abs_path",
        "medsam3_seg":  "masked_cxr_jpg",
        "medsam3_crop": "cropped_cxr_jpg",
        "chexmask_seg": "masked_cxr_jpg",
        "chexmask_crop":"cropped_cxr_jpg",
        "medsam3_center":   "cropped_cxr_jpg",
        "medsam3_margin0":  "cropped_cxr_jpg",
        "medsam3_margin10": "cropped_cxr_jpg",
        "medsam3_margin20": "cropped_cxr_jpg",
        "medsam3_soft":     "masked_cxr_jpg",
    }
    # manifest seg tag: chexmask 계열은 "chexmask", 나머지는 "medsam3"
    _MODE_SEG_TAG = {
        "raw":          "medsam3",
        "medsam3_seg":  "medsam3",
        "medsam3_crop": "medsam3",
        "chexmask_seg": "chexmask",
        "chexmask_crop":"chexmask",
        "medsam3_center":   "medsam3",
        "medsam3_margin0":  "medsam3",
        "medsam3_margin10": "medsam3",
        "medsam3_margin20": "medsam3",
        "medsam3_soft":     "medsam3",
    }
    if data_mode not in _MODE_FIELD:
        raise ValueError(
            f"data_mode는 'raw' / 'medsam3_seg' / 'medsam3_crop' / "
            f"'chexmask_seg' / 'chexmask_crop' / "
            f"'medsam3_center' / 'medsam3_margin0' / 'medsam3_margin10' / "
            f"'medsam3_margin20' / 'medsam3_soft' 중 하나여야 합니다: {data_mode!r}"
        )
    img_field = _MODE_FIELD[data_mode]
    seg_tag = _MODE_SEG_TAG[data_mode]
    manifest_name = f"manifest_{view_tag}_{seg_tag}_ok.json"

    data_root_n = os.path.normpath(str(data_root))
    manifest_path = os.path.join(data_root_n, manifest_name)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"{manifest_name}을 찾을 수 없습니다: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    label_lookup: Dict[str, int] = {}
    if labels_json_path and os.path.isfile(labels_json_path):
        with open(labels_json_path, "r", encoding="utf-8") as f:
            labels_data = json.load(f)
        for le in labels_data:
            did = le.get("dicom_id")
            pn = le.get("pneumonia")
            if did and pn is not None:
                label_lookup[str(did)] = int(pn)

    items: List[Tuple[Any, ...]] = []
    labels: List[int] = []
    group_ids: List[str] = []
    metadata: List[Dict[str, Any]] = []
    skipped_no_label = 0
    skipped_no_file = 0
    skipped_no_field = 0
    remapped_paths = 0
    source_image_root_n = (
        os.path.normpath(str(source_image_root)) if source_image_root else None
    )
    for e in entries:
        img_path_raw = e.get(img_field) or ""
        if not img_path_raw:
            skipped_no_field += 1
            continue
        p = str(img_path_raw)
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(data_root_n, p.lstrip("/")))
        if require_files:
            if not os.path.isfile(p):
                remapped = None
                if img_field == "source_image_abs_path" and source_image_root_n:
                    source_rel = str(e.get("source_image_rel_path") or "").strip()
                    if source_rel:
                        candidate = os.path.normpath(
                            os.path.join(source_image_root_n, source_rel.lstrip("/\\"))
                        )
                        if os.path.isfile(candidate):
                            remapped = candidate
                elif img_field in {"masked_cxr_jpg", "cropped_cxr_jpg"}:
                    # 가공 이미지 manifest에는 이전 컴퓨터의 절대경로가 남아 있을 수 있습니다.
                    # masked_cxr/ 또는 cropped_cxr/ 이하의 상대경로를 현재 data_root에 붙입니다.
                    # crop 계열은 data_root/files/... 구조인 경우도 있어 /files/ fallback을 둡니다.
                    normalized_path = str(img_path_raw).replace("\\", "/")
                    marker_candidates = (
                        ["masked_cxr"] if img_field == "masked_cxr_jpg"
                        else ["cropped_cxr", "files"]
                    )
                    for marker in marker_candidates:
                        marker_token = f"/{marker}/"
                        if marker_token not in normalized_path:
                            continue
                        relative_tail = normalized_path.split(marker_token, 1)[1]
                        candidate = os.path.normpath(
                            os.path.join(data_root_n, marker, relative_tail)
                        )
                        if os.path.isfile(candidate):
                            remapped = candidate
                            break
                if remapped is not None:
                    p = remapped
                    remapped_paths += 1
            if not os.path.isfile(p):
                skipped_no_file += 1
                continue
        elif not p:
            p = str(e.get("dicom_id") or f"ref_{len(items)}")

        pn_label = e.get("pneumonia")
        if pn_label is None:
            dicom_id = str(e.get("dicom_id", ""))
            pn_label = label_lookup.get(dicom_id)
        if pn_label is None:
            skipped_no_label += 1
            continue
        y = int(pn_label)

        sid = e.get("subject_id")
        if sid is None:
            if strict_grouping:
                raise RuntimeError(f"manifest 항목에 subject_id가 없습니다: {p}")
            gid = os.path.basename(p)
        else:
            gid = f"sub{int(sid)}"
        items.append((p, y, p))
        labels.append(y)
        group_ids.append(gid)
        metadata.append({
            "dicom_id": str(e.get("dicom_id") or ""),
            "subject_id": int(sid) if sid is not None else None,
            "label": y,
            "group_id": gid,
        })

    if skipped_no_field > 0:
        print(f"ℹ️ 데이터 로드 스킵: '{img_field}' 필드 없음={skipped_no_field} (data_mode={data_mode!r})")
    if remapped_paths > 0:
        print(f"☁️ 클라우드 경로 재매핑: {remapped_paths}개 ({data_mode})")
    if skipped_no_label > 0 or (require_files and skipped_no_file > 0):
        print(f"ℹ️ 데이터 로드 스킵: 라벨 없음={skipped_no_label}, 파일 없음={skipped_no_file}")
    if not items:
        raise RuntimeError(
            f"유효한 CXR 샘플이 없습니다. "
            f"manifest({manifest_name}) 경로와 '{img_field}' 필드를 확인하세요 "
            f"(data_mode={data_mode!r}, data_root={data_root_n})"
        )
    if return_metadata:
        return items, labels, group_ids, metadata
    return items, labels, group_ids


def build_grouped_split_assignment(
    items,
    labels,
    groups,
    n_folds: int,
    outer_test_ratio: float,
    seed: int,
):
    """Return subject-group assignments for outer holdout and CV folds."""
    use_outer_test = float(outer_test_ratio) > 0.0
    all_indices = np.arange(len(items))
    outer_method = None

    if use_outer_test:
        outer_n_splits = max(2, int(round(1.0 / max(1e-6, float(outer_test_ratio)))))
        try:
            splitter = StratifiedGroupKFold(
                n_splits=outer_n_splits,
                shuffle=True,
                random_state=int(seed),
            )
            trainval_idx, test_idx = next(iter(splitter.split(items, labels, groups)))
            outer_method = f"StratifiedGroupKFold(n_splits={outer_n_splits})"
        except Exception:
            splitter = GroupShuffleSplit(
                n_splits=1,
                train_size=1.0 - float(outer_test_ratio),
                random_state=int(seed),
            )
            trainval_idx, test_idx = next(splitter.split(items, labels, groups=groups))
            outer_method = "GroupShuffleSplit(n_splits=1)"
    else:
        trainval_idx = all_indices
        test_idx = np.array([], dtype=np.int64)

    trainval_idx = np.asarray(trainval_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    cv_items = [items[int(i)] for i in trainval_idx]
    cv_labels = [labels[int(i)] for i in trainval_idx]
    cv_groups = [groups[int(i)] for i in trainval_idx]

    cv_splitter = StratifiedGroupKFold(
        n_splits=int(n_folds),
        shuffle=True,
        random_state=int(seed),
    )
    folds = []
    for train_rel, val_rel in cv_splitter.split(cv_items, cv_labels, cv_groups):
        folds.append({
            "train_groups": sorted({cv_groups[int(i)] for i in train_rel}),
            "val_groups": sorted({cv_groups[int(i)] for i in val_rel}),
        })

    return {
        "outer_method": outer_method,
        "outer_trainval_groups": sorted({groups[int(i)] for i in trainval_idx}),
        "outer_test_groups": sorted({groups[int(i)] for i in test_idx}),
        "folds": folds,
        "reference_n_images": int(len(items)),
        "reference_n_subjects": int(len(set(groups))),
    }


def map_grouped_split_assignment(all_groups, assignment):
    """Map a reference subject assignment onto the samples loaded for this run."""
    active_groups = set(all_groups)
    reference_groups = (
        set(assignment["outer_trainval_groups"])
        | set(assignment["outer_test_groups"])
    )
    unknown = sorted(active_groups - reference_groups)
    if unknown:
        raise RuntimeError(
            "현재 데이터에 reference split에 없는 subject가 있습니다: "
            f"{len(unknown)}명, 예={unknown[:10]}"
        )

    def indices_for(group_values):
        wanted = set(group_values)
        return np.asarray(
            [i for i, group in enumerate(all_groups) if group in wanted],
            dtype=np.int64,
        )

    outer_trainval_idx = indices_for(assignment["outer_trainval_groups"])
    outer_test_idx = indices_for(assignment["outer_test_groups"])
    fold_indices = [
        (indices_for(fold["train_groups"]), indices_for(fold["val_groups"]))
        for fold in assignment["folds"]
    ]
    return outer_trainval_idx, outer_test_idx, fold_indices


SPLIT_CSV_COLUMNS = ("subject_id", "dicom_id", "label", "view", "split", "fold")


def _group_id_from_subject(subject_id) -> str:
    return f"sub{int(subject_id)}"


def _split_csv_paths(split_dir: str, view: str) -> Tuple[str, str]:
    view_key = str(view).upper()
    split_dir_n = os.path.normpath(str(split_dir))
    return (
        os.path.join(split_dir_n, f"{view_key}_outer_test.csv"),
        os.path.join(split_dir_n, f"{view_key}_fold_assignment.csv"),
    )


def load_manifest_split_records(
    data_root: str,
    *,
    labels_json_path: Optional[str] = None,
    data_mode: str = "raw",
    view_tag: str = "pa",
    strict_grouping: bool = True,
) -> Tuple[List[Dict[str, Any]], str]:
    """Manifest에서 split CSV용 메타데이터만 읽습니다 (이미지 파일 검사 없음)."""
    _ALIAS = {"masked": "medsam3_seg", "cropped": "medsam3_crop"}
    data_mode = _ALIAS.get(data_mode, data_mode)
    _MODE_SEG_TAG = {
        "raw": "medsam3",
        "medsam3_seg": "medsam3",
        "medsam3_crop": "medsam3",
        "chexmask_seg": "chexmask",
        "chexmask_crop": "chexmask",
    }
    if data_mode not in _MODE_SEG_TAG:
        raise ValueError(f"Unsupported data_mode for split export: {data_mode!r}")
    seg_tag = _MODE_SEG_TAG[data_mode]
    manifest_name = f"manifest_{view_tag}_{seg_tag}_ok.json"
    data_root_n = os.path.normpath(str(data_root))
    manifest_path = os.path.join(data_root_n, manifest_name)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"{manifest_name}을 찾을 수 없습니다: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    label_lookup: Dict[str, int] = {}
    if labels_json_path and os.path.isfile(labels_json_path):
        with open(labels_json_path, "r", encoding="utf-8") as f:
            labels_data = json.load(f)
        for le in labels_data:
            did = le.get("dicom_id")
            pn = le.get("pneumonia")
            if did and pn is not None:
                label_lookup[str(did)] = int(pn)

    records: List[Dict[str, Any]] = []
    for e in entries:
        pn_label = e.get("pneumonia")
        if pn_label is None:
            dicom_id = str(e.get("dicom_id", ""))
            pn_label = label_lookup.get(dicom_id)
        if pn_label is None:
            continue

        sid = e.get("subject_id")
        if sid is None:
            if strict_grouping:
                raise RuntimeError(
                    f"manifest 항목에 subject_id가 없습니다: dicom_id={e.get('dicom_id')}"
                )
            gid = str(e.get("dicom_id") or len(records))
            subject_id_val = None
        else:
            subject_id_val = int(sid)
            gid = _group_id_from_subject(subject_id_val)

        records.append({
            "subject_id": subject_id_val,
            "dicom_id": str(e.get("dicom_id") or ""),
            "label": int(pn_label),
            "group_id": gid,
        })

    if not records:
        raise RuntimeError(
            f"split export용 manifest 레코드가 없습니다: {manifest_path} "
            f"(data_mode={data_mode!r})"
        )
    return records, manifest_path


def build_image_split_table_rows(
    records: List[Dict[str, Any]],
    assignment: Dict[str, Any],
    view: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Group assignment → image-level outer_test / fold_assignment CSV rows."""
    outer_test_groups = set(assignment["outer_test_groups"])
    group_to_val_fold: Dict[str, int] = {}
    for fold_idx, fold in enumerate(assignment["folds"], start=1):
        for group_id in fold["val_groups"]:
            group_to_val_fold[group_id] = fold_idx

    outer_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    view_key = str(view).upper()
    for rec in records:
        group_id = rec["group_id"]
        base = {
            "subject_id": rec["subject_id"],
            "dicom_id": rec["dicom_id"],
            "label": rec["label"],
            "view": view_key,
        }
        if group_id in outer_test_groups:
            outer_rows.append({**base, "split": "outer_test", "fold": ""})
        elif group_id in group_to_val_fold:
            fold_rows.append({
                **base,
                "split": "trainval",
                "fold": group_to_val_fold[group_id],
            })
    return outer_rows, fold_rows


def _write_split_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SPLIT_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            fold_val = out.get("fold")
            if fold_val == "" or fold_val is None:
                out["fold"] = ""
            else:
                out["fold"] = int(fold_val)
            writer.writerow(out)


def save_fixed_split_csvs(
    *,
    split_dir: str,
    view: str,
    records: List[Dict[str, Any]],
    assignment: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Write {VIEW}_outer_test.csv and {VIEW}_fold_assignment.csv under split_dir."""
    outer_rows, fold_rows = build_image_split_table_rows(records, assignment, view)
    outer_path, fold_path = _split_csv_paths(split_dir, view)
    _write_split_csv(outer_path, outer_rows)
    _write_split_csv(fold_path, fold_rows)
    if meta is not None:
        meta_path = os.path.join(
            os.path.normpath(str(split_dir)),
            f"{str(view).upper()}_split_meta.json",
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    return outer_path, fold_path


def _read_split_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Split CSV not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_assignment_from_split_csvs(
    split_dir: str,
    view: str,
    *,
    n_folds: int,
) -> Dict[str, Any]:
    """Rebuild grouped split assignment from precomputed CSV files."""
    outer_path, fold_path = _split_csv_paths(split_dir, view)
    outer_rows = _read_split_csv(outer_path)
    fold_rows = _read_split_csv(fold_path)

    outer_test_groups = sorted({
        _group_id_from_subject(row["subject_id"])
        for row in outer_rows
        if row.get("subject_id") not in (None, "")
    })
    group_to_fold: Dict[str, int] = {}
    for row in fold_rows:
        sid = row.get("subject_id")
        fold_raw = row.get("fold")
        if sid in (None, "") or fold_raw in (None, ""):
            continue
        group_to_fold[_group_id_from_subject(sid)] = int(fold_raw)

    trainval_groups = sorted(group_to_fold.keys())
    if not trainval_groups:
        raise RuntimeError(f"trainval 그룹이 비어 있습니다: {fold_path}")

    overlap = set(outer_test_groups).intersection(trainval_groups)
    if overlap:
        raise RuntimeError(
            "outer_test와 trainval subject가 겹칩니다: "
            f"{sorted(overlap)[:10]}"
        )

    folds = []
    for fold_idx in range(1, int(n_folds) + 1):
        val_groups = sorted(g for g, f in group_to_fold.items() if f == fold_idx)
        train_groups = sorted(set(trainval_groups) - set(val_groups))
        folds.append({"train_groups": train_groups, "val_groups": val_groups})

    missing_folds = [i + 1 for i, fold in enumerate(folds) if not fold["val_groups"]]
    if missing_folds:
        raise RuntimeError(
            f"fold assignment CSV에 validation 그룹이 없는 fold가 있습니다: {missing_folds}"
        )

    return {
        "outer_method": "fixed_csv",
        "outer_trainval_groups": trainval_groups,
        "outer_test_groups": outer_test_groups,
        "folds": folds,
        "reference_n_images": int(len(outer_rows) + len(fold_rows)),
        "reference_n_subjects": int(len(set(outer_test_groups) | set(trainval_groups))),
        "split_csv": {
            "outer_test": os.path.normpath(outer_path),
            "fold_assignment": os.path.normpath(fold_path),
        },
    }


PREPROCESSING_MODES: Tuple[str, ...] = (
    "raw",
    "medsam3_seg",
    "medsam3_crop",
    "chexmask_seg",
    "chexmask_crop",
)
# 본 실험 5종과 같은 환자 목록의 추가 대조. 학습 시에만 parity 검사에 포함합니다.
CONTROL_PREPROCESSING_MODES: Tuple[str, ...] = (
    "medsam3_center",
    "medsam3_margin0",
    "medsam3_margin10",
    "medsam3_margin20",
    "medsam3_soft",
)


def infer_data_base_dir(data_root: str) -> str:
    """``cxr_*_{ap,pa}`` 데이터 폴더의 상위 IEEE_ICCBE 루트를 추정합니다."""
    return os.path.dirname(os.path.normpath(str(data_root)))


def data_root_for_preprocessing(data_base_dir: str, view: str, data_mode: str) -> str:
    """스윕 스크립트와 동일한 규칙으로 view/mode별 data_root를 반환합니다."""
    suffix = str(view).lower()
    mode = str(data_mode).lower()
    base = os.path.normpath(str(data_base_dir))
    if mode in ("raw", "medsam3_seg"):
        return os.path.join(base, f"cxr_medsam3_lung_seg_{suffix}")
    if mode == "medsam3_crop":
        return os.path.join(base, f"cxr_medsam3_lung_seg_cropped_{suffix}")
    if mode == "chexmask_seg":
        return os.path.join(base, f"cxr_chexmask_lung_seg_{suffix}")
    if mode == "chexmask_crop":
        return os.path.join(base, f"cxr_chexmask_lung_seg_cropped_{suffix}")
    if mode in CONTROL_PREPROCESSING_MODES:
        tag = mode[len("medsam3_"):]
        return os.path.join(base, f"cxr_medsam3_control_{tag}_{suffix}")
    raise ValueError(f"Unsupported preprocessing mode: {data_mode!r}")


# CheXpert training-time uncertainty policies (Irvin et al., 2019).
# explicit_only는 기존 0/1 학습과 동일하고, u_zero/u_one만 -1을 학습 라벨로 쓴다.
UNCERTAINTY_POLICIES: Tuple[str, ...] = ("explicit_only", "u_zero", "u_one")
UNCERTAINTY_POLICY_LABEL: Dict[str, int] = {"u_zero": 0, "u_one": 1}


def uncertain_trainval_data_root(data_base_dir: str, view: str, data_mode: str) -> str:
    """trainval 환자 Pneumonia=-1 영상의 전처리별 data_root.

    ``finalize_uncertain_trainval_cohort.py``가 이 경로의 manifest를 교집합으로
    고정하므로, 학습도 같은 규칙을 써야 Raw와 crop이 같은 영상을 추가한다.
    """
    view_tag = str(view).lower()
    mode = str(data_mode).lower()
    base = os.path.normpath(str(data_base_dir))
    folder = {
        "raw": f"cxr_medsam3_lung_seg_uncertain_trainval_{view_tag}",
        "medsam3_seg": f"cxr_medsam3_lung_seg_uncertain_trainval_{view_tag}",
        "medsam3_crop": f"cxr_medsam3_lung_seg_cropped_uncertain_trainval_{view_tag}",
        "chexmask_seg": f"cxr_chexmask_lung_seg_uncertain_trainval_{view_tag}",
        "chexmask_crop": f"cxr_chexmask_lung_seg_cropped_uncertain_trainval_{view_tag}",
    }
    if mode not in folder:
        raise ValueError(
            f"uncertainty training policy는 raw/medsam3/chexmask 전처리만 지원합니다: {data_mode!r}"
        )
    return os.path.join(base, folder[mode])


def load_policy_training_additions(
    *,
    uncertain_root: str,
    cohort_csv: str,
    data_mode: str,
    view_tag: str,
    mapped_label: int,
    source_image_root: Optional[str],
    strict_grouping: bool,
) -> Tuple[List[Tuple[Any, ...]], List[int], List[str], Dict[str, int]]:
    """고정 cohort CSV의 -1 영상만 읽어 학습 라벨로 바꿉니다.

    cohort에 없는 영상(outer-test 환자, 분할 밖 환자)은 넣지 않고,
    cohort 영상이 하나라도 없으면 학습을 멈춥니다.
    """
    cohort_rows = _read_split_csv(cohort_csv)
    cohort: Dict[str, Dict[str, int]] = {}
    for row in cohort_rows:
        dicom_id = str(row.get("dicom_id") or "")
        if not dicom_id:
            continue
        if dicom_id in cohort:
            raise RuntimeError(f"uncertainty cohort CSV에 dicom_id가 중복됩니다: {dicom_id}")
        label = int(row["label"])
        if label != -1:
            raise RuntimeError(f"uncertainty cohort CSV 라벨은 -1이어야 합니다: {dicom_id}={label}")
        cohort[dicom_id] = {
            "subject_id": int(row["subject_id"]),
            "fold": int(row["fold"]),
        }
    if not cohort:
        raise RuntimeError(f"uncertainty cohort CSV가 비어 있습니다: {cohort_csv}")

    items, labels, groups, metadata = load_all_cxr_samples_with_groups(
        uncertain_root,
        strict_grouping=strict_grouping,
        labels_json_path=None,
        data_mode=data_mode,
        view_tag=view_tag,
        source_image_root=source_image_root,
        require_files=True,
        return_metadata=True,
    )
    by_dicom: Dict[str, Tuple[Any, str]] = {}
    for item, label, group, meta in zip(items, labels, groups, metadata):
        dicom_id = str(meta.get("dicom_id") or "")
        if dicom_id not in cohort:
            continue
        if int(label) != -1:
            raise RuntimeError(
                f"uncertainty manifest 라벨이 -1이 아닙니다: {dicom_id}={label} ({uncertain_root})"
            )
        spec = cohort[dicom_id]
        subject_id = meta.get("subject_id")
        if subject_id is None or int(subject_id) != spec["subject_id"]:
            raise RuntimeError(
                f"uncertainty 영상의 subject_id가 cohort CSV와 다릅니다: {dicom_id}"
            )
        if group != _group_id_from_subject(spec["subject_id"]):
            raise RuntimeError(f"uncertainty 영상의 group_id가 subject와 다릅니다: {dicom_id}")
        by_dicom[dicom_id] = (item, group)

    missing = sorted(set(cohort) - set(by_dicom))
    if missing:
        raise RuntimeError(
            f"cohort CSV 영상 {len(missing)}장이 {uncertain_root}에 없습니다. 예: {missing[:5]}"
        )

    out_items: List[Tuple[Any, ...]] = []
    out_labels: List[int] = []
    out_groups: List[str] = []
    by_fold: Dict[str, int] = {}
    for dicom_id, spec in cohort.items():
        item, group = by_dicom[dicom_id]
        out_items.append((item[0], int(mapped_label), item[2]))
        out_labels.append(int(mapped_label))
        out_groups.append(group)
        fold_key = str(spec["fold"])
        by_fold[fold_key] = by_fold.get(fold_key, 0) + 1
    return out_items, out_labels, out_groups, by_fold


def _metadata_to_sample_identity(metadata: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    identity: Dict[str, Dict[str, Any]] = {}
    for row in metadata:
        dicom_id = str(row.get("dicom_id") or "")
        if not dicom_id:
            continue
        identity[dicom_id] = {
            "subject_id": row.get("subject_id"),
            "label": int(row["label"]),
            "group_id": str(row.get("group_id") or ""),
        }
    return identity


def load_preprocessing_sample_identity(
    data_root: str,
    *,
    data_mode: str,
    view_tag: str,
    labels_json_path: Optional[str] = None,
    source_image_root: Optional[str] = None,
    strict_grouping: bool = True,
    require_files: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """학습 loader와 동일한 inclusion rule로 dicom_id → sample identity dict."""
    _, _, _, metadata = load_all_cxr_samples_with_groups(
        data_root,
        strict_grouping=strict_grouping,
        labels_json_path=labels_json_path,
        data_mode=data_mode,
        view_tag=view_tag,
        source_image_root=source_image_root,
        require_files=require_files,
        return_metadata=True,
    )
    return _metadata_to_sample_identity(metadata)


def load_dicom_split_table(split_dir: str, view: str) -> Dict[str, Dict[str, Any]]:
    """Fixed split CSV를 dicom_id → {subject_id, label, split, fold} dict로 읽습니다."""
    outer_path, fold_path = _split_csv_paths(split_dir, view)
    table: Dict[str, Dict[str, Any]] = {}
    for row in _read_split_csv(outer_path):
        dicom_id = str(row.get("dicom_id") or "")
        if not dicom_id:
            continue
        table[dicom_id] = {
            "subject_id": int(row["subject_id"]),
            "label": int(row["label"]),
            "split": str(row["split"]),
            "fold": "",
        }
    for row in _read_split_csv(fold_path):
        dicom_id = str(row.get("dicom_id") or "")
        if not dicom_id:
            continue
        table[dicom_id] = {
            "subject_id": int(row["subject_id"]),
            "label": int(row["label"]),
            "split": str(row["split"]),
            "fold": int(row["fold"]),
        }
    return table


def _subject_split_maps_from_dicom_table(
    split_table: Dict[str, Dict[str, Any]],
) -> Tuple[set, Dict[int, int]]:
    outer_subjects: set = set()
    subject_fold: Dict[int, int] = {}
    for row in split_table.values():
        sid = int(row["subject_id"])
        if row["split"] == "outer_test":
            outer_subjects.add(sid)
        elif row["split"] == "trainval":
            subject_fold[sid] = int(row["fold"])
    return outer_subjects, subject_fold


def assert_preprocessing_sample_parity(
    *,
    view: str,
    data_base_dir: str,
    split_dir: Optional[str],
    labels_json_path: Optional[str] = None,
    source_image_root: Optional[str] = None,
    strict_grouping: bool = True,
    current_data_mode: Optional[str] = None,
    current_n_samples: Optional[int] = None,
    require_files: bool = False,
) -> None:
    """
    5 preprocessing condition이 동일한 sample·label·split을 갖는지 assert합니다.

    기본(require_files=False)은 manifest 메타데이터 기준으로 dicom/subject/label을 비교합니다.
    fixed split CSV도 같은 기준으로 생성되므로, 전처리 간 sample identity 검증에 적합합니다.
    현재 학습 mode가 주어지면 require_files=True 로 실제 로드 가능 sample 수도 추가 검사합니다.

    검사 항목:
      - dicom_id set
      - subject_id set
      - dicom_id별 label / subject_id
      - fixed split CSV의 outer_test / fold assignment (split_dir 제공 시)
    """
    view_key = str(view).upper()
    view_tag = view_key.lower()
    if view_key not in ("AP", "PA"):
        return

    modes = list(PREPROCESSING_MODES)
    extra_mode = str(current_data_mode or "").lower()
    if extra_mode and extra_mode not in modes:
        modes.append(extra_mode)

    identities: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for mode in modes:
        mode_root = data_root_for_preprocessing(data_base_dir, view_key, mode)
        identities[mode] = load_preprocessing_sample_identity(
            mode_root,
            data_mode=mode,
            view_tag=view_tag,
            labels_json_path=labels_json_path,
            source_image_root=source_image_root,
            strict_grouping=strict_grouping,
            require_files=require_files,
        )

    ref_mode = "raw"
    ref_identity = identities[ref_mode]
    ref_dicom_ids = set(ref_identity.keys())
    ref_subject_ids = {int(v["subject_id"]) for v in ref_identity.values() if v["subject_id"] is not None}

    for mode in modes:
        if mode == ref_mode:
            continue
        cur = identities[mode]
        cur_dicom_ids = set(cur.keys())
        assert cur_dicom_ids == ref_dicom_ids, (
            f"[{view_key}] preprocessing parity failed: dicom_id set mismatch "
            f"between {ref_mode} ({len(ref_dicom_ids)}) and {mode} ({len(cur_dicom_ids)}). "
            f"only_in_{ref_mode}={sorted(ref_dicom_ids - cur_dicom_ids)[:5]}, "
            f"only_in_{mode}={sorted(cur_dicom_ids - ref_dicom_ids)[:5]}"
        )
        cur_subject_ids = {int(v["subject_id"]) for v in cur.values() if v["subject_id"] is not None}
        assert cur_subject_ids == ref_subject_ids, (
            f"[{view_key}] preprocessing parity failed: subject_id set mismatch "
            f"between {ref_mode} ({len(ref_subject_ids)}) and {mode} ({len(cur_subject_ids)}). "
            f"only_in_{ref_mode}={sorted(ref_subject_ids - cur_subject_ids)[:5]}, "
            f"only_in_{mode}={sorted(cur_subject_ids - ref_subject_ids)[:5]}"
        )
        label_mismatches = [
            dicom_id
            for dicom_id in ref_dicom_ids
            if int(cur[dicom_id]["label"]) != int(ref_identity[dicom_id]["label"])
        ]
        assert not label_mismatches, (
            f"[{view_key}] preprocessing parity failed: label mismatch between "
            f"{ref_mode} and {mode} for dicom_id examples={label_mismatches[:5]}"
        )
        subject_mismatches = [
            dicom_id
            for dicom_id in ref_dicom_ids
            if int(cur[dicom_id]["subject_id"]) != int(ref_identity[dicom_id]["subject_id"])
        ]
        assert not subject_mismatches, (
            f"[{view_key}] preprocessing parity failed: subject_id mismatch between "
            f"{ref_mode} and {mode} for dicom_id examples={subject_mismatches[:5]}"
        )

    if split_dir:
        split_table = load_dicom_split_table(split_dir, view_key)
        split_dicom_ids = set(split_table.keys())
        assert split_dicom_ids == ref_dicom_ids, (
            f"[{view_key}] fixed split CSV dicom_id set mismatch with preprocessing manifests: "
            f"csv={len(split_dicom_ids)}, raw={len(ref_dicom_ids)}, "
            f"only_in_csv={sorted(split_dicom_ids - ref_dicom_ids)[:5]}, "
            f"only_in_raw={sorted(ref_dicom_ids - split_dicom_ids)[:5]}"
        )
        csv_label_mismatches = [
            dicom_id
            for dicom_id in ref_dicom_ids
            if int(split_table[dicom_id]["label"]) != int(ref_identity[dicom_id]["label"])
        ]
        assert not csv_label_mismatches, (
            f"[{view_key}] split CSV label mismatch vs raw manifest: "
            f"examples={csv_label_mismatches[:5]}"
        )
        csv_subject_mismatches = [
            dicom_id
            for dicom_id in ref_dicom_ids
            if int(split_table[dicom_id]["subject_id"]) != int(ref_identity[dicom_id]["subject_id"])
        ]
        assert not csv_subject_mismatches, (
            f"[{view_key}] split CSV subject_id mismatch vs raw manifest: "
            f"examples={csv_subject_mismatches[:5]}"
        )

        outer_subjects, subject_fold = _subject_split_maps_from_dicom_table(split_table)
        for mode in modes:
            mode_outer = {
                int(identities[mode][dicom_id]["subject_id"])
                for dicom_id in ref_dicom_ids
                if split_table[dicom_id]["split"] == "outer_test"
            }
            assert mode_outer == outer_subjects, (
                f"[{view_key}] outer_test subject set mismatch for mode={mode}: "
                f"expected={len(outer_subjects)}, got={len(mode_outer)}"
            )
            mode_subject_fold = {}
            for dicom_id in ref_dicom_ids:
                row = split_table[dicom_id]
                if row["split"] != "trainval":
                    continue
                sid = int(identities[mode][dicom_id]["subject_id"])
                mode_subject_fold[sid] = int(row["fold"])
            assert mode_subject_fold == subject_fold, (
                f"[{view_key}] fold assignment mismatch for mode={mode}: "
                f"examples={sorted(set(mode_subject_fold.items()) ^ set(subject_fold.items()))[:5]}"
            )

    if current_data_mode is not None:
        mode_key = str(current_data_mode).lower()
        cur_identity = identities.get(mode_key)
        assert cur_identity is not None, (
            f"[{view_key}] unknown current_data_mode for parity check: {current_data_mode!r}"
        )
        if current_n_samples is not None and not require_files:
            mode_root = data_root_for_preprocessing(data_base_dir, view_key, mode_key)
            file_identity = load_preprocessing_sample_identity(
                mode_root,
                data_mode=mode_key,
                view_tag=view_tag,
                labels_json_path=labels_json_path,
                source_image_root=source_image_root,
                strict_grouping=strict_grouping,
                require_files=True,
            )
            assert len(file_identity) == int(current_n_samples), (
                f"[{view_key}] current run sample count mismatch for mode={current_data_mode}: "
                f"loaded={current_n_samples}, file-backed={len(file_identity)}"
            )
            assert set(file_identity.keys()).issubset(set(cur_identity.keys())), (
                f"[{view_key}] current run dicom_ids not subset of manifest identity for "
                f"mode={current_data_mode}"
            )
            for dicom_id, info in file_identity.items():
                assert cur_identity[dicom_id]["label"] == info["label"]
                assert cur_identity[dicom_id]["subject_id"] == info["subject_id"]
        elif current_n_samples is not None:
            assert len(cur_identity) == int(current_n_samples), (
                f"[{view_key}] current run sample count mismatch for mode={current_data_mode}: "
                f"loaded={current_n_samples}, identity={len(cur_identity)}"
            )

    check_label = "manifest" if not require_files else "file-backed"
    print(
        f"✅ [{view_key}] preprocessing parity OK ({check_label}): "
        f"{len(ref_dicom_ids)} dicom_ids / {len(ref_subject_ids)} subjects across "
        f"{len(modes)} modes"
        + (f"; fixed split CSV verified ({split_dir})" if split_dir else "")
    )


class SingleCXRDataset(Dataset):
    """단일 CXR(JPG 한 장)을 읽어 텐서로 만듭니다.
    all_items: (path, y, ref) 3-tuple.
    """
    def __init__(self, all_items, indices, input_size=512, train=True,
                 norm_mean=None, norm_std=None,
                 flip_prob=0.5, rot_prob=0.3, rot_deg=10,
                 bc_prob=0.3, brightness=(0.8, 1.2), contrast=(0.8, 1.2),
                 shift_prob=0.0, shift_limit=0.05,
                 scale_prob=0.0, scale_limit=0.1,
                 preloaded_images: Optional[Dict[str, np.ndarray]] = None,
                 split_name="train",
                 **_kwargs):
        super().__init__()
        self.input_size = int(input_size)
        self.items = [all_items[i] for i in indices]
        self.train_mode = bool(train)
        self.preloaded_images = preloaded_images
        self.mean = norm_mean if norm_mean is not None else [0.485, 0.456, 0.406]
        self.std = norm_std if norm_std is not None else [0.229, 0.224, 0.225]
        self.flip_prob = float(flip_prob)
        self.rot_prob = float(rot_prob)
        self.rot_deg = float(rot_deg)
        self.bc_prob = float(bc_prob)
        self.brightness = tuple(brightness)
        self.contrast = tuple(contrast)
        self.shift_prob = float(shift_prob)
        self.shift_limit = float(shift_limit)
        self.scale_prob = float(scale_prob)
        self.scale_limit = float(scale_limit)
        self.split_name = str(split_name)

    def __len__(self):
        return len(self.items)

    def _safe_open(self, path):
        if self.preloaded_images is not None and path in self.preloaded_images:
            return Image.fromarray(self.preloaded_images[path])
        try:
            img = Image.open(path)
            img.load()
            return img
        except Exception:
            return None

    def _transform_img(self, img, train=True):
        img = img.convert('RGB')
        if img.size != (self.input_size, self.input_size):
            img = img.resize((self.input_size, self.input_size), resample=Image.BILINEAR)
        if train:
            if random.random() < self.flip_prob:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < self.rot_prob:
                angle = random.uniform(-self.rot_deg, self.rot_deg)
                img = img.rotate(angle, expand=False, fillcolor=(0, 0, 0))
            if random.random() < self.shift_prob:
                tx = random.uniform(-self.shift_limit, self.shift_limit) * self.input_size
                ty = random.uniform(-self.shift_limit, self.shift_limit) * self.input_size
                img = img.transform(img.size, Image.AFFINE,
                                    (1, 0, -tx, 0, 1, -ty),
                                    resample=Image.BILINEAR, fillcolor=(0, 0, 0))
            if random.random() < self.scale_prob:
                s = random.uniform(1.0 - self.scale_limit, 1.0 + self.scale_limit)
                cx = self.input_size / 2.0
                cy = self.input_size / 2.0
                inv_s = 1.0 / s
                img = img.transform(img.size, Image.AFFINE,
                                    (inv_s, 0, cx * (1 - inv_s), 0, inv_s, cy * (1 - inv_s)),
                                    resample=Image.BILINEAR, fillcolor=(0, 0, 0))
            if random.random() < self.bc_prob:
                from PIL import ImageEnhance
                brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
                contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
                img = ImageEnhance.Brightness(img).enhance(brightness_factor)
                img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        img_tensor = transforms.functional.to_tensor(img)
        img_tensor = transforms.functional.normalize(img_tensor, self.mean, self.std)
        return img_tensor

    def __getitem__(self, idx):
        row = self.items[idx]
        img_path = row[0]
        y = row[1]
        img = self._safe_open(img_path)
        if img is None:
            return {'__drop__': True, 'split': self.split_name, 'label': int(y), 'reason': 'image_load_failed'}
        img_tensor = self._transform_img(img, train=self.train_mode)
        return img_tensor, int(y), img_path


def _preload_worker(args):
    """멀티프로세스 preload worker — (path, input_size) → (path, array | None)."""
    p, input_size = args
    try:
        img = Image.open(p)
        img.load()
        img = img.convert('RGB').resize((int(input_size), int(input_size)), resample=Image.BILINEAR)
        return p, np.asarray(img, dtype=np.uint8).copy()
    except Exception:
        return p, None


def preload_all_images_single(
    all_items,
    input_size: int,
    rank: int = 0,
    npy_cache_dir: Optional[str] = None,
    num_workers: int = 16,
) -> Dict[str, np.ndarray]:
    """메인 프로세스에서 모든 마스크드 CXR 이미지를 미리 로드 (uint8 H×W×3, fork worker와 공유).

    ``load_all_cxr_samples_with_groups`` 가 반환한 ``(image_path, label, ref)`` 리스트에서
    고유 ``image_path`` 만 수집합니다.

    npy_cache_dir 가 지정되면 ``preload_<hash>.npz`` 파일로 캐시를 저장/재사용합니다.
    캐시 키는 (정렬된 경로 목록 + input_size) 의 MD5 해시이므로, 데이터셋이나 해상도가
    바뀌면 자동으로 새 캐시가 생성됩니다.
    """
    unique_paths = {str(item[0]) for item in all_items}
    paths_list = sorted(unique_paths)

    # ── NPY 디스크 캐시: 기존 캐시 파일이 있으면 바로 로드 ──────────────────
    cache_npz: Optional[str] = None
    if npy_cache_dir:
        os.makedirs(npy_cache_dir, exist_ok=True)
        key_bytes = (f"{input_size}:" + ":".join(paths_list)).encode("utf-8")
        cache_hash = hashlib.md5(key_bytes).hexdigest()[:16]
        cache_npz = os.path.join(npy_cache_dir, f"preload_{cache_hash}.npz")

        if os.path.isfile(cache_npz):
            if rank == 0:
                print(f"💾 NPY 캐시 발견 → 로드 중: {cache_npz}")
            try:
                data = np.load(cache_npz, allow_pickle=True)
                saved_paths: np.ndarray = data["paths"]    # (N,) object array of str
                saved_arrays: np.ndarray = data["arrays"]  # (N, H, W, 3) uint8
                cache: Dict[str, np.ndarray] = {
                    str(saved_paths[i]): saved_arrays[i]
                    for i in range(len(saved_paths))
                }
                if rank == 0:
                    mem_mb = sum(a.nbytes for a in cache.values()) / (1024 * 1024)
                    print(f"✅ NPY 캐시 로드 완료: {len(cache)}장 ({mem_mb:.1f} MB)")
                return cache
            except Exception as e:
                if rank == 0:
                    print(f"⚠️ NPY 캐시 로드 실패 ({e}), 원본 이미지에서 재로드합니다 ...")

    # ── 원본 이미지에서 직접 preload ─────────────────────────────────────────
    cache = {}
    failed = 0

    if rank == 0:
        print(f"🖼️ Preloading {len(paths_list)} images at {input_size}×{input_size} "
              f"into memory (workers={num_workers}) ...")

    from concurrent.futures import ProcessPoolExecutor, as_completed

    args_list = [(p, input_size) for p in paths_list]
    with ProcessPoolExecutor(max_workers=int(num_workers)) as ex:
        futures = {ex.submit(_preload_worker, a): a[0] for a in args_list}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Preloading images", disable=(rank != 0)):
            p, arr = fut.result()
            if arr is not None:
                cache[p] = arr
            else:
                failed += 1

    if rank == 0:
        mem_mb = sum(a.nbytes for a in cache.values()) / (1024 * 1024)
        print(f"✅ Preloaded {len(cache)}/{len(paths_list)} images ({mem_mb:.1f} MB), {failed} failed")

    # ── NPY 캐시 저장 (rank 0 만, 백그라운드 스레드로 수행) ──────────────────
    # savez_compressed는 수만 장 압축 시 수 분이 걸릴 수 있어서, 메인 프로세스에서
    # 직접 수행하면 DDP barrier timeout을 유발한다. 별도 스레드에서 비동기 저장.
    if cache_npz and rank == 0 and cache:
        _save_path = cache_npz
        _save_cache = dict(cache)  # 참조 스냅샷

        def _bg_save():
            print(f"💾 NPY 캐시 저장 시작 (백그라운드) → {_save_path} ...")
            try:
                sorted_keys = sorted(_save_cache.keys())
                paths_arr = np.array(sorted_keys, dtype=object)
                arrays_arr = np.stack([_save_cache[k] for k in sorted_keys], axis=0)
                # numpy는 파일명이 .npz로 끝나지 않으면 자동으로 .npz를 붙입니다.
                # 따라서 임시 파일도 반드시 .npz로 끝나게 만들어야 os.replace가 정확히 동작합니다.
                tmp_path = _save_path[:-4] + f".tmp.{os.getpid()}.npz"
                np.savez_compressed(tmp_path, paths=paths_arr, arrays=arrays_arr)
                os.replace(tmp_path, _save_path)  # 원자적으로 최종 파일로 교체
                fsize_mb = os.path.getsize(_save_path) / (1024 * 1024)
                print(f"✅ NPY 캐시 저장 완료: {_save_path} ({fsize_mb:.1f} MB on disk)")
            except Exception as e:
                print(f"⚠️ NPY 캐시 저장 실패: {e}")

        _t = threading.Thread(target=_bg_save, daemon=False, name="npz-cache-saver")
        _t.start()

    return cache


# --------------
# Collate function to skip None samples
# --------------
import threading

_collate_drop_stats: Dict[str, Dict[str, Any]] = {}
_collate_drop_lock = threading.Lock()


def collate_skip_none(batch):
    """Filter invalid samples and log drop counts by split/reason/class."""
    valid = []
    for item in batch:
        if item is None:
            with _collate_drop_lock:
                d = _collate_drop_stats.setdefault("unknown", {"total": 0, "pos": 0, "neg": 0, "reasons": {}})
                d["total"] += 1
                d["reasons"]["none_item"] = int(d["reasons"].get("none_item", 0)) + 1
            continue
        if isinstance(item, dict) and bool(item.get("__drop__", False)):
            split = str(item.get("split", "unknown"))
            label = int(item.get("label", -1))
            reason = str(item.get("reason", "unknown"))
            with _collate_drop_lock:
                d = _collate_drop_stats.setdefault(split, {"total": 0, "pos": 0, "neg": 0, "reasons": {}})
                d["total"] += 1
                if label == 1:
                    d["pos"] += 1
                elif label == 0:
                    d["neg"] += 1
                d["reasons"][reason] = int(d["reasons"].get(reason, 0)) + 1
            continue
        valid.append(item)
    if not valid:
        return None
    return torch.utils.data.dataloader.default_collate(valid)


def consume_collate_drop_stats(split_name):
    with _collate_drop_lock:
        d = _collate_drop_stats.pop(str(split_name), None)
    if d is None:
        return {"total": 0, "pos": 0, "neg": 0, "reasons": {}}
    return d


# --------------
# Focal Loss
# --------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def _binary_logits_from_outputs(outputs):
    """모델 출력을 BCE가 최적화하는 단일 binary logit으로 변환합니다."""
    if outputs.dim() == 2 and outputs.size(1) == 2:
        return outputs[:, 1] - outputs[:, 0]
    if outputs.dim() == 2 and outputs.size(1) == 1:
        return outputs.squeeze(1)
    return outputs


def binary_positive_probability(outputs):
    """BCE binary logit과 일치하는 양성(class 1) 확률을 반환합니다."""
    return torch.sigmoid(_binary_logits_from_outputs(outputs))


class BCELossWrapper(nn.Module):
    """Binary Cross Entropy loss for a 2-logit (softmax-style) classifier head."""
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = None if pos_weight is None else float(pos_weight)

    def forward(self, inputs, targets):
        binary_logits = _binary_logits_from_outputs(inputs)
        targets_float = targets.float()
        if self.pos_weight is None:
            return F.binary_cross_entropy_with_logits(binary_logits, targets_float)
        pw = torch.tensor([self.pos_weight], device=binary_logits.device, dtype=binary_logits.dtype)
        return F.binary_cross_entropy_with_logits(binary_logits, targets_float, pos_weight=pw)


def _set_requires_grad_by_name(
    module: nn.Module,
    name_parts: Tuple[str, ...],
    requires_grad: bool,
) -> None:
    for name, param in module.named_parameters():
        if any(part in name for part in name_parts):
            param.requires_grad = bool(requires_grad)


class RadDinoSpatialBackbone(nn.Module):
    """RAD-DINO의 CLS 토큰을 제외한 패치 토큰을 2D 특징맵으로 반환합니다."""

    def __init__(self, model_name: str = RAD_DINO_HF_ID):
        super().__init__()
        try:
            from transformers import AutoModel  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "RAD-DINO를 사용하려면 conda sepsis 환경에 transformers와 "
                "safetensors가 필요합니다."
            ) from exc

        self.model_name = _canonical_rad_dino_model_name(model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name)
        # 분류 forward에서 쓰이지 않는 DINO mask token을 학습 대상으로 두면
        # DDP가 unused parameter로 판단할 수 있습니다.
        _set_requires_grad_by_name(self.encoder, ('mask_token',), False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(pixel_values=x)
        tokens = getattr(out, 'last_hidden_state', None)
        if tokens is None or tokens.ndim != 3:
            raise RuntimeError("RAD-DINO 출력에서 patch token을 찾지 못했습니다.")
        patch_tokens = tokens[:, 1:, :]
        side = int(math.isqrt(patch_tokens.shape[1]))
        if side * side != patch_tokens.shape[1]:
            raise RuntimeError(
                f"RAD-DINO patch token 수가 정사각형이 아닙니다: {patch_tokens.shape[1]}"
            )
        return patch_tokens.permute(0, 2, 1).reshape(
            patch_tokens.shape[0], patch_tokens.shape[2], side, side
        )


class CheXficientSpatialBackbone(nn.Module):
    """CheXficient의 흉부 X-ray 사전학습 패치 특징을 2D 맵으로 반환합니다."""

    @staticmethod
    def _resize_encoder_input(encoder: nn.Module, input_size: int) -> None:
        target = int(input_size)
        if target == CHEXFICIENT_INPUT_SIZE:
            return
        from timm.layers import resample_abs_pos_embed  # type: ignore

        patch = encoder.patch_embed.patch_size
        patch = int(patch[0] if isinstance(patch, tuple) else patch)
        new_grid = (target // patch, target // patch)
        encoder.pos_embed = nn.Parameter(resample_abs_pos_embed(
            encoder.pos_embed,
            new_size=new_grid,
            old_size=encoder.patch_embed.grid_size,
            num_prefix_tokens=getattr(encoder, 'num_prefix_tokens', 1),
        ))
        encoder.patch_embed.img_size = (target, target)
        encoder.patch_embed.grid_size = new_grid

    def __init__(
        self,
        model_name: str = CHEXFICIENT_HF_ID,
        input_size: int = CHEXFICIENT_INPUT_SIZE,
    ):
        super().__init__()
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
            from safetensors.torch import load_file  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "CheXficient를 사용하려면 conda sepsis 환경에 "
                "huggingface_hub와 safetensors가 필요합니다."
            ) from exc

        self.model_name = _canonical_chexficient_model_name(model_name)
        self.encoder = timm.create_model(
            'vit_base_patch14_dinov2',
            pretrained=False,
            num_classes=0,
            img_size=CHEXFICIENT_INPUT_SIZE,
        )
        self.image_projection = nn.Linear(
            int(getattr(self.encoder, 'num_features', 768)),
            512,
        )

        state_path = hf_hub_download(self.model_name, 'model.safetensors')
        state = load_file(state_path, device='cpu')
        image_state = {
            key[len('image_encoder.model.'):]: value
            for key, value in state.items()
            if key.startswith('image_encoder.model.')
            and key != 'image_encoder.model.mask_token'
        }
        missing, unexpected = self.encoder.load_state_dict(image_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "CheXficient image encoder 가중치가 맞지 않습니다: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        self.image_projection.weight.data.copy_(
            state['image_projection.projection.weight']
        )
        self.image_projection.bias.data.copy_(
            state['image_projection.projection.bias']
        )
        self._resize_encoder_input(self.encoder, int(input_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.forward_features(x)
        if tokens.ndim != 3:
            raise RuntimeError(
                f"CheXficient patch token 형태가 예상과 다릅니다: {tuple(tokens.shape)}"
            )
        n_prefix = int(getattr(self.encoder, 'num_prefix_tokens', 1) or 1)
        patch_tokens = self.image_projection(tokens[:, n_prefix:, :])
        patch_tokens = F.normalize(patch_tokens, dim=-1)
        side = int(math.isqrt(patch_tokens.shape[1]))
        if side * side != patch_tokens.shape[1]:
            raise RuntimeError(
                f"CheXficient patch token 수가 정사각형이 아닙니다: {patch_tokens.shape[1]}"
            )
        return patch_tokens.permute(0, 2, 1).reshape(
            patch_tokens.shape[0], patch_tokens.shape[2], side, side
        )


class CXRSpatialClassifier(nn.Module):
    """단일 CXR (RGB 3채널): 사전학습 백본 → 2D 피처맵 → GAP → 이진 분류 헤드 (폐렴 분류)."""

    def __init__(
        self,
        model_name: str,
        input_size: int = 512,
        dropout: float = 0.1,
        eva_x_ckpt_dir: Optional[str] = None,
    ):
        super().__init__()
        self._is_evax = bool(EVA_X_NAMES and model_name in EVA_X_NAMES)
        self._num_prefix = 0
        self._patch_h = self._patch_w = 0
        self._feat_fmt = "bchw"

        if self._is_evax:
            if get_evax_backbone is None:
                raise ImportError(
                    "EVA-X loader import failed. Expected eva_x.py under "
                    f"{os.path.join(PROJECT_ROOT, 'eva-x')}."
                )
            if not eva_x_ckpt_dir:
                raise ValueError(
                    f"EVA-X backbone '{model_name}' requires eva_x_ckpt_dir (path to .pt or folder). "
                    "Download from https://huggingface.co/MapleF/eva_x"
                )
            self.backbone = get_evax_backbone(model_name, eva_x_ckpt_dir, img_size=input_size, verbose=True)
            with torch.no_grad():
                dummy = torch.randn(1, 3, int(input_size), int(input_size))
                feats = self.backbone.forward_features(dummy)
                self._num_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
                patch_tok = feats[:, self._num_prefix :]
                N, C = patch_tok.shape[1], patch_tok.shape[2]
                H = int(math.isqrt(N))
                if H * H != N:
                    raise ValueError(
                        f"EVA-X 패치 토큰 수({N})가 정사각형 격자로 변환되지 않습니다. input_size={input_size}"
                    )
                self._patch_h = self._patch_w = H
                feat_dim = C
                self._feat_fmt = "evax"
        elif _is_rad_dino_model(model_name):
            self.backbone = RadDinoSpatialBackbone(model_name)
            feat_dim = int(getattr(self.backbone.encoder.config, 'hidden_size'))
            self._feat_fmt = "bchw"
        elif _is_chexficient_model(model_name):
            self.backbone = CheXficientSpatialBackbone(model_name, input_size=input_size)
            feat_dim = int(self.backbone.image_projection.out_features)
            self._feat_fmt = "bchw"
        else:
            try:
                self.backbone = timm.create_model(
                    model_name, pretrained=True, num_classes=0, global_pool="", img_size=int(input_size)
                )
            except TypeError as e:
                if "img_size" not in str(e):
                    raise
                self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="")
            with torch.no_grad():
                dummy = torch.randn(1, 3, int(input_size), int(input_size))
                out = self.backbone(dummy)
                if out.dim() == 4:
                    self._feat_fmt = "bchw"
                    feat_dim = out.shape[1]
                elif out.dim() == 3:
                    self._num_prefix = int(getattr(self.backbone, "num_prefix_tokens", 0) or 0)
                    N_total, feat_dim = out.shape[1], out.shape[2]
                    N = int(N_total - self._num_prefix)
                    H = int(math.isqrt(N))
                    if self._num_prefix < 0 or N <= 0 or H * H != N:
                        raise ValueError(
                            f"ViT 패치 토큰을 격자로 변환할 수 없습니다. model={model_name}, input_size={input_size}"
                        )
                    self._patch_h = self._patch_w = H
                    self._feat_fmt = "bnc"
                else:
                    raise ValueError(f"지원하지 않는 백본 출력 형태: {out.shape}")

        self.gap = nn.AdaptiveAvgPool2d(1)
        head_in = feat_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(dropout),
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def _backbone_to_2d(self, x: torch.Tensor) -> torch.Tensor:
        if self._feat_fmt == "evax":
            feats = self.backbone.forward_features(x)
            patch_tok = feats[:, self._num_prefix :]
            B, N, C = patch_tok.shape
            return patch_tok.permute(0, 2, 1).reshape(B, C, self._patch_h, self._patch_w)
        if self._feat_fmt == "bchw":
            return self.backbone(x)
        out = self.backbone(x)
        spatial_tokens = out[:, int(getattr(self, "_num_prefix", 0)) :, :]
        B, N, C = spatial_tokens.shape
        return spatial_tokens.permute(0, 2, 1).reshape(B, C, self._patch_h, self._patch_w)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self._backbone_frozen = True

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "_backbone_frozen", False):
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor, return_extras: bool = False):
        fm = self._backbone_to_2d(x)
        if return_extras:
            feat = self.gap(fm).flatten(1)
            logits = self.head(feat)
            return logits, {
                "feat_map": fm,
                "pooled_feat": feat,
            }
        feat = self.gap(fm).flatten(1)
        return self.head(feat)


# ----------------------
# Data loading helpers
# ----------------------
# Worker init must stay picklable for multiprocessing dataloaders.
class _SeedWorker:
    def __init__(self, seed: int):
        self.seed = int(seed)

    def __call__(self, worker_id: int):
        worker_seed = self.seed + int(worker_id)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)


def _prepare_torch_compile_dynamo(recompile_limit: int = 48) -> None:
    """torch.compile + Dynamo: train(autocast+grad) vs val(autocast+no_grad) 등으로 재컴파일이 누적될 수 있어 상한을 완화."""
    try:
        import torch._dynamo
        torch._dynamo.config.recompile_limit = int(recompile_limit)
    except Exception:
        pass


def _safe_stat(path: str):
    try:
        st = os.stat(path)
        return {
            'exists': True,
            'size_bytes': int(st.st_size),
            'mtime_epoch': float(st.st_mtime),
        }
    except Exception:
        return {'exists': False}


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _c_contiguous_strides(shape):
    """C-contiguous(행 우선) 레이아웃의 stride를 계산합니다.

    size=1 인 차원은 PyTorch ``is_contiguous()`` 가 실제 stride를 무시하므로,
    이 함수로 '기본 레이아웃' stride를 직접 비교해야 합니다.
    예: shape [1024, 1, 7, 7] → strides [49, 49, 7, 1]
    """
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * int(shape[i + 1])
    return tuple(strides)


def _canonicalize_grad_strides(grad):
    """DDP bucket이 기대하는 기본 stride로 gradient를 맞춥니다.

    ConvNeXt depthwise conv(가중치 shape [C, 1, 7, 7])의 backward는
    ``is_contiguous()==True`` 이면서 stride만 [49, 1, 7, 1] 인 gradient를
    만들 수 있습니다. DDP는 [49, 49, 7, 1] 을 기대하므로 경고가 납니다.
    ``contiguous()`` 는 no-op 이라서, 필요할 때만 새 텐서로 복사합니다.
    """
    if grad is None:
        return None
    if tuple(grad.stride()) == _c_contiguous_strides(tuple(grad.shape)):
        return grad
    return grad.clone(memory_format=torch.contiguous_format)


def _register_ddp_grad_stride_hooks(model):
    """depthwise conv 가중치에 stride 교정 hook을 등록합니다.

    DDP 래핑 *전에* 등록해야 reducer보다 먼저 실행됩니다.
    모든 파라미터에 걸면 매 backward마다 stride 비교 비용이 생기므로,
    in_channels-per-group 이 1인 4D 가중치(depthwise conv)만 대상으로 합니다.
    """
    n_hooks = 0
    for module in model.modules():
        if not isinstance(module, nn.Conv2d):
            continue
        weight = getattr(module, 'weight', None)
        if weight is None or not weight.requires_grad:
            continue
        if weight.ndim != 4 or int(weight.shape[1]) != 1:
            continue
        weight.register_hook(_canonicalize_grad_strides)
        n_hooks += 1
    return n_hooks


def _get_state_dict_for_save(model):
    """DDP 및 torch.compile 래퍼를 벗겨서 저장용 state_dict 반환 (키에 _orig_mod. 없음)."""
    m = model.module if hasattr(model, 'module') else model
    m = getattr(m, '_orig_mod', m)
    return m.state_dict()


def _load_state_dict_compat(model, state_dict, strict=True):
    """로드 시 state_dict에 _orig_mod. 접두사가 있으면 제거 후 load_state_dict."""
    if state_dict and any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=strict)


def build_run_meta(*, data_root: str, save_dir: str, view: str, pos_mode: str, split_mode: str, split_tag: str, config: dict,
                   data_mode: str = "medsam3_seg", view_tag: str = "pa"):
    data_root_n = os.path.normpath(str(data_root))
    save_dir_n = os.path.normpath(str(save_dir))
    dataset_name = os.path.basename(data_root_n.rstrip("\\/"))
    try:
        data_root_rel = os.path.relpath(data_root_n, PROJECT_ROOT)
    except Exception:
        data_root_rel = None

    _seg_tag = "chexmask" if "chexmask" in data_mode else "medsam3"
    manifest_path = os.path.join(data_root_n, f"manifest_{view_tag}_{_seg_tag}_ok.json")
    manifest_stat = _safe_stat(manifest_path)
    manifest_sha256 = None
    if manifest_stat.get('exists'):
        try:
            manifest_sha256 = _sha256_file(manifest_path)
        except Exception:
            manifest_sha256 = None

    split_reference = None
    split_reference_root = config.get('split_reference_data_root')
    if split_reference_root:
        split_reference_root = os.path.normpath(str(split_reference_root))
        split_reference_mode = str(config.get('split_reference_data_mode') or 'raw')
        split_reference_seg = "chexmask" if "chexmask" in split_reference_mode else "medsam3"
        split_reference_manifest = os.path.join(
            split_reference_root,
            f"manifest_{view_tag}_{split_reference_seg}_ok.json",
        )
        split_reference_stat = _safe_stat(split_reference_manifest)
        split_reference_sha256 = None
        if split_reference_stat.get('exists'):
            try:
                split_reference_sha256 = _sha256_file(split_reference_manifest)
            except Exception:
                split_reference_sha256 = None
        split_reference = {
            'data_root': split_reference_root,
            'data_mode': split_reference_mode,
            'manifest_path': split_reference_manifest,
            **split_reference_stat,
            'sha256': split_reference_sha256,
        }

    git_head = None
    git_dirty = None
    try:
        git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL).decode('utf-8').strip()
        git_dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL).decode('utf-8').strip())
    except Exception:
        git_head = None
        git_dirty = None

    return {
        'dataset': {
            'data_root': data_root_n,
            'data_root_rel_to_project': data_root_rel,
            'dataset_name': dataset_name,
            'view': str(view),
            'pos_mode': str(pos_mode),
        },
        'run': {
            'save_dir': save_dir_n,
            'split_mode': str(split_mode),
            'split_tag': str(split_tag),
            'seed': int(config.get('seed', 0)),
            'split_seed': int(config.get('split_seed', 42)),
            'split_dir': config.get('split_dir'),
            'arch': str(config.get('arch') or ''),
            'model_name': str(config.get('model_name') or ''),
            'freeze_backbone': bool(config.get('freeze_backbone', False)),
            'max_folds': int(config.get('max_folds')) if config.get('max_folds') is not None else None,
        },
        'manifest': {
            'path': os.path.normpath(manifest_path),
            **manifest_stat,
            'sha256': manifest_sha256,
        },
        'split_reference': split_reference,
        'exec': {
            'python': sys.executable,
            'argv': list(sys.argv),
            'cwd': os.getcwd(),
        },
        'git': {
            'head': git_head,
            'dirty': git_dirty,
        }
    }


# ----------------------
# Sampling utilities
# ----------------------
def downsample_negative_to_ratio(indices, labels, neg_per_pos=1.0, seed=42):
    idx_list = list(indices)
    pos_idx = [i for i in idx_list if labels[i] == 1]
    neg_idx = [i for i in idx_list if labels[i] == 0]
    pos_n = len(pos_idx)
    neg_n = len(neg_idx)
    if pos_n == 0 or neg_n == 0:
        return idx_list
    target_neg = int(np.ceil(max(0.0, float(neg_per_pos)) * pos_n))
    target_neg = max(0, min(target_neg, neg_n))
    rng = np.random.default_rng(seed)
    keep_neg = rng.choice(neg_idx, size=target_neg, replace=False).tolist() if target_neg < neg_n else neg_idx
    mixed = pos_idx + keep_neg
    rng.shuffle(mixed)
    return mixed


# ----------------------
# Train/Validate epoch
# ----------------------
def _ddp_any_nan(loss: torch.Tensor, device: torch.device) -> bool:
    """DDP 환경에서 어느 rank든 NaN/Inf loss가 있으면 True 반환 (단일 GPU면 로컬 검사만).

    DDP 학습 중 한 rank만 NaN을 감지해 backward()를 건너뛰면,
    다른 rank는 gradient ALLREDUCE를 기다리다 NCCL 타임아웃(교착 상태)이 발생합니다.
    이 함수로 모든 rank가 동시에 skip 여부를 결정하면 NCCL 연산 순서가 항상 일치합니다.
    """
    is_nan = torch.isnan(loss) or torch.isinf(loss)
    if int(os.environ.get('WORLD_SIZE', 1)) > 1 and dist.is_available() and dist.is_initialized():
        flag = torch.tensor(1 if is_nan else 0, dtype=torch.long, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return flag.item() > 0
    return bool(is_nan)


def _model_tag_for_save_dir(config: dict, arch: Optional[str] = None) -> str:
    """결과 폴더명용 모델 태그 (--arch 우선, HF repo id는 org 접두사 제거)."""
    if arch:
        return str(arch).replace('/', '-').replace('\\', '-')
    model_name = str(config.get('model_name') or 'model')
    _known_tags = {
        RAD_DINO_HF_ID.lower(): 'rad_dino',
        'rad-dino': 'rad_dino',
        CHEXFICIENT_HF_ID.lower(): 'chexficient',
        'stanfordaimi/chexficient': 'chexficient',
    }
    key = model_name.replace('\\', '/').lower()
    if key in _known_tags:
        return _known_tags[key]
    if '/' in model_name:
        model_name = model_name.rsplit('/', 1)[-1]
    return model_name.replace('/', '-').replace('\\', '-')


def _ddp_barrier(local_rank: int = 0) -> None:
    """모든 rank가 이 지점에 도달할 때까지 대기.

    rank 0 전용 파일 I/O(체크포인트·그림 저장) 직후에 호출한다. 이 동기화가 없으면
    rank!=0 이 먼저 다음 epoch 의 collective(ALLREDUCE)에 진입해, 저장이 끝나지 않은
    rank 0 을 기다리다 NCCL watchdog 타임아웃으로 학습 전체가 죽는다.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def train_epoch_single(model, dataloader, criterion, optimizer, device, scaler=None, use_bf16=False, max_grad_norm=1.0):
    """단일 CXR 폐렴 분류 모델 학습 epoch"""
    if use_bf16:
        scaler = None  # bf16 사용 시 GradScaler 사용 안 함
    model.train()
    running_loss = 0.0
    n_batches = 0
    n_nan_skipped = 0
    all_probs, all_labels, all_preds = [], [], []
    _ddp_quiet = int(os.environ.get('WORLD_SIZE', 1)) > 1 and int(os.environ.get('RANK', 0)) != 0
    for batch in tqdm(dataloader, desc="Training(Single)", disable=_ddp_quiet):
        if batch is None:
            continue
        imgs, labels, _ = batch
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_bf16 and device.type == 'cuda':
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            if _ddp_any_nan(loss, device):
                n_nan_skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        elif scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            if _ddp_any_nan(loss, device):
                n_nan_skipped += 1
                scaler.update()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if _ddp_any_nan(loss, device):
                n_nan_skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        n_batches += 1
        running_loss += loss.item()
        out_f = outputs.detach().float()
        if isinstance(criterion, BCELossWrapper):
            probs_pos = binary_positive_probability(out_f)
            batch_probs = probs_pos.cpu().numpy()
            predicted = (probs_pos > 0.5).long()
        else:
            probs = torch.softmax(out_f, dim=1)
            batch_probs = probs[:, 1].cpu().numpy()
            _, predicted = torch.max(probs, 1)
        batch_labels = labels.cpu().numpy()
        batch_preds = predicted.cpu().numpy()
        valid_mask = ~np.isnan(batch_probs)
        all_probs.extend(batch_probs[valid_mask].tolist())
        all_labels.extend(batch_labels[valid_mask].tolist())
        all_preds.extend(batch_preds[valid_mask].tolist())
    if n_nan_skipped > 0 and int(os.environ.get('RANK', 0)) == 0:
        print(f"⚠️ train_epoch_single: NaN/Inf loss {n_nan_skipped} 배치 스킵됨 (grad explosion?)")
    epoch_loss = running_loss / max(1, n_batches)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels)) if all_preds else 0.0
    epoch_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    epoch_auprc = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1]) if all_preds else np.zeros((2, 2))
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        epoch_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    else:
        epoch_sensitivity = 0
    split_name = getattr(getattr(dataloader, 'dataset', None), 'split_name', 'train')
    ds = consume_collate_drop_stats(split_name)
    if int(ds.get("total", 0)) > 0 and int(os.environ.get('RANK', 0)) == 0:
        print(f"🧾 Runtime drop[{split_name}] total={ds['total']} pos={ds['pos']} neg={ds['neg']} reasons={ds.get('reasons', {})}")
    return epoch_loss, epoch_acc, epoch_auc, epoch_auprc, epoch_sensitivity


def validate_epoch_single(model, dataloader, criterion, device, use_bf16: bool = False, use_fp32: bool = False):
    """단일 CXR 폐렴 분류 모델 검증 epoch"""
    model.eval()
    running_loss = 0.0
    n_batches = 0
    all_probs, all_labels, all_preds = [], [], []
    val_dirs = []
    with torch.no_grad():
        _ddp_quiet = int(os.environ.get('WORLD_SIZE', 1)) > 1 and int(os.environ.get('RANK', 0)) != 0
        for batch in tqdm(dataloader, desc="Validation(Single)", disable=_ddp_quiet):
            if batch is None:
                continue
            imgs, labels, dirs = batch
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if torch.isnan(imgs).any() or torch.isinf(imgs).any():
                continue
            if device.type == 'cuda':
                if use_fp32:
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                elif use_bf16:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                else:
                    with torch.amp.autocast('cuda'):
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            n_batches += 1
            running_loss += loss.item()
            out_f = outputs.float()
            if isinstance(criterion, BCELossWrapper):
                probs_pos = binary_positive_probability(out_f)
                predicted = (probs_pos > 0.5).long()
                all_probs.extend(probs_pos.cpu().numpy())
            else:
                probs = torch.softmax(out_f, dim=1)
                _, predicted = torch.max(probs, 1)
                all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            if isinstance(dirs, (list, tuple)):
                val_dirs.extend(list(dirs))
            else:
                val_dirs.append(dirs)
    epoch_loss = running_loss / max(1, n_batches)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels)) if all_preds else 0.0
    epoch_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    epoch_auprc = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1]) if all_preds else np.zeros((2, 2))
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        epoch_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    else:
        epoch_sensitivity = 0
    split_name = getattr(getattr(dataloader, 'dataset', None), 'split_name', 'val')
    ds = consume_collate_drop_stats(split_name)
    if int(ds.get("total", 0)) > 0 and int(os.environ.get('RANK', 0)) == 0:
        print(f"🧾 Runtime drop[{split_name}] total={ds['total']} pos={ds['pos']} neg={ds['neg']} reasons={ds.get('reasons', {})}")
    return epoch_loss, epoch_acc, epoch_auc, epoch_auprc, epoch_sensitivity, all_labels, all_preds, all_probs, val_dirs


# ----------------------
# Threshold utilities
# ----------------------
def tune_threshold(y_true, y_prob, mode='youden', min_specificity=0.9, min_sensitivity=0.8):
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob, dtype=float)
    if y_prob.size == 0:
        return 0.5

    if mode == 'mcc':
        thr_candidates = np.unique(np.concatenate([y_prob, [0.0, 1.0]]))
        best_th, best_mcc = 0.5, -1.0
        for th in thr_candidates:
            y_pred = (y_prob >= th).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_th = th
        return float(best_th)

    if mode == 'youden':
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        if len(thr) > 0:
            youden = tpr - fpr  # = sensitivity + specificity - 1
            return float(thr[int(np.argmax(youden))])
        return 0.5

    if mode == 'f1':
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        f1_arr = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
        if len(thr) > 0:
            idx = int(np.argmax(f1_arr))
            return float(thr[idx])

    # sensitivity guard: achieve at least min_sensitivity, then maximize specificity
    if mode == 'sens':
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        if len(thr) == 0:
            return 0.5
        mask = tpr >= min_sensitivity
        if mask.any():
            idxs = np.where(mask)[0]
            # among thresholds with sens >= min_sensitivity, pick highest specificity (lowest fpr)
            best_i = idxs[np.argmin(fpr[idxs])]
            return float(thr[best_i])
        # cannot achieve min_sensitivity: use threshold that maximizes sensitivity
        best_i = int(np.argmax(tpr))
        return float(thr[best_i])

    # default: specificity guard (spec mode)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr
    mask = spec >= min_specificity
    if mask.any():
        idxs = np.where(mask)[0]
        best_i = idxs[np.argmax(tpr[idxs])]
        return float(thr[best_i])
    elif len(thr) > 0:
        best_i = int(np.argmax(spec))
        return float(thr[best_i])
    return 0.5


def compute_metrics_with_threshold(y_true, y_prob, threshold):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    if y_true.size == 0 or y_prob.size == 0:
        return {'accuracy': 0.0, 'sensitivity': 0.0, 'specificity': 0.0,
                'precision': 0.0, 'f1_score': 0.0, 'auc': 0.0, 'auprc': 0.0, 'mcc': 0.0}
    y_pred = (y_prob >= float(threshold)).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        sens = tp / max(1, (tp + fn))
        spec = tn / max(1, (tn + fp))
        prec = tp / max(1, (tp + fp))
        f1 = 2 * tp / max(1, (2*tp + fp + fn))
    else:
        acc = sens = spec = prec = f1 = 0.0
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    auprc = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
        mcc_val = 0.0 if (mcc != mcc) else float(mcc)  # NaN check: nan != nan
    except Exception:
        mcc_val = 0.0
    return {
        'accuracy': float(acc), 'sensitivity': float(sens), 'specificity': float(spec),
        'precision': float(prec), 'f1_score': float(f1), 'auc': float(auc), 'auprc': float(auprc),
        'mcc': mcc_val, 'threshold': float(threshold)
    }


def bootstrap_oof_metric_cis(y_true, y_prob, threshold, n_boot=2000, seed=42):
    """
    Stratified-like bootstrap via sample-level resampling (with replacement).
    Returns 95% percentile CI for AUROC/AUPRC and threshold-based metrics.
    """
    y_arr = np.array(y_true, dtype=int)
    p_arr = np.array(y_prob, dtype=float)
    rng = np.random.RandomState(int(seed))

    boot_values = {
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'f1_score': [],
        'mcc': [],
        'auc': [],
        'auprc': [],
    }

    n = len(y_arr)
    for _ in range(int(n_boot)):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_arr[idx]
        p_b = p_arr[idx]
        if len(np.unique(y_b)) < 2:
            continue
        m = compute_metrics_with_threshold(y_b, p_b, threshold)
        for k in boot_values.keys():
            boot_values[k].append(float(m.get(k, 0.0)))

    ci = {}
    for k, vals in boot_values.items():
        if vals:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            ci[k] = [float(lo), float(hi)]
        else:
            ci[k] = [0.0, 0.0]
    return ci


def compute_metrics_from_pred(y_true, y_pred, y_prob=None):
    """사전 계산된 y_pred(이미 threshold 적용됨) 로 지표 계산.

    A2 수정용: 샘플별로 threshold 가 다른 경우(fold 별 threshold 적용)에도
    confusion matrix 기반 지표를 일관되게 계산할 수 있도록 helper.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    except Exception:
        tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())
        fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
        fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
        tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
    n_total = tn + fp + fn + tp
    acc = (tp + tn) / max(1, n_total)
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    f1 = (2.0 * tp) / max(1, 2 * tp + fp + fn)
    try:
        mcc_raw = matthews_corrcoef(y_true_arr, y_pred_arr)
        mcc_val = 0.0 if (mcc_raw != mcc_raw) else float(mcc_raw)
    except Exception:
        mcc_val = 0.0
    out = {
        'accuracy': float(acc), 'sensitivity': float(sens), 'specificity': float(spec),
        'precision': float(prec), 'f1_score': float(f1), 'mcc': float(mcc_val),
    }
    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=float)
        out['brier_score'] = float(calc_brier(y_true_arr, y_prob_arr))
        if len(set(y_true_arr.tolist())) > 1:
            out['auc'] = float(roc_auc_score(y_true_arr, y_prob_arr))
            out['auprc'] = float(average_precision_score(y_true_arr, y_prob_arr))
        else:
            out['auc'] = 0.0
            out['auprc'] = 0.0
    return out


def bootstrap_metric_cis_per_sample_threshold(y_true, y_prob, thresholds_per_sample, n_boot=2000, seed=42):
    """A2 수정용: 샘플별 threshold(각 fold 가 자기 val 에서 튜닝한 값)을 적용해 bootstrap CI 계산.

    pooled OOF 에서 threshold 를 재-튜닝하지 않는 (덜 편향된) 보고 방식.
    """
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_prob, dtype=float)
    t_arr = np.asarray(thresholds_per_sample, dtype=float)
    rng = np.random.RandomState(int(seed))
    keys = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'mcc', 'auc', 'auprc']
    boot = {k: [] for k in keys}
    n = len(y_arr)
    for _ in range(int(n_boot)):
        idx = rng.randint(0, n, size=n)
        yb = y_arr[idx]; pb = p_arr[idx]; tb = t_arr[idx]
        if len(set(yb.tolist())) < 2:
            continue
        ypb = (pb >= tb).astype(int)
        m = compute_metrics_from_pred(yb, ypb, pb)
        for k in keys:
            boot[k].append(float(m.get(k, 0.0)))
    ci = {}
    for k, vals in boot.items():
        if vals:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            ci[k] = [float(lo), float(hi)]
        else:
            ci[k] = [0.0, 0.0]
    return ci


def aggregate_patient_level(labels, probs, subject_ids, agg: str = 'mean'):
    """B2 수정용: 영상(image) 레벨 예측을 환자(subject) 레벨로 집계.

    Args:
        labels: list/array, image-level 0/1 labels
        probs: list/array, image-level positive class probabilities
        subject_ids: list of subject identifiers (one per image)
        agg: 'mean' (권장, 평균 확률) 또는 'max' (보수적 trigger)

    Returns:
        (pat_labels: list[int], pat_probs: list[float], pat_ids: list)
        각 환자당 1개의 예측 확률과 라벨. 같은 환자의 라벨이 엇갈리면 max(= 양성 우선) 으로 보수적 처리.
    """
    from collections import defaultdict
    probs_by_subj: Dict[Any, List[float]] = defaultdict(list)
    label_by_subj: Dict[Any, int] = {}
    for l, p, s in zip(labels, probs, subject_ids):
        if s is None:
            continue
        try:
            probs_by_subj[s].append(float(p))
        except Exception:
            continue
        li = int(l)
        if s in label_by_subj:
            if label_by_subj[s] != li:
                label_by_subj[s] = max(label_by_subj[s], li)
        else:
            label_by_subj[s] = li
    pat_ids = sorted(probs_by_subj.keys(), key=lambda x: str(x))
    pat_probs: List[float] = []
    for s in pat_ids:
        vals = probs_by_subj[s]
        if not vals:
            pat_probs.append(0.0)
        elif agg == 'max':
            pat_probs.append(float(max(vals)))
        else:
            pat_probs.append(float(sum(vals) / len(vals)))
    pat_labels = [int(label_by_subj.get(s, 0)) for s in pat_ids]
    return pat_labels, pat_probs, pat_ids


# ----------------------
# Visualization helpers
# ----------------------
def _empty_label_arrays(*arrays) -> bool:
    """list / numpy array 모두에서 빈 입력 여부를 안전하게 검사합니다."""
    for arr in arrays:
        if arr is None:
            return True
        try:
            if len(arr) == 0:
                return True
        except TypeError:
            if not arr:
                return True
    return False


def save_confusion_matrix(y_true, y_pred, save_path, normalize=True, cmap='Blues'):
    if _empty_label_arrays(y_true, y_pred):
        return
    labels = ['Normal', 'Pneumonia']
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if normalize:
        cm_norm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize='true')
        vmin, vmax = 0.0, 1.0
    else:
        cm_norm = cm.astype(float)
        vmin, vmax = 0.0, float(cm_norm.max()) if cm_norm.max() > 0 else 1.0
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion' if normalize else 'Count', rotation=270, labelpad=14)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    fmt = '.1%' if normalize else 'd'
    thresh = (vmax - vmin) / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count_txt = f"{cm[i, j]}"
            if normalize:
                pct = cm_norm[i, j]
                pct_txt = f"\n{pct*100:.1f}%"
            else:
                pct_txt = ''
            text = count_txt + pct_txt
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=11)
    ax.set_title('Confusion Matrix (row-normalized colors)' if normalize else 'Confusion Matrix')
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_roc_curve(y_true, y_prob, save_path):
    if _empty_label_arrays(y_true, y_prob) or len(set(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC={auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

    # Also save raw ROC points for downstream aggregation/overlay plots
    try:
        import json
        roc_json_path = save_path.replace('.png', '.json')
        with open(roc_json_path, 'w') as f:
            json.dump({
                'fpr': [float(x) for x in fpr.tolist()],
                'tpr': [float(x) for x in tpr.tolist()],
                'auc': float(auc)
            }, f)
    except Exception:
        pass


def save_pr_curve(y_true, y_prob, save_path):
    if _empty_label_arrays(y_true, y_prob) or len(set(y_true)) < 2:
        return
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    plt.figure(figsize=(8, 6))
    plt.plot(rec, prec, label=f'PR (AUPRC={auprc:.4f})', linewidth=2)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()


# ----------------------
# Extra evaluation utilities (parity with convnext_5fold)
# ----------------------
def calc_prevalence(y_true):
    y = np.array(y_true).astype(int)
    return float(y.mean()) if y.size > 0 else float('nan')


def calc_brier(y_true, y_prob):
    y = np.array(y_true, dtype=float)
    p = np.clip(np.array(y_prob, dtype=float), 1e-7, 1-1e-7)
    return float(np.mean((p - y) ** 2))


def calc_ece(y_true, y_prob, n_bins=10):
    y = np.array(y_true, dtype=float)
    p = np.clip(np.array(y_prob, dtype=float), 1e-7, 1-1e-7)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p < hi) if i < n_bins-1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / max(1, N)) * abs(acc - conf)
    return float(ece)


def calc_calibration_intercept_slope(y_true, y_prob):
    y = np.array(y_true, dtype=int)
    p = np.clip(np.array(y_prob, dtype=float), 1e-6, 1-1e-6)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(fit_intercept=True, solver='lbfgs')
        lr.fit(logit_p, y)
        slope = float(lr.coef_.ravel()[0])
        intercept = float(lr.intercept_.ravel()[0])
        return intercept, slope
    except Exception:
        return float('nan'), float('nan')


def compute_primary_calibration_metrics(y_true, y_prob, n_bins: int = 10) -> Dict[str, float]:
    """Threshold-free primary metrics + calibration summary for JSON export."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_prob, dtype=float)
    if y_arr.size == 0 or p_arr.size == 0:
        return {
            'auroc': 0.0,
            'auprc': 0.0,
            'brier_score': float('nan'),
            'ece': float('nan'),
            'calibration_intercept': float('nan'),
            'calibration_slope': float('nan'),
            'n_bins_ece': int(n_bins),
        }
    has_both_classes = len(set(y_arr.tolist())) > 1
    auroc = float(roc_auc_score(y_arr, p_arr)) if has_both_classes else 0.0
    auprc = float(average_precision_score(y_arr, p_arr)) if has_both_classes else 0.0
    brier = calc_brier(y_arr, p_arr)
    ece = calc_ece(y_arr, p_arr, n_bins=n_bins)
    cal_intercept, cal_slope = calc_calibration_intercept_slope(y_arr, p_arr)
    return {
        'auroc': auroc,
        'auprc': auprc,
        'brier_score': float(brier),
        'ece': float(ece),
        'calibration_intercept': float(cal_intercept),
        'calibration_slope': float(cal_slope),
        'n_bins_ece': int(n_bins),
    }


def find_threshold_for_spec(y_true, y_prob, target_spec=0.95):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr
    mask = spec >= target_spec
    if not mask.any():
        idx = int(np.argmin(np.abs(spec - target_spec)))
    else:
        idxs = np.where(mask)[0]
        best = idxs[np.argmax(tpr[idxs])]
        idx = int(best)
    return float(thr[idx])


def save_example_predictions_grid(labels, probs, img_paths, threshold, save_path,
                                   data_root, per_row=4, **_kwargs):
    """
    Test/Val 예측 결과를 4개 카테고리로 시각화: TN, TP, FP, FN
    ``img_paths``: 마스크드 CXR JPG 파일 경로(절대/상대) 리스트.
    """
    labels = np.array(labels)
    probs = np.array(probs)
    preds = (probs >= threshold).astype(int)
    
    tn_items, tp_items, fp_items, fn_items = [], [], [], []
    
    for i in range(len(labels)):
        y = int(labels[i])
        p = int(preds[i])
        prob = float(probs[i])
        img_path = img_paths[i] if i < len(img_paths) else None
        
        item = {'label': y, 'pred': p, 'prob': prob, 'img_path': img_path}
        
        if y == 0 and p == 0:
            tn_items.append(item)
        elif y == 1 and p == 1:
            tp_items.append(item)
        elif y == 0 and p == 1:
            fp_items.append(item)
        elif y == 1 and p == 0:
            fn_items.append(item)
    
    import random
    random.seed(42)
    for items in [tn_items, tp_items, fp_items, fn_items]:
        random.shuffle(items)
    
    rows, cols = 4, per_row
    fig = plt.figure(figsize=(5*cols, 5*rows))
    
    def render_row(items, row_idx, title, category_color):
        take = min(per_row, len(items))
        for j in range(take):
            it = items[j]
            idx = row_idx * cols + j + 1
            ax = plt.subplot(rows, cols, idx)
            
            target_size = (512, 512)
            if it['img_path']:
                ref = str(it['img_path'])
                cand = ref if os.path.isabs(ref) else os.path.join(data_root, ref)
                if os.path.isfile(cand):
                    try:
                        img = Image.open(cand).convert('RGB')
                        img = img.resize(target_size, Image.LANCZOS)
                        ax.imshow(np.array(img))
                    except Exception:
                        ax.imshow(np.zeros((*target_size, 3), dtype=np.uint8))
                else:
                    ax.imshow(np.zeros((*target_size, 3), dtype=np.uint8))
            else:
                ax.imshow(np.zeros((*target_size, 3), dtype=np.uint8))
            
            color = 'green' if it['pred'] == it['label'] else 'red'
            pos_p = it['prob']
            ax.set_xlabel(
                f"{title}\n"
                f"True: {'Pneumonia' if it['label'] == 1 else 'Normal'}, "
                f"Pred: {'Pneumonia' if it['pred'] == 1 else 'Normal'}\n"
                f"P(Pneumonia): {pos_p:.3f}",
                fontsize=11, color=color, weight='bold'
            )
            ax.set_xticks([])
            ax.set_yticks([])
            
            for spine in ax.spines.values():
                spine.set_edgecolor(category_color)
                spine.set_linewidth(3)
        
        for j in range(take, per_row):
            idx = row_idx * cols + j + 1
            ax = plt.subplot(rows, cols, idx)
            ax.axis('off')
    
    # 각 행 렌더링 (카테고리별 색상)
    render_row(tn_items, 0, 'TN (Normal→Normal)', 'blue')
    render_row(tp_items, 1, 'TP (Pneumonia→Pneumonia)', 'green')
    render_row(fp_items, 2, 'FP (Normal→Pneumonia)', 'orange')
    render_row(fn_items, 3, 'FN (Pneumonia→Normal)', 'red')
    
    plt.suptitle(f'Prediction Examples (Threshold: {threshold:.3f})', 
                 fontsize=16, weight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 예측 예시 그리드 저장: {save_path}")
    print(f"   TN: {len(tn_items)}, TP: {len(tp_items)}, FP: {len(fp_items)}, FN: {len(fn_items)}")

    # Return sampled items for downstream occlusion analysis
    sampled = {
        "tn": tn_items[:min(per_row, len(tn_items))],
        "tp": tp_items[:min(per_row, len(tp_items))],
        "fp": fp_items[:min(per_row, len(fp_items))],
        "fn": fn_items[:min(per_row, len(fn_items))],
    }
    return sampled


# ----------------------
# 설명 가능성(XAI) 스텁
# ----------------------
def save_occlusion_for_sampled_examples(*_args, **_kwargs):
    return None


def save_attention_map_for_sampled_examples(*_args, **_kwargs):
    return None


def save_spatial_attention_for_sampled_examples(*_args, **_kwargs):
    return None


def metrics_at_threshold(y_true, y_prob, thr):
    y = np.array(y_true).astype(int)
    p = np.array(y_prob)
    y_pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return {}
    tn, fp, fn, tp = cm.ravel()
    N = tn + fp + fn + tp
    sens = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    ppv = tp / max(1, (tp + fp))
    npv = tn / max(1, (tn + fn))
    lr_pos = sens / max(1e-9, (1 - spec))
    lr_neg = (1 - sens) / max(1e-9, spec)
    alert_rate = (tp + fp) / max(1, N)
    nne = (1 / ppv) if ppv > 0 else float('inf')
    return {
        'threshold': float(thr),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'ppv': float(ppv),
        'npv': float(npv),
        'lr+': float(lr_pos),
        'lr-': float(lr_neg),
        'alert_rate': float(alert_rate),
        'nne': float(nne)
    }


def decision_curve(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    y = np.array(y_true).astype(int)
    p = np.array(y_prob)
    N = len(y)
    out = []
    for pt in thresholds:
        thr = pt
        y_pred = (p >= thr).astype(int)
        tp = ((y_pred == 1) & (y == 1)).sum()
        fp = ((y_pred == 1) & (y == 0)).sum()
        nb = (tp / max(1, N)) - (fp / max(1, N)) * (pt / max(1e-9, (1 - pt)))
        out.append({'threshold': float(pt), 'net_benefit_model': float(nb)})
    prev = calc_prevalence(y)
    nb_all = [prev - (1 - prev) * (pt / max(1e-9, (1 - pt))) for pt in thresholds]
    nb_none = [0.0 for _ in thresholds]
    return thresholds, out, nb_all, nb_none


def save_calibration_plot(y_true, y_prob, save_path, n_bins=10):
    y = np.array(y_true).astype(float)
    p = np.clip(np.array(y_prob).astype(float), 1e-7, 1-1e-7)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p < hi) if i < n_bins-1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        xs.append(p[mask].mean()); ys.append(y[mask].mean()); ns.append(mask.sum())
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
    if xs:
        sizes = (np.array(ns) / max(ns)) * 100 + 20
        plt.scatter(xs, ys, s=sizes, c='C0', alpha=0.7, label='Model (bin avg)')
        plt.plot(xs, ys, 'C0-', alpha=0.5)
    plt.xlabel('Predicted probability'); plt.ylabel('Observed frequency'); plt.title('Calibration (Reliability)')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()


def save_dca_plot(thresholds, nb_model, nb_all, nb_none, save_path):
    th = np.array(thresholds, dtype=float)
    nb_m = np.array([r['net_benefit_model'] if isinstance(r, dict) else r for r in nb_model], dtype=float)
    nb_all = np.array(nb_all, dtype=float)
    nb_none = np.array(nb_none, dtype=float)
    plt.figure(figsize=(7, 5))
    plt.plot(th, nb_m, label='Model', linewidth=2)
    plt.plot(th, nb_all, label='Treat-all', linestyle='--')
    plt.plot(th, nb_none, label='Treat-none', linestyle=':')
    plt.xlabel('Threshold probability'); plt.ylabel('Net benefit'); plt.title('Decision Curve Analysis')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()


# ----------------------
# Ensemble utilities
# ----------------------
def ensemble_predictions(prob_lists, method='mean'):
    """
    Combine predictions from multiple models.
    
    Args:
        prob_lists: List of probability arrays, each [n_samples]
        method: 'mean' (soft voting), 'median', 'max'
    
    Returns:
        Combined probability array
    """
    probs = np.array(prob_lists, dtype=float)  # [n_models, n_samples]
    if method == 'mean':
        return np.mean(probs, axis=0)
    elif method == 'median':
        return np.median(probs, axis=0)
    elif method == 'max':
        return np.max(probs, axis=0)
    else:
        return np.mean(probs, axis=0)


def load_cv_fold_predictions(cv_result_dir, n_folds=5):
    """
    Load predictions from all folds of a CV run.
    
    Returns:
        dict with 'labels', 'probs', 'fold_auprcs', 'mean_auprc', 'std_auprc'
    """
    all_labels = []
    all_probs = []
    fold_auprcs = []
    
    for fold_idx in range(1, n_folds + 1):
        fold_dir = os.path.join(cv_result_dir, f'fold_{fold_idx}')
        test_result_path = os.path.join(fold_dir, 'fold_test_results.json')
        
        if os.path.exists(test_result_path):
            with open(test_result_path, 'r') as f:
                data = json.load(f)
            fold_auprcs.append(float(data.get('auprc', 0)))
    
    # Load outer test if available
    outer_dir = os.path.join(cv_result_dir, 'outer_test')
    outer_path = os.path.join(outer_dir, 'cv5_outer_test_results.json')
    outer_data = None
    if os.path.exists(outer_path):
        with open(outer_path, 'r') as f:
            outer_data = json.load(f)
    
    return {
        'fold_auprcs': fold_auprcs,
        'mean_auprc': float(np.mean(fold_auprcs)) if fold_auprcs else 0.0,
        'std_auprc': float(np.std(fold_auprcs)) if fold_auprcs else 0.0,
        'outer_test': outer_data,
    }


def evaluate_multi_model_ensemble(model_results_list, test_loader, device, ModelCls, configs):
    """
    Evaluate ensemble of multiple trained models on test data.
    
    Args:
        model_results_list: List of dicts with 'model_path', 'config'
        test_loader: DataLoader for test data
        device: torch device
        ModelCls: ``CXRSpatialClassifier`` 등 단일 CXR 분류기
        configs: List of config dicts for each model
    
    Returns:
        Ensemble predictions and metrics
    """
    all_probs = []
    test_labels = None
    
    for i, (model_info, cfg) in enumerate(zip(model_results_list, configs)):
        model_path = model_info['model_path']
        try:
            model_kwargs = dict(
                model_name=cfg['model_name'],
                input_size=cfg['input_size'],
                dropout=cfg['dropout'],
                eva_x_ckpt_dir=cfg.get('eva_x_ckpt_dir'),
            )
            model = ModelCls(**model_kwargs).to(device)
            state_dict = torch.load(model_path, map_location=device)
            _load_state_dict_compat(model, state_dict)
            model.eval()
            
            probs = []
            labels = []
            with torch.no_grad():
                for batch in test_loader:
                    if batch is None:
                        continue
                    imgs, y, _ = batch
                    imgs = imgs.to(device, non_blocking=True)
                    out = model(imgs)
                    if cfg.get('loss_type') in ('bce', 'weighted_bce'):
                        prob = binary_positive_probability(out)
                    else:
                        prob = torch.softmax(out, dim=1)[:, 1]
                    probs.extend(prob.float().cpu().numpy())
                    labels.extend(y.numpy())
            
            all_probs.append(probs)
            if test_labels is None:
                test_labels = labels
                
        except Exception as e:
            print(f"[ensemble] Failed to load model {model_path}: {e}")
            continue
    
    if not all_probs:
        return None
    
    # Ensemble
    ens_probs = ensemble_predictions(all_probs, method='mean')
    
    # Metrics
    auc = roc_auc_score(test_labels, ens_probs) if len(set(test_labels)) > 1 else 0.0
    auprc = average_precision_score(test_labels, ens_probs) if len(set(test_labels)) > 1 else 0.0
    
    return {
        'labels': test_labels,
        'probs': ens_probs.tolist(),
        'auc': float(auc),
        'auprc': float(auprc),
        'n_models': len(all_probs),
    }


def save_ensemble_roc_overlay(model_results, save_path, title='Multi-Model ROC Comparison'):
    """
    Create ROC curve overlay plot for multiple models.
    
    Args:
        model_results: List of dicts with 'name', 'fpr', 'tpr', 'auc'
        save_path: Output path for the plot
        title: Plot title
    """
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_results)))
    
    for i, res in enumerate(model_results):
        plt.plot(
            res['fpr'], res['tpr'],
            color=colors[i],
            linewidth=2,
            label=f"{res['name']} (AUC={res['auc']:.3f})"
        )
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_ensemble_pr_overlay(model_results, save_path, title='Multi-Model PR Comparison'):
    """
    Create PR curve overlay plot for multiple models.
    
    Args:
        model_results: List of dicts with 'name', 'precision', 'recall', 'auprc'
        save_path: Output path for the plot
        title: Plot title
    """
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_results)))
    
    for i, res in enumerate(model_results):
        plt.plot(
            res['recall'], res['precision'],
            color=colors[i],
            linewidth=2,
            label=f"{res['name']} (AUPRC={res['auprc']:.3f})"
        )
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ----------------------
# Presets and config
# --------------------------------------------------------------------------
# PRESETS — 폐렴 CXR 분류용 권장 백본·하이퍼파라미터
#   목적:
#   - 논문 디펜스/비교표에 자주 쓰는 대표 백본만 유지
#   - 모든 backbone이 같은 학습 세팅을 씀 (공정 비교)
#   - preset마다 다른 값은 model_name 뿐
#   - config JSON이나 CLI를 주지 않으면 이 값을 바로 사용
#   참고:
#   - 현재 기본 fallback preset은 ConvNeXt V2 Tiny 입니다.
#   - 선형 프로브를 원하면 --freeze-backbone 과 별도 LR 설정을 권장합니다.
# --------------------------------------------------------------------------
SHARED_TRAINING = {
    'input_size': 512,
    'learning_rate': 2e-5,
    'weight_decay': 0.05,
    'batch_size': 64,
    'dropout': 0.2,
    'epochs': 50,
    'patience': 8,
    'warmup_epochs': 3,
    'loss_type': 'weighted_bce',
    'use_bf16': True,
    'use_fp32': False,
    'downsample_negative': False,
    'downsample_neg_per_pos': 2.0,
}


def _preset(model_name: str, **overrides) -> dict:
    out = {'model_name': model_name, **SHARED_TRAINING}
    out.update(overrides)
    return out


PRESETS = {
    # Classic Baseline ------------------------------------------------------
    'resnet': _preset('resnet50'),                    # ~25M
    'resnet152': _preset('resnet152'),                # ~58M
    # Classic CXR CNN -------------------------------------------------------
    'densenet': _preset('densenet201'),               # ~20M
    # Modern CNN ------------------------------------------------------------
    'convnextv2': _preset('convnextv2_tiny'),          # ~28M
    'convnextv2_base': _preset('convnextv2_base'),     # ~88M
    # Modern Hierarchical ViT -----------------------------------------------
    'swint': _preset('swinv2_tiny_window8_256'),      # ~28M
    'swint_base': _preset('swinv2_base_window16_256'), # ~87M
    # Self-supervised ViT ---------------------------------------------------
    'dinov3': _preset('vit_small_patch16_dinov3'),    # ~22M
    'dinov3_base': _preset('vit_base_patch16_dinov3'), # ~86M
    # Foundation Model ------------------------------------------------------
    'eva_x_small': _preset('eva_x_small'),            # ~22M
    'eva_x_base': _preset('eva_x_base'),              # ~86M
    'rad_dino': _preset(RAD_DINO_HF_ID, use_bf16=False, use_fp32=True),  # bf16 backward NaN → fp32
    'chexficient': _preset(CHEXFICIENT_HF_ID),        # ViT-Base, ~86M
}

def build_default_config():
    return {
        'batch_size': 32,
        'epochs': 50,           # 18k data: cosine scheduler 후반 decay 충분히 활용
        'learning_rate': None,
        'weight_decay': None,   # preset-overridden
        'patience': 8,          # 18k data: 수렴이 안정적이지만 느릴 수 있음
        'warmup_epochs': 3,
        'seed': 42,
        'split_seed': 42,
        'split_dir': None,
        'input_size': None,    # will be preset-overridden
        'num_workers': 8,  # DataLoader workers (train)
        'eval_num_workers': 8,  # DataLoader workers (val/test)
        'prefetch_factor': 2,
        'persistent_workers': True,
        'preload_images': True,
        'preload_workers': 16,
        'npy_cache_dir': '/home/cglab/preload_cache_npy',   # preload 결과를 .npz 로 저장 후 재사용
        # cache_images_in_memory 제거됨 — preload_images 만 사용
        'use_bf16': True,  # 기본: bfloat16 autocast (GradScaler 없음, Ampere+ 권장). False면 fp16+GradScaler
        'use_fp32': False,  # True면 autocast 없이 float32 (메모리·시간↑, 수치 가장 안정)
        'allow_tf32': True,
        'use_compile': False,  # 기본 OFF (H200 등에서 torch.compile backward 불안정 가능)
        'matmul_precision': 'high',
        'model_name': None,    # will be preset-overridden
        'dropout': 0.2,        # 기본 dropout 0.2
        'scheduler_type': 'cosine',  # 'plateau', 'cosine', 'cosine_restart', 'step'
        # Normalization:
        # - Default: AUTO (infer from timm model cfg) for best alignment with pretrained backbones.
        # - Fallback: ImageNet mean/std if cfg doesn't provide them.
        # - To force a fixed normalization across models for "fair preprocessing", set norm_mean/norm_std explicitly in config JSON.
        'norm_mean': None,
        'norm_std': None,
        # Augmentation
        'aug_flip_prob': 0.0,          # CXR laterality 이슈를 고려해 기본은 비활성화
        'aug_rot_prob': 0.5,           # ±7° rotation to simulate patient posture variation
        'aug_rot_deg': 7,
        'aug_bc_prob': 0.3,            # brightness / contrast jitter
        'aug_brightness': [0.9, 1.1],
        'aug_contrast': [0.9, 1.1],
        'aug_shift_prob': 0.5,         # ±5% spatial shift
        'aug_shift_limit': 0.05,
        'aug_scale_prob': 0.5,         # 0.9×~1.1× zoom (different capture distances)
        'aug_scale_limit': 0.1,
        'n_folds': 1,
        # Loss function selection
        'loss_type': 'weighted_bce',  # 'ce', 'weighted_ce', 'focal', 'bce', 'weighted_bce'
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        'label_smoothing': 0.0,
        'bce_pos_weight': None,  # None=auto (imbalance_ratio), or specify value
        # Imbalance handling (default: no downsampling; use --downsample-neg-per-pos 2.0 or 1.0 for 2:1 or 1:1)
        'downsample_negative': False,
        'downsample_neg_per_pos': 2.0,
        # threshold tuning
        'thr_mode': 'youden',
        'min_specificity': 0.8,
        'min_sensitivity': 0.8,
        # data leakage guards (group-wise split must be stable)
        'strict_grouping': True,
        # reproducibility controls
        'repro_mode': False,
        'deterministic_algorithms': False,
        # Backbone freeze (linear probe): default OFF — 백본 전체 학습
        'freeze_backbone': False,
        'model_type': 'cxr_single',
        # EVA-X: path to directory containing .pt files, or path to a specific .pt file
        'eva_x_ckpt_dir': None,
    }

def infer_input_size_from_timm(model_name, fallback=512):
    if EVA_X_NAMES and model_name in EVA_X_NAMES:
        return 512
    if _is_rad_dino_model(model_name):
        return RAD_DINO_INPUT_SIZE
    if _is_chexficient_model(model_name):
        return CHEXFICIENT_INPUT_SIZE
    try:
        m = timm.create_model(model_name, pretrained=False, num_classes=0)
        sz = None
        if hasattr(m, 'pretrained_cfg') and isinstance(getattr(m, 'pretrained_cfg'), dict):
            sz = m.pretrained_cfg.get('input_size', None)
        elif hasattr(m, 'default_cfg') and isinstance(getattr(m, 'default_cfg'), dict):
            sz = m.default_cfg.get('input_size', None)
        if sz and isinstance(sz, (list, tuple)) and len(sz) == 3:
            return int(sz[-1])
    except Exception:
        pass
    return int(fallback)

def infer_norm_stats_from_timm(model_name, fallback_mean=None, fallback_std=None):
    """
    Infer mean/std from timm model config without requiring pretrained weight download.
    Returns: (mean_list, std_list, source_str)
    """
    fb_mean = fallback_mean if fallback_mean is not None else [0.485, 0.456, 0.406]
    fb_std = fallback_std if fallback_std is not None else [0.229, 0.224, 0.225]
    if EVA_X_NAMES and model_name in EVA_X_NAMES:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 'evax_imagenet'
    if _is_rad_dino_model(model_name):
        return list(RAD_DINO_NORM_MEAN), list(RAD_DINO_NORM_STD), 'rad_dino_imagenet'
    if _is_chexficient_model(model_name):
        return list(CHEXFICIENT_NORM_MEAN), list(CHEXFICIENT_NORM_STD), 'chexficient_processor'
    try:
        m = timm.create_model(model_name, pretrained=False, num_classes=0)
        cfg = None
        if hasattr(m, 'pretrained_cfg') and isinstance(getattr(m, 'pretrained_cfg'), dict) and getattr(m, 'pretrained_cfg'):
            cfg = m.pretrained_cfg
            src = 'timm.pretrained_cfg'
        elif hasattr(m, 'default_cfg') and isinstance(getattr(m, 'default_cfg'), dict) and getattr(m, 'default_cfg'):
            cfg = m.default_cfg
            src = 'timm.default_cfg'
        else:
            cfg = None
            src = 'fallback'
        if cfg:
            mean = cfg.get('mean', None)
            std = cfg.get('std', None)
            if mean is not None and std is not None:
                mean_l = [float(x) for x in list(mean)]
                std_l = [float(x) for x in list(std)]
                if len(mean_l) == 3 and len(std_l) == 3:
                    return mean_l, std_l, src
    except Exception:
        pass
    return [float(x) for x in fb_mean], [float(x) for x in fb_std], 'fallback_imagenet'


def apply_preset(config, arch, explicit_model_name=None):
    # Explicit model-name highest priority for model_name
    if explicit_model_name and config.get('model_name') in (None, 'AUTO'):
        config['model_name'] = explicit_model_name

    # helper for wd checking
    def _needs_wd_fill(wd):
        if wd in (None, 'AUTO'):
            return True
        try:
            return float(wd) <= 0.0
        except Exception:
            return True
    def _needs_lr_fill(lr):
        if lr in (None, 'AUTO'):
            return True
        try:
            return float(lr) <= 0.0
        except Exception:
            return True
    def _apply_optional_fields(preset_obj):
        # preset에 포함된 추천값을 config에 반영합니다.
        for key in [
            'batch_size', 'warmup_epochs', 'dropout', 'epochs', 'patience',
            'scheduler_type', 'loss_type', 'focal_alpha', 'focal_gamma',
            'label_smoothing', 'downsample_negative', 'downsample_neg_per_pos',
            'bce_pos_weight', 'use_bf16', 'use_fp32',
        ]:
            if key in preset_obj:
                config[key] = preset_obj[key]

    # Default when nothing provided: use convnextv2 preset
    if arch is None and config.get('model_name') in (None, 'AUTO'):
        p = PRESETS['convnextv2']
        config['model_name'] = p['model_name']
        if config.get('input_size') in (None, 'AUTO'):
            config['input_size'] = p['input_size']
        if _needs_wd_fill(config.get('weight_decay')):
            config['weight_decay'] = p['weight_decay']
        if _needs_lr_fill(config.get('learning_rate')):
            config['learning_rate'] = p.get('learning_rate', config.get('learning_rate'))
        _apply_optional_fields(p)
        return

    # If arch matches preset, apply gently (config-first)
    p = PRESETS.get(arch)
    if p:
        if config.get('model_name') in (None, 'AUTO'):
            config['model_name'] = p['model_name']
        if config.get('input_size') in (None, 'AUTO'):
            config['input_size'] = p['input_size']
        if _needs_wd_fill(config.get('weight_decay')):
            config['weight_decay'] = p['weight_decay']
        if _needs_lr_fill(config.get('learning_rate')):
            config['learning_rate'] = p.get('learning_rate', config.get('learning_rate'))
        _apply_optional_fields(p)
        return

    # Unknown arch: treat as timm model name
    if arch is not None:
        if config.get('model_name') in (None, 'AUTO'):
            config['model_name'] = arch
        if config.get('input_size') in (None, 'AUTO'):
            config['input_size'] = infer_input_size_from_timm(arch, fallback=512)
        # weight_decay: keep existing config default unless explicitly unset
        if _needs_wd_fill(config.get('weight_decay')):
            # stay with current default (do not assume special wd)
            wd_default = config.get('weight_decay')
            try:
                wd_val = float(wd_default)
            except Exception:
                wd_val = 1e-4
            if wd_val <= 0.0:
                wd_val = 1e-4
            config['weight_decay'] = wd_val


def main():
    # torchrun이 띄운 DDP 환경에서 rank != 0 프로세스는 stdout을 억제해
    # 터미널 로그 중복 출력을 방지합니다. (stderr는 살려 두어 에러는 표시됨)
    # RANK 환경변수는 dist.init_process_group() 전에 이미 설정되어 있으므로
    # main() 시작 시점에 바로 처리해야 config 로딩·CLI override 출력도 억제됩니다.
    _early_rank = int(os.environ.get('RANK', 0))
    if _early_rank != 0:
        sys.stdout = open(os.devnull, 'w')

    print("🚀 단일 CXR 폐렴(Pneumonia) 이진 분류 학습 (ResNet / DenseNet / ConvNeXtV2 / SwinV2 / DINOv3 / EVA-X / RAD-DINO / CheXficient)", flush=True)
    print("="*60, flush=True)

    # Base config (default values)
    config = build_default_config()

    # CLI parser (우선순위: CLI 인자 > Config file > PRESET > Default Config)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file (overrides defaults/presets; CLI overrides if provided)')
    parser.add_argument('--arch', type=str, default=None, help='Preset key (resnet/resnet152/densenet/convnextv2/convnextv2_base/swint/swint_base/dinov3/dinov3_base/eva_x_small/eva_x_base/rad_dino/chexficient) or any timm model name')
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--data-root', type=str, default=None, help='CXR 데이터 루트 (manifest_pa_medsam3_ok.json 포함). 미지정 시 --data-mode에 따라 자동 설정')
    parser.add_argument(
        '--source-image-root',
        type=str,
        default=None,
        help=(
            'raw 원본 CXR의 새 루트. 클라우드 이전으로 manifest의 source_image_abs_path가 '
            '유효하지 않을 때 source_image_rel_path 앞에 붙입니다.'
        ),
    )
    parser.add_argument('--labels-json', type=str, default=None, help='pneumonia_labels.json 경로 (라벨 보완용). 기본: IEEE_ICCBE/pneumonia_labels.json')
    parser.add_argument(
        '--data-mode', type=str, default='medsam3_seg',
        choices=[
            'raw',
            'medsam3_seg', 'medsam3_crop',
            'chexmask_seg', 'chexmask_crop',
            'medsam3_center', 'medsam3_margin0', 'medsam3_margin10', 'medsam3_margin20',
            'medsam3_soft',
            'masked', 'cropped',  # 하위 호환 alias
        ],
        help=(
            '학습에 사용할 이미지 종류 (기본: medsam3_seg)\n'
            '  raw           : MIMIC-CXR 원본 CXR\n'
            '  medsam3_seg   : MedSAM3 폐 영역만 남긴 CXR  (cxr_medsam3_lung_seg)\n'
            '  medsam3_crop  : MedSAM3 폐 마스크 기준 크롭 (cxr_medsam3_lung_seg_cropped)\n'
            '  chexmask_seg  : ChexMask 폐 영역만 남긴 CXR  (cxr_chexmask_lung_seg)\n'
            '  chexmask_crop : ChexMask 폐 마스크 기준 크롭 (cxr_chexmask_lung_seg_cropped)\n'
            '  medsam3_center / medsam3_margin0 / medsam3_margin10 / medsam3_margin20 / medsam3_soft\n'
            '                : 전처리 대조 (cxr_medsam3_control_*)\n'
            '  masked/cropped: 하위 호환 alias (medsam3_seg/medsam3_crop과 동일)'
        ),
    )
    parser.add_argument(
        '--view', type=str, default='PA',
        choices=['PA', 'AP', 'both', 'ALL'],
        help=(
            '학습에 사용할 촬영 방향 (기본: PA)\n'
            '  PA   : Posterior-Anterior 영상 (cxr_*_lung_seg/)\n'
            '  AP   : Anterior-Posterior 영상  (cxr_*_lung_seg_ap/)\n'
            '  both : PA + AP 합쳐서 학습      (cxr_*_lung_seg_pa_ap/)\n'
            '  ALL  : both와 동일 (하위 호환)'
        ),
    )
    parser.add_argument('--pos-mode', type=str, default='pneumonia', choices=['pneumonia'], help='결과 폴더 분류용 라벨 (폐렴)')
    parser.add_argument('--model-type', type=str, default=None, choices=['cxr_single'], help='단일 CXR 분류 (고정: cxr_single)')
    parser.add_argument('--split-mode', type=str, default='cv5', choices=['cv5'])
    parser.add_argument('--n-folds', type=int, default=None, help='Number of folds for CV (default: 5 when split-mode=cv5)')
    parser.add_argument('--outer-test-ratio', type=float, default=0.15)
    parser.add_argument(
        '--split-dir',
        type=str,
        default=None,
        help=(
            '미리 생성해 둔 patient split CSV 디렉터리 '
            '(예: splits/AP_outer_test.csv, splits/AP_fold_assignment.csv). '
            '지정하면 training seed와 무관하게 동일한 환자 분할을 사용합니다.'
        ),
    )
    parser.add_argument(
        '--data-base-dir',
        type=str,
        default=None,
        help=(
            'cxr_*_{ap,pa} 데이터 폴더들의 상위 디렉터리. '
            'preprocessing parity 검사에 사용 (미지정 시 --data-root 상위 폴더 추정).'
        ),
    )
    parser.add_argument(
        '--skip-preprocessing-parity-check',
        action='store_true',
        help='5 preprocessing condition sample/split parity assert를 건너뜁니다 (권장하지 않음).',
    )
    parser.add_argument(
        '--require-files-for-parity',
        action='store_true',
        help='parity 검사 시 5 preprocessing mode 모두 실제 파일 존재까지 요구 (raw는 --source-image-root 필요).',
    )
    parser.add_argument(
        '--uncertainty-policy',
        type=str,
        default='explicit_only',
        choices=list(UNCERTAINTY_POLICIES),
        help=(
            '학습 단계에서 CheXpert Pneumonia=-1을 다루는 정책 (기본: explicit_only).\n'
            '  explicit_only : 0/1만 학습. 기존 70-run과 동일\n'
            '  u_zero        : train fold 환자의 -1을 0으로 보고 재학습\n'
            '  u_one         : train fold 환자의 -1을 1로 보고 재학습\n'
            '검증과 outer test는 세 정책 모두 기존 explicit 0/1만 사용합니다.'
        ),
    )
    parser.add_argument(
        '--uncertain-data-root',
        type=str,
        default=None,
        help=(
            'Pneumonia=-1 전처리 폴더. 미지정 시 data_base_dir 아래 '
            'cxr_*_uncertain_trainval_{ap,pa}를 data-mode에 맞춰 사용합니다.'
        ),
    )
    parser.add_argument(
        '--uncertain-cohort-csv',
        type=str,
        default=None,
        help=(
            '학습에 넣을 -1 영상 고정 목록. 미지정 시 '
            '{split-dir}/{VIEW}_uncertain_trainval.csv'
        ),
    )
    parser.add_argument(
        '--split-seed',
        type=int,
        default=None,
        help='Patient split 전용 seed (기본 42). training --seed 와 분리해 고정합니다.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Training random seed (초기화, DataLoader, bootstrap 등). Split seed와 별개.',
    )
    parser.add_argument(
        '--export-splits-only',
        action='store_true',
        help=(
            'patient split CSV만 --split-dir 에 저장하고 종료합니다. '
            'reference manifest(raw) 기준으로 AP/PA CSV를 생성할 때 사용합니다.'
        ),
    )
    parser.add_argument(
        '--split-reference-data-root',
        type=str,
        default=None,
        help=(
            '현재 데이터 대신 이 data_root의 manifest에서 subject 단위 outer/CV split을 계산한 뒤 '
            '현재 샘플에 매핑합니다. 전처리별 manifest 크기가 달라도 동일한 환자 분할을 재사용할 때 사용합니다.'
        ),
    )
    parser.add_argument(
        '--split-reference-data-mode',
        type=str,
        default='raw',
        choices=['raw', 'medsam3_seg', 'medsam3_crop', 'chexmask_seg', 'chexmask_crop'],
        help='--split-reference-data-root에서 split 계산에 사용할 manifest/이미지 필드 (기본: raw)',
    )
    parser.add_argument(
        '--no-outer-test', action='store_true',
        help='Disable the outer holdout split (pure 5-fold CV, legacy behavior). Overrides --outer-test-ratio.',
    )
    parser.add_argument('--cv5-inner-811', action='store_true', help='Enable inner 8:1 split inside each cv5 train fold')
    parser.add_argument('--max-folds', type=int, default=None, help='Run only the first N folds in cv5 for quick screening (e.g. 1)')
    parser.add_argument('--allow-cli-override', action='store_true', help='[DEPRECATED] CLI already overrides config when options are provided')
    # Hyperparameters
    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate (e.g., 1e-5, 7e-6)')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate (preset 기본: 0.2)')
    parser.add_argument('--epochs', type=int, default=None, help='Max training epochs (e.g., 40, 50)')
    parser.add_argument('--patience', type=int, default=None, help='Early stopping patience (e.g., 5, 6, 7)')
    parser.add_argument('--warmup-epochs', type=int, default=None, help='Number of warmup epochs (e.g., 2, 5, 10)')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (e.g., 16, 32)')
    parser.add_argument(
        '--input-size',
        type=int,
        default=None,
        help='Input image size (square). Overrides arch preset when set explicitly.',
    )
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader workers for train (default: 8)')
    parser.add_argument('--eval-num-workers', type=int, default=8, help='DataLoader workers for val/test (default: 8)')
    parser.add_argument('--prefetch-factor', type=int, default=None, help='DataLoader prefetch_factor (default: 2)')
    parser.add_argument('--compile', dest='use_compile', action='store_true', help='Use torch.compile(model) for faster training (PyTorch 2.0+, CUDA)')
    parser.add_argument(
        '--no-compile',
        action='store_true',
        dest='no_compile',
        help='Disable torch.compile (default config may enable compile; use to save VRAM / avoid OOM).',
    )
    parser.add_argument(
        '--auto-ddp',
        action='store_true',
        help='GPU 2장 이상일 때만: torch.distributed.run으로 프로세스를 띄워 DDP 학습 (기본: 자동 재실행 안 함).',
    )
    parser.add_argument('--no-ddp', action='store_true', help=argparse.SUPPRESS)  # 레거시 (무시됨; 자동 DDP 기본 OFF)
    parser.add_argument('--fp32', dest='use_fp32_cli', action='store_true', help='float32 학습·검증 (autocast 없음, 메모리·시간 증가)')
    parser.add_argument('--fp16', dest='use_fp16_cli', action='store_true', help='fp16 autocast + GradScaler (이전 기본; BatchNorm CNN에서 NaN 위험 가능)')
    parser.add_argument('--bf16', dest='use_bf16_cli', action='store_true', help='bfloat16 autocast (GradScaler 없음, 기본값과 동일)')
    parser.add_argument('--weight-decay', type=float, default=None, help='Weight decay (e.g., 0.01, 0.05)')
    parser.add_argument('--scheduler', type=str, default=None, 
                        choices=['plateau', 'cosine', 'cosine_restart', 'step'],
                        help='LR scheduler: plateau (default), cosine, cosine_restart, step')
    # Threshold tuning
    parser.add_argument(
        '--thr-mode', type=str, default=None,
        choices=['f1', 'spec', 'sens', 'mcc', 'youden'],
        help=(
            'Threshold tuning mode (default youden). 논문에서 운영점 해석 시:\n'
            '  - youden: Youden\'s J = sens+spec-1 최대 (기본값, 클리니컬 해석이 직관적)\n'
            '  - f1: precision-recall 균형 (imbalanced 에 적합, 해석 난이도 중간)\n'
            '  - spec: 지정한 min_specificity 를 만족하는 최소 threshold (screening)\n'
            '  - sens: 지정한 min_sensitivity 를 만족하는 최대 threshold (triage)\n'
            '  - mcc: Matthews correlation coefficient 최대\n'
            'fold 별 best_threshold.json 에는 f1/youden/spec80/spec90 4개 운영점이 '
            '함께 저장되므로, 학습 후에도 원하는 운영점으로 자유롭게 재보고 가능.'
        ),
    )
    parser.add_argument('--min-sensitivity', type=float, default=None, help='Min sensitivity for thr_mode=sens (e.g. 0.8)')
    parser.add_argument('--min-specificity', type=float, default=None, help='Min specificity for thr_mode=spec')
    # Loss function selection
    parser.add_argument('--loss-type', type=str, default=None, 
                        choices=['ce', 'weighted_ce', 'focal', 'bce', 'weighted_bce'], 
                        help='Loss function type: ce, weighted_ce, focal, bce, weighted_bce')
    parser.add_argument('--focal-alpha', type=float, default=None, help='Focal loss alpha parameter')
    parser.add_argument('--focal-gamma', type=float, default=None, help='Focal loss gamma parameter')
    parser.add_argument('--label-smoothing', type=float, default=None, help='Label smoothing factor (0.0 to 0.2)')
    parser.add_argument('--bce-pos-weight', type=float, default=None, help='BCE positive class weight (None=auto from imbalance ratio)')
    # Downsampling options
    parser.add_argument('--downsample-neg-per-pos', type=float, default=None, help='Downsample negative to positive ratio (e.g., 1.0, 2.0, 3.0, 4.0). If None, uses config/preset value.')
    parser.add_argument('--no-downsample', action='store_true', help='Disable negative downsampling (use all negative samples)')
    # Balanced batch sampler (alternative to downsampling)
    # Augmentation overrides
    parser.add_argument('--aug-flip-prob', type=float, default=None, help='Horizontal flip probability (default: from config; recommended 0.0 for CXR)')
    parser.add_argument('--aug-rot-prob', type=float, default=None, help='Rotation augmentation probability (default: from config)')
    parser.add_argument('--aug-rot-deg', type=float, default=None, help='Max rotation degree (default: from config)')
    parser.add_argument('--aug-bc-prob', type=float, default=None, help='Brightness/Contrast augmentation probability (default: from config)')
    parser.add_argument('--aug-brightness', type=float, nargs=2, default=None, metavar=('LOW', 'HIGH'),
                        help='Brightness range, e.g., --aug-brightness 0.8 1.2 (default: from config)')
    parser.add_argument('--aug-contrast', type=float, nargs=2, default=None, metavar=('LOW', 'HIGH'),
                        help='Contrast range, e.g., --aug-contrast 0.8 1.2 (default: from config)')
    parser.add_argument('--aug-shift-prob', type=float, default=None, help='Shift augmentation probability (default: from config)')
    parser.add_argument('--aug-shift-limit', type=float, default=None, help='Max shift fraction of image size (default: from config, e.g. 0.05 = ±5%%)')
    parser.add_argument('--aug-scale-prob', type=float, default=None, help='Scale augmentation probability (default: from config)')
    parser.add_argument('--aug-scale-limit', type=float, default=None, help='Max scale deviation (default: from config, e.g. 0.1 = 0.9~1.1×)')
    # Backbone freeze (default: unfrozen / full fine-tuning)
    parser.add_argument('--freeze-backbone', action='store_true', default=None,
                        help='[Explicit] Freeze backbone weights and train only the classification head')
    parser.add_argument('--unfreeze-backbone', action='store_true',
                        help='Unfreeze backbone for full fine-tuning (this is already the default)')
    # Resume from a previous run's save_dir (loads last_checkpoint.pth per fold)
    parser.add_argument('--resume-dir', type=str, default=None,
                        help='이전 학습 결과 폴더. fold_N/last_checkpoint.pth에서 재개 (--save-dir 미지정 시 출력도 이 폴더 사용)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='결과 저장 폴더 (미지정 시 새 timestamp 폴더 생성; --resume-dir 과 함께 쓰면 동일 run 이어하기)')
    # EVA-X checkpoint (required when using arch eva_x_tiny / eva_x_small / eva_x_base)
    parser.add_argument('--eva-x-ckpt-dir', type=str, default=None,
                        help='EVA-X .pt 파일 또는 폴더. 생략 시 <repo>/eva-x/<모델명>.pt (예: eva-x/eva_x_small_patch16_merged520k_mim.pt)')
    # Reproducibility / determinism
    parser.add_argument('--repro', action='store_true', help='Enable reproducibility mode (cudnn.benchmark=False, cudnn.deterministic=True)')
    parser.add_argument('--deterministic-algorithms', action='store_true', help='Use torch deterministic algorithms (may be slower / may error on unsupported ops)')
    parser.add_argument('--no-preload-images', dest='preload_images', action='store_false', default=None, help='Disable image preloading (disk I/O every epoch)')
    parser.add_argument('--preload-workers', type=int, default=None, help='preload 병렬 worker 수 (기본 16)')
    parser.add_argument('--npy-cache-dir', dest='npy_cache_dir', type=str, default=None,
                        help='preload 결과를 .npz 캐시로 저장할 디렉터리. 지정하면 첫 실행 시 저장, 이후 실행에서 즉시 로드합니다.')
    # --cache-images-in-memory 제거됨 (preload_images만 사용)
    # Optional toggles (W&B 기본 비활성화; 켜려면 --wandb)
    parser.add_argument('--wandb', action='store_true', help='Weights & Biases 로깅 활성화 (기본: 끔)')
    parser.add_argument('--no-wandb', action='store_true', help=argparse.SUPPRESS)  # 하위 호환: 무시됨
    parser.add_argument('--wandb-project', type=str, default='cxr-pneumonia')
    parser.add_argument('--wandb-entity', type=str, default=None)
    parser.add_argument('--wandb-api-key', type=str, default=None)
    parser.add_argument('--wandb-host', type=str, default=None)
    parser.add_argument('--wandb-relogin', action='store_true')
    parser.add_argument('--no-dp', action='store_true')
    parser.add_argument('--force-dp', action='store_true')
    parser.add_argument(
        '--no-strict-grouping',
        action='store_true',
        help='Allow grouping fallback if manifest/path mapping is incomplete (NOT recommended; leakage risk). Default: strict grouping.',
    )
    args, _ = parser.parse_known_args()

    # --no-outer-test: outer holdout 분리를 끔 (pure 5-fold CV)
    if getattr(args, 'no_outer_test', False):
        args.outer_test_ratio = 0.0
        print("🔧 --no-outer-test: outer holdout 비활성화 (pure 5-fold CV)")

    # Apply architecture preset (preset values override defaults)
    apply_preset(config, args.arch, explicit_model_name=args.model_name)

    # Load config file (overrides defaults/presets). JSON only.
    if getattr(args, 'config', None):
        cfg_path = str(args.config)
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg_obj = json.load(f)
            if isinstance(cfg_obj, dict):
                config.update(cfg_obj)
                print(f"🧾 Loaded config file: {cfg_path}")
            else:
                print(f"⚠️ Config file must be a JSON object (dict). Ignored: {cfg_path}")
        except Exception as e:
            print(f"⚠️ Failed to load config file ({cfg_path}): {e}")

    # CLI always wins when explicitly provided (including model-name)
    if args.model_name is not None:
        config['model_name'] = args.model_name
        print(f"🔧 CLI Override: model_name={args.model_name}")
    if getattr(args, 'eva_x_ckpt_dir', None) is not None:
        config['eva_x_ckpt_dir'] = args.eva_x_ckpt_dir
        print(f"🔧 CLI Override: eva_x_ckpt_dir={args.eva_x_ckpt_dir}")
    # EVA-X: 미지정 시 레포 ``eva-x/<모델>.pt`` (예: eva-x/eva_x_small_patch16_merged520k_mim.pt)
    if EVA_X_NAMES and config.get('model_name') in EVA_X_NAMES and not config.get('eva_x_ckpt_dir'):
        config['eva_x_ckpt_dir'] = default_eva_x_ckpt_path(config['model_name'])
        print(f"📂 EVA-X 체크포인트 기본: {config['eva_x_ckpt_dir']}")
    if getattr(args, 'model_type', None) is not None:
        config['model_type'] = 'cxr_single'
        print("🔧 CLI Override: model_type=cxr_single (단일 CXR 고정)")
    # 구버전: time_diff sinusoidal embedding 제거됨 — JSON/summary에 남아 있어도 무시
    for _legacy in ('time_emb_dim', 'time_diff_max_hours', 'time_diff_default_hours'):
        config.pop(_legacy, None)

    # Normalization: infer mean/std from timm config by default (unless explicitly provided via config JSON)
    if config.get('norm_mean') is None or config.get('norm_std') is None:
        nm, ns, src = infer_norm_stats_from_timm(
            config.get('model_name'),
            fallback_mean=[0.485, 0.456, 0.406],
            fallback_std=[0.229, 0.224, 0.225],
        )
        config['norm_mean'] = nm
        config['norm_std'] = ns
        print(f"🧼 Norm stats ({src}): mean={nm}, std={ns}")

    # Apply CLI overrides for hyperparameters (CLI overrides everything including presets)
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
        print(f"🔧 CLI Override: learning_rate={args.learning_rate}")
    if args.dropout is not None:
        config['dropout'] = args.dropout
        print(f"🔧 CLI Override: dropout={args.dropout}")
    if getattr(args, 'epochs', None) is not None:
        config['epochs'] = args.epochs
        print(f"🔧 CLI Override: epochs={args.epochs}")
    if getattr(args, 'patience', None) is not None:
        config['patience'] = args.patience
        print(f"🔧 CLI Override: patience={args.patience}")
    if args.warmup_epochs is not None:
        config['warmup_epochs'] = args.warmup_epochs
        print(f"🔧 CLI Override: warmup_epochs={args.warmup_epochs}")
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
        print(f"🔧 CLI Override: batch_size={args.batch_size}")
    if getattr(args, 'input_size', None) is not None:
        config['input_size'] = int(args.input_size)
        print(f"🔧 CLI Override: input_size={config['input_size']}")
    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
        print(f"🔧 CLI Override: num_workers={args.num_workers}")
    if getattr(args, 'eval_num_workers', None) is not None:
        config['eval_num_workers'] = int(args.eval_num_workers)
        print(f"🔧 CLI Override: eval_num_workers={config['eval_num_workers']}")
    if getattr(args, 'prefetch_factor', None) is not None:
        config['prefetch_factor'] = int(args.prefetch_factor)
        print(f"🔧 CLI Override: prefetch_factor={config['prefetch_factor']}")
    if getattr(args, 'no_compile', False):
        config['use_compile'] = False
        print("🔧 CLI: torch.compile(model) disabled (--no-compile)")
    elif getattr(args, 'use_compile', False):
        config['use_compile'] = True
        print(f"🔧 CLI: torch.compile(model) enabled")
    if getattr(args, 'use_fp32_cli', False):
        config['use_fp32'] = True
        config['use_bf16'] = False
        print("🔧 CLI Override: use_fp32=True → float32 (autocast 없음)")
    elif getattr(args, 'use_fp16_cli', False):
        config['use_bf16'] = False
        config['use_fp32'] = False
        print("🔧 CLI Override: use_bf16=False → fp16 autocast + GradScaler")
    elif getattr(args, 'use_bf16_cli', False):
        config['use_bf16'] = True
        config['use_fp32'] = False
        print("🔧 CLI Override: use_bf16=True → bfloat16 autocast (GradScaler 없음)")
    _arch_key = str(args.arch or '').lower()
    _mn = str(config.get('model_name') or '')
    if (
        _arch_key in ('rad_dino', 'rad-dino')
        or _is_rad_dino_model(_mn)
    ) and config.get('use_bf16', False) and not config.get('use_fp32', False):
        config['use_bf16'] = False
        config['use_fp32'] = True
        print("🔧 RAD-DINO: bf16 backward NaN → float32 학습으로 전환")
    if getattr(args, 'preload_images', None) is not None:
        config['preload_images'] = bool(args.preload_images)
        print(f"🔧 CLI Override: preload_images={config['preload_images']}")
    if getattr(args, 'preload_workers', None) is not None:
        config['preload_workers'] = int(args.preload_workers)
        print(f"🔧 CLI Override: preload_workers={config['preload_workers']}")
    if getattr(args, 'npy_cache_dir', None) is not None:
        config['npy_cache_dir'] = args.npy_cache_dir
        print(f"🔧 CLI Override: npy_cache_dir={config['npy_cache_dir']}")
    # --cache-images-in-memory 제거됨 — preload_images만 사용
    if args.weight_decay is not None:
        config['weight_decay'] = args.weight_decay
        print(f"🔧 CLI Override: weight_decay={args.weight_decay}")
    if args.scheduler is not None:
        config['scheduler_type'] = args.scheduler
        print(f"🔧 CLI Override: scheduler_type={args.scheduler}")
    if args.thr_mode is not None:
        config['thr_mode'] = args.thr_mode
        print(f"🔧 CLI Override: thr_mode={args.thr_mode}")
    if args.min_sensitivity is not None:
        config['min_sensitivity'] = args.min_sensitivity
        print(f"🔧 CLI Override: min_sensitivity={args.min_sensitivity}")
    if args.min_specificity is not None:
        config['min_specificity'] = args.min_specificity
        print(f"🔧 CLI Override: min_specificity={args.min_specificity}")

    # Apply CLI overrides for loss function (CLI overrides everything)
    if args.loss_type is not None:
        config['loss_type'] = args.loss_type
        print(f"🔧 CLI Override: loss_type={args.loss_type}")
    if args.focal_alpha is not None:
        config['focal_alpha'] = args.focal_alpha
        print(f"🔧 CLI Override: focal_alpha={args.focal_alpha}")
    if args.focal_gamma is not None:
        config['focal_gamma'] = args.focal_gamma
        print(f"🔧 CLI Override: focal_gamma={args.focal_gamma}")
    if args.label_smoothing is not None:
        config['label_smoothing'] = args.label_smoothing
        print(f"🔧 CLI Override: label_smoothing={args.label_smoothing}")
    if args.bce_pos_weight is not None:
        config['bce_pos_weight'] = args.bce_pos_weight
        print(f"🔧 CLI Override: bce_pos_weight={args.bce_pos_weight}")
    
    # Apply CLI override for backbone freeze (default=True in config)
    if getattr(args, 'unfreeze_backbone', False):
        config['freeze_backbone'] = False
        print(f"🔥 CLI Override: freeze_backbone=False (full fine-tuning)")
    elif getattr(args, 'freeze_backbone', None):
        config['freeze_backbone'] = True
        print(f"🧊 CLI Override: freeze_backbone=True (linear probe)")
    # else: use config default (True)

    # Apply CLI overrides for downsampling (CLI overrides everything)
    if args.no_downsample:
        config['downsample_negative'] = False
        print(f"🔧 CLI Override: downsample_negative=False (disabled)")
    elif args.downsample_neg_per_pos is not None:
        config['downsample_negative'] = True
        config['downsample_neg_per_pos'] = args.downsample_neg_per_pos
        print(f"🔧 CLI Override: downsample_neg_per_pos={args.downsample_neg_per_pos}")
    if args.aug_flip_prob is not None:
        config['aug_flip_prob'] = float(args.aug_flip_prob)
        print(f"🔧 CLI Override: aug_flip_prob={config['aug_flip_prob']}")
    if getattr(args, 'aug_rot_prob', None) is not None:
        config['aug_rot_prob'] = float(args.aug_rot_prob)
        print(f"🔧 CLI Override: aug_rot_prob={config['aug_rot_prob']}")
    if getattr(args, 'aug_rot_deg', None) is not None:
        config['aug_rot_deg'] = float(args.aug_rot_deg)
        print(f"🔧 CLI Override: aug_rot_deg={config['aug_rot_deg']}")
    if getattr(args, 'aug_bc_prob', None) is not None:
        config['aug_bc_prob'] = float(args.aug_bc_prob)
        print(f"🔧 CLI Override: aug_bc_prob={config['aug_bc_prob']}")
    if getattr(args, 'aug_brightness', None) is not None:
        try:
            lo, hi = float(args.aug_brightness[0]), float(args.aug_brightness[1])
            config['aug_brightness'] = [lo, hi]
            print(f"🔧 CLI Override: aug_brightness={config['aug_brightness']}")
        except Exception:
            pass
    if getattr(args, 'aug_contrast', None) is not None:
        try:
            lo, hi = float(args.aug_contrast[0]), float(args.aug_contrast[1])
            config['aug_contrast'] = [lo, hi]
            print(f"🔧 CLI Override: aug_contrast={config['aug_contrast']}")
        except Exception:
            pass
    if getattr(args, 'aug_shift_prob', None) is not None:
        config['aug_shift_prob'] = float(args.aug_shift_prob)
        print(f"🔧 CLI Override: aug_shift_prob={config['aug_shift_prob']}")
    if getattr(args, 'aug_shift_limit', None) is not None:
        config['aug_shift_limit'] = float(args.aug_shift_limit)
        print(f"🔧 CLI Override: aug_shift_limit={config['aug_shift_limit']}")
    if getattr(args, 'aug_scale_prob', None) is not None:
        config['aug_scale_prob'] = float(args.aug_scale_prob)
        print(f"🔧 CLI Override: aug_scale_prob={config['aug_scale_prob']}")
    if getattr(args, 'aug_scale_limit', None) is not None:
        config['aug_scale_limit'] = float(args.aug_scale_limit)
        print(f"🔧 CLI Override: aug_scale_limit={config['aug_scale_limit']}")
    if getattr(args, 'repro', False):
        config['repro_mode'] = True
        print("🔧 CLI Override: repro_mode=True")
    if getattr(args, 'deterministic_algorithms', False):
        config['deterministic_algorithms'] = True
        print("🔧 CLI Override: deterministic_algorithms=True")
    if getattr(args, 'no_strict_grouping', False):
        config['strict_grouping'] = False
        print("🔧 CLI Override: strict_grouping=False (WARNING: leakage risk)")

    if getattr(args, 'seed', None) is not None:
        config['seed'] = int(args.seed)
        print(f"🔧 CLI Override: training seed={config['seed']}")
    if getattr(args, 'split_seed', None) is not None:
        config['split_seed'] = int(args.split_seed)
        print(f"🔧 CLI Override: split_seed={config['split_seed']}")
    else:
        config.setdefault('split_seed', 42)
    if getattr(args, 'split_dir', None) is not None:
        config['split_dir'] = os.path.normpath(str(args.split_dir))
        print(f"🔧 CLI Override: split_dir={config['split_dir']}")
    if getattr(args, 'data_base_dir', None) is not None:
        config['data_base_dir'] = os.path.normpath(str(args.data_base_dir))
        print(f"🔧 CLI Override: data_base_dir={config['data_base_dir']}")

    # Get view and pos-mode (needed for result directory name)
    view = str(config.get('view', args.view or 'PA')).upper()
    # ALL → BOTH 정규화 (하위 호환)
    if view == "ALL":
        view = "BOTH"
    pos_mode = str(config.get('pos_mode', args.pos_mode or 'pneumonia')).lower()

    # view → view_tag(manifest 파일명), view_suffix(폴더 접미사) 결정
    _VIEW_TAG    = {"PA": "pa",    "AP": "ap",    "BOTH": "pa_ap"}
    _VIEW_SUFFIX = {"PA": "",      "AP": "_ap",   "BOTH": "_pa_ap"}
    view_tag    = _VIEW_TAG.get(view, "pa")
    view_suffix = _VIEW_SUFFIX.get(view, "")
    # Split mode & CV folds (fix: cv5 must actually run)
    split_mode = 'cv5'
    # If user specified n_folds, honor it; otherwise default to 5 when running cv5
    if getattr(args, 'n_folds', None) is not None:
        config['n_folds'] = int(args.n_folds)
    else:
        try:
            nf = int(config.get('n_folds', 1))
        except Exception:
            nf = 1
        if nf <= 1:
            config['n_folds'] = 5
    max_folds = getattr(args, 'max_folds', None)
    if max_folds is not None:
        max_folds = int(max_folds)
        if max_folds < 1:
            raise ValueError("--max-folds must be >= 1")
        if max_folds > int(config['n_folds']):
            print(f"⚠️ --max-folds={max_folds} > n_folds={config['n_folds']}; clipping to {config['n_folds']}")
            max_folds = int(config['n_folds'])
    config['max_folds'] = max_folds
    # Default cv5: no inner split / no outer test unless explicitly enabled by CLI

    config['image_only'] = True

    # data_mode: CLI > config > 기본 'medsam3_seg'  ← save_dir 보다 먼저 결정해야 함
    data_mode = str(getattr(args, 'data_mode', None) or config.get('data_mode') or 'medsam3_seg').lower()
    # 하위 호환 alias 정규화 (masked → medsam3_seg, cropped → medsam3_crop)
    _DM_ALIAS = {"masked": "medsam3_seg", "cropped": "medsam3_crop"}
    data_mode = _DM_ALIAS.get(data_mode, data_mode)
    _VALID_MODES = (
        'raw', 'medsam3_seg', 'medsam3_crop', 'chexmask_seg', 'chexmask_crop',
        *CONTROL_PREPROCESSING_MODES,
    )
    if data_mode not in _VALID_MODES:
        raise ValueError(
            f"--data-mode는 {' / '.join(_VALID_MODES)} 중 하나여야 합니다: {data_mode!r}"
        )
    config['data_mode'] = data_mode

    uncertainty_policy = str(getattr(args, 'uncertainty_policy', None) or 'explicit_only')
    if uncertainty_policy not in UNCERTAINTY_POLICIES:
        raise ValueError(
            f"--uncertainty-policy는 {' / '.join(UNCERTAINTY_POLICIES)} 중 하나여야 합니다: "
            f"{uncertainty_policy!r}"
        )
    config['uncertainty_policy'] = uncertainty_policy
    config['uncertainty_mapped_label'] = UNCERTAINTY_POLICY_LABEL.get(uncertainty_policy)

    # 결과 폴더: results_pneumonia/… (data_mode + max_folds 있으면 폴더명에 반영)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _fold_label = config['max_folds'] if config.get('max_folds') is not None else config['n_folds']
    _default_tag = f"cv_{_fold_label}fold"
    split_tag = str(config.get('split_tag') or _default_tag)
    # 결과 폴더명에 holdout 여부·비율을 명시 (스윕/논문 재현 시 한눈에 구분)
    _otr_for_name = float(getattr(args, 'outer_test_ratio', 0.0) or 0.0)
    if split_mode == 'cv5' and _otr_for_name > 0.0 and 'holdout' not in split_tag.lower():
        _h_pct = int(round(_otr_for_name * 100))
        split_tag = f"{split_tag}_holdout{_h_pct}pct"
    if uncertainty_policy != 'explicit_only':
        _policy_tag = 'uzero' if uncertainty_policy == 'u_zero' else 'uone'
        if not split_tag.endswith(f"_{_policy_tag}"):
            split_tag = f"{split_tag}_{_policy_tag}"
    _resume_dir_arg = getattr(args, 'resume_dir', None)
    _save_dir_arg = getattr(args, 'save_dir', None)
    if _save_dir_arg:
        save_dir = os.path.normpath(_save_dir_arg)
        if not os.path.isabs(save_dir):
            save_dir = os.path.normpath(os.path.join(os.getcwd(), save_dir))
    elif _resume_dir_arg:
        save_dir = os.path.normpath(_resume_dir_arg)
        if not os.path.isabs(save_dir):
            save_dir = os.path.normpath(os.path.join(os.getcwd(), save_dir))
    else:
        _results_root = os.environ.get('RESULTS_ROOT', 'results_pneumonia')
        if not os.path.isabs(_results_root):
            _results_root = os.path.normpath(os.path.join(os.getcwd(), _results_root))
        save_dir = f"{_results_root}/{timestamp}_{_model_tag_for_save_dir(config, args.arch)}_{view}_{data_mode}_{split_tag}"
    os.makedirs(save_dir, exist_ok=True)

    # Device (DDP: torchrun으로 띄우면 프로세스당 device 설정)
    use_ddp = ('RANK' in os.environ and 'WORLD_SIZE' in os.environ)
    if use_ddp:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        world_size = int(os.environ['WORLD_SIZE'])
        backend = 'nccl'
        # 결과를 NTFS(/mnt/d) 같은 느린 마운트에 저장하면 rank 0 의 체크포인트 쓰기가
        # 기본 10분을 넘길 수 있어, collective 대기 한도를 넉넉히 잡는다.
        ddp_timeout_min = int(os.environ.get('DDP_TIMEOUT_MIN', '60'))
        ddp_timeout = timedelta(minutes=ddp_timeout_min)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if torch.cuda.is_available():
            try:
                dist.init_process_group(
                    backend=backend,
                    init_method='env://',
                    timeout=ddp_timeout,
                    device_id=torch.device(f'cuda:{local_rank}'),
                )
            except TypeError:
                # Older PyTorch versions do not support device_id here.
                dist.init_process_group(backend=backend, init_method='env://', timeout=ddp_timeout)
        else:
            dist.init_process_group(backend=backend, init_method='env://', timeout=ddp_timeout)
        device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        if rank == 0:
            print(f"📱 DDP: world_size={world_size}, backend={backend}, device={device}, timeout={ddp_timeout_min}min")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Device: {device}")
    if device.type == 'cuda':
        if bool(config.get('allow_tf32', True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision(str(config.get('matmul_precision', 'high')))
        except Exception:
            pass
        # Reproducibility mode: deterministic > speed
        if bool(config.get('repro_mode', False)):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            try:
                if bool(config.get('deterministic_algorithms', False)):
                    torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            print("🧪 Repro mode ON: cudnn.benchmark=False, cudnn.deterministic=True")
        else:
            torch.backends.cudnn.benchmark = True

    # DataLoader common kwargs
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(config['seed']))

    train_num_workers = int(config.get('num_workers', 8))
    eval_num_workers = int(config.get('eval_num_workers', 8))
    loader_prefetch_factor = int(config.get('prefetch_factor', 2))
    persistent_workers = bool(config.get('persistent_workers', True))

    def _make_loader_kwargs(num_workers: int) -> Dict[str, Any]:
        kwargs = {
            'batch_size': config['batch_size'],
            'num_workers': int(num_workers),
            'worker_init_fn': _SeedWorker(config['seed']),
            'generator': loader_generator,
        }
        if device.type == 'cuda':
            kwargs['pin_memory'] = True
        if int(num_workers) > 0:
            kwargs['persistent_workers'] = persistent_workers
            kwargs['prefetch_factor'] = loader_prefetch_factor
        return kwargs

    loader_kwargs = _make_loader_kwargs(train_num_workers)
    eval_loader_kwargs = _make_loader_kwargs(eval_num_workers)

    DatasetCls = SingleCXRDataset
    config['model_type'] = 'cxr_single'
    ModelCls = CXRSpatialClassifier
    train_epoch_fn = train_epoch_single
    validate_epoch_fn = validate_epoch_single

    # 데이터 루트 (CLI > config > data_mode+view별 기본 경로)
    # view_suffix: PA="" / AP="_ap" / BOTH="_pa_ap"
    _default_root_by_mode = {
        'raw':           os.path.join(DATA_BASE_DIR, "cxr_medsam3_lung_seg"        + view_suffix),
        'medsam3_seg':   os.path.join(DATA_BASE_DIR, "cxr_medsam3_lung_seg"        + view_suffix),
        'medsam3_crop':  os.path.join(DATA_BASE_DIR, "cxr_medsam3_lung_seg_cropped" + view_suffix),
        'chexmask_seg':  os.path.join(DATA_BASE_DIR, "cxr_chexmask_lung_seg"       + view_suffix),
        'chexmask_crop': os.path.join(DATA_BASE_DIR, "cxr_chexmask_lung_seg_cropped" + view_suffix),
        'medsam3_center':   os.path.join(DATA_BASE_DIR, f"cxr_medsam3_control_center_{view_tag}"),
        'medsam3_margin0':  os.path.join(DATA_BASE_DIR, f"cxr_medsam3_control_margin0_{view_tag}"),
        'medsam3_margin10': os.path.join(DATA_BASE_DIR, f"cxr_medsam3_control_margin10_{view_tag}"),
        'medsam3_margin20': os.path.join(DATA_BASE_DIR, f"cxr_medsam3_control_margin20_{view_tag}"),
        'medsam3_soft':     os.path.join(DATA_BASE_DIR, f"cxr_medsam3_control_soft_{view_tag}"),
    }
    if args.data_root:
        data_root = args.data_root
    elif config.get('data_root'):
        data_root = str(config.get('data_root'))
    else:
        data_root = _default_root_by_mode[data_mode]

    # 라벨 JSON (CLI > config > 기본: pneumonia_labels.json)
    labels_json_path = getattr(args, 'labels_json', None)
    if not labels_json_path:
        labels_json_path = config.get('labels_json_path')
    if not labels_json_path:
        _default_labels = os.path.join(DATA_BASE_DIR, "pneumonia_labels.json")
        if os.path.isfile(_default_labels):
            labels_json_path = _default_labels

    # Persist key run paths in config for reproducibility / results audit
    data_root = os.path.normpath(str(data_root))
    config['data_root'] = data_root
    source_image_root = getattr(args, 'source_image_root', None)
    if source_image_root:
        source_image_root = os.path.normpath(str(source_image_root))
    config['source_image_root'] = source_image_root
    split_reference_data_root = getattr(args, 'split_reference_data_root', None)
    if split_reference_data_root:
        split_reference_data_root = os.path.normpath(str(split_reference_data_root))
    config['split_reference_data_root'] = split_reference_data_root
    config['split_reference_data_mode'] = str(
        getattr(args, 'split_reference_data_mode', None) or 'raw'
    )
    config['save_dir'] = save_dir
    config['split_mode'] = split_mode
    config['split_tag'] = split_tag
    config['view'] = view
    config['pos_mode'] = pos_mode
    config['arch'] = str(getattr(args, 'arch', '') or config.get('arch', ''))

    # Seed
    torch.manual_seed(config['seed']); np.random.seed(config['seed']); random.seed(config['seed'])

    # Run metadata (written once + embedded into results)
    run_meta = build_run_meta(
        data_root=data_root,
        save_dir=save_dir,
        view=view,
        pos_mode=pos_mode,
        split_mode=split_mode,
        split_tag=split_tag,
        config=config,
        data_mode=data_mode,
        view_tag=view_tag,
    )
    if rank == 0:
        try:
            with open(os.path.join(save_dir, 'run_meta.json'), 'w', encoding='utf-8') as f:
                json.dump(run_meta, f, indent=2)
        except Exception:
            pass

    # Print config (config-first, rank 0 only)
    if rank == 0:
        print("\n📋 학습 설정:")
        print(f"   Arch: {args.arch}")
        print(f"   Model: {config['model_name']} (dropout={config['dropout']})")
        print(f"   Model Type: {config.get('model_type', 'cxr_single')}")
        print(f"   Freeze Backbone: {config.get('freeze_backbone', False)} {'(linear probe)' if config.get('freeze_backbone', False) else '(full fine-tuning)'}")
        print(f"   View: {view}")
        print(f"   Task: Pneumonia Classification (폐렴 이진 분류)")
        _MODE_DESC = {
            'raw':           '원본 CXR',
            'medsam3_seg':   'MedSAM3 폐마스크',
            'medsam3_crop':  'MedSAM3 폐크롭',
            'chexmask_seg':  'ChexMask 폐마스크',
            'chexmask_crop': 'ChexMask 폐크롭',
            'medsam3_center':   'MedSAM3 가운데 크롭 (본 실험 크롭과 같은 가로·세로)',
            'medsam3_margin0':  'MedSAM3 마진 0% 크롭',
            'medsam3_margin10': 'MedSAM3 마진 10% 크롭',
            'medsam3_margin20': 'MedSAM3 마진 20% 크롭',
            'medsam3_soft':     'MedSAM3 soft mask',
        }
        print(f"   Data Mode: {data_mode}  ({_MODE_DESC.get(data_mode, data_mode)})")
        print(f"   Data Root: {data_root}")
        if source_image_root:
            print(f"   Raw Source Root: {source_image_root}")
        print(f"   Labels JSON: {labels_json_path if labels_json_path else '(manifest 내장 라벨 사용)'}")
        print(f"   Input Size: {config['input_size']}")
        print(f"   Batch Size: {config['batch_size']}")
        print(f"   Epochs: {config['epochs']}")
        print(f"   Learning Rate: {config['learning_rate']}")
        print(f"   Weight Decay: {config['weight_decay']}")
        print(f"   Patience: {config['patience']}")
        print(f"   Warmup Epochs: {config['warmup_epochs']}")
        print(f"   Training Seed: {config['seed']}")
        print(f"   Split Seed: {config.get('split_seed', 42)}")
        if config.get('split_dir'):
            print(f"   Split Dir: {config['split_dir']}")
        print(f"   Num Workers: {config['num_workers']}")
        print(f"   Eval Num Workers: {config.get('eval_num_workers', 8)}")
        print(f"   Prefetch Factor: {config.get('prefetch_factor', 2)}, Persistent Workers: {config.get('persistent_workers', True)}")
        print(f"   Preload Images: {config.get('preload_images', True)}, Preload Workers: {config.get('preload_workers', 16)}")
        print(f"   Augment: flip={config['aug_flip_prob']}, rot_p={config['aug_rot_prob']}, rot_deg={config['aug_rot_deg']}, "
              f"bc_p={config['aug_bc_prob']}, shift_p={config['aug_shift_prob']}, shift_lim={config['aug_shift_limit']}, "
              f"scale_p={config['aug_scale_prob']}, scale_lim={config['aug_scale_limit']}")
        print(f"   N Folds: {config['n_folds']}")
        print(f"   Loss Type: {config.get('loss_type', 'ce')} (alpha={config['focal_alpha']}, gamma={config['focal_gamma']}, smoothing={config['label_smoothing']})")
        print(f"   Uncertainty training policy: {config.get('uncertainty_policy', 'explicit_only')}")
        print(f"   Downsample Negative: {config['downsample_negative']} (neg_per_pos={config['downsample_neg_per_pos']})")
        print(f"   Split Mode: {split_mode}")
        if split_mode == 'cv5':
            print(f"   CV5 Inner 8:1:1: {getattr(args, 'cv5_inner_811', False)}, Outer Test Ratio: {getattr(args, 'outer_test_ratio', 0.0)}")
            print(f"   Max Folds To Run: {config.get('max_folds') if config.get('max_folds') is not None else 'all'}")
        print(f"   Save Dir: {save_dir}")
        if _resume_dir_arg:
            print(f"   Resume Dir: {_resume_dir_arg} (완료된 fold는 건너뜀, 미완료 fold는 last_checkpoint.pth에서 재개)")
        print()

    # Load data
    strict_grouping = bool(config.get('strict_grouping', True))
    all_items, all_labels, all_groups = load_all_cxr_samples_with_groups(
        data_root,
        strict_grouping=strict_grouping,
        labels_json_path=labels_json_path,
        data_mode=data_mode,
        view_tag=view_tag,
        source_image_root=source_image_root,
    )
    if rank == 0:
        print(f"📊 Total CXR samples: {len(all_items)}, Pneumonia(+): {sum(all_labels)}, Normal(-): {len(all_labels)-sum(all_labels)}")
        print(f"📊 Unique subjects(groups): {len(set(all_groups))}")

    split_items, split_labels, split_groups = all_items, all_labels, all_groups
    if split_reference_data_root:
        split_items, split_labels, split_groups = load_all_cxr_samples_with_groups(
            split_reference_data_root,
            strict_grouping=strict_grouping,
            labels_json_path=labels_json_path,
            data_mode=config['split_reference_data_mode'],
            view_tag=view_tag,
            source_image_root=source_image_root,
            require_files=False,
        )
        if rank == 0:
            print(
                "🔗 Reference subject split enabled: "
                f"root={split_reference_data_root}, mode={config['split_reference_data_mode']}"
            )
            print(
                f"🔗 Reference samples={len(split_items)}, subjects={len(set(split_groups))}; "
                f"active samples={len(all_items)}, subjects={len(set(all_groups))}"
            )

    split_seed = int(config.get('split_seed', 42))
    split_dir = config.get('split_dir')
    if split_dir:
        split_dir = os.path.normpath(str(split_dir))

    if (
        rank == 0
        and view in ("AP", "PA")
        and not getattr(args, 'export_splits_only', False)
        and not getattr(args, 'skip_preprocessing_parity_check', False)
    ):
        data_base_dir = config.get('data_base_dir') or infer_data_base_dir(data_root)
        config['data_base_dir'] = data_base_dir
        print(f"🔍 Preprocessing parity check: view={view}, data_base_dir={data_base_dir}")
        assert_preprocessing_sample_parity(
            view=view,
            data_base_dir=data_base_dir,
            split_dir=split_dir,
            labels_json_path=labels_json_path,
            source_image_root=source_image_root,
            strict_grouping=strict_grouping,
            current_data_mode=data_mode,
            current_n_samples=len(all_items),
            require_files=bool(getattr(args, 'require_files_for_parity', False)),
        )
    if use_ddp:
        dist.barrier()

    if getattr(args, 'export_splits_only', False):
        if rank != 0:
            if use_ddp:
                dist.barrier()
            return
        if not split_dir:
            raise ValueError("--export-splits-only requires --split-dir")
        use_outer_for_export = float(getattr(args, 'outer_test_ratio', 0.0) or 0.0) > 0.0
        export_root = split_reference_data_root or data_root
        export_mode = config['split_reference_data_mode'] if split_reference_data_root else 'raw'
        export_records, export_manifest = load_manifest_split_records(
            export_root,
            labels_json_path=labels_json_path,
            data_mode=export_mode,
            view_tag=view_tag,
            strict_grouping=strict_grouping,
        )
        export_items = [(r["dicom_id"], r["label"], r["dicom_id"]) for r in export_records]
        export_labels = [r["label"] for r in export_records]
        export_groups = [r["group_id"] for r in export_records]
        export_assignment = build_grouped_split_assignment(
            export_items,
            export_labels,
            export_groups,
            n_folds=int(config['n_folds']),
            outer_test_ratio=float(args.outer_test_ratio) if use_outer_for_export else 0.0,
            seed=split_seed,
        )
        split_meta = {
            "view": view,
            "split_seed": split_seed,
            "n_folds": int(config['n_folds']),
            "outer_test_ratio": float(args.outer_test_ratio) if use_outer_for_export else 0.0,
            "reference_data_root": export_root,
            "reference_data_mode": export_mode,
            "reference_manifest": export_manifest,
            "outer_split_method": export_assignment.get("outer_method"),
            "reference_n_images": export_assignment.get("reference_n_images"),
            "reference_n_subjects": export_assignment.get("reference_n_subjects"),
        }
        outer_csv, fold_csv = save_fixed_split_csvs(
            split_dir=split_dir,
            view=view,
            records=export_records,
            assignment=export_assignment,
            meta=split_meta,
        )
        print(f"✅ Fixed split CSV saved (split_seed={split_seed})")
        print(f"   outer_test      : {outer_csv}")
        print(f"   fold_assignment : {fold_csv}")
        return

    # explicit 0/1만으로 parity·split 기준을 끝낸 뒤에 -1 영상을 붙입니다.
    # explicit_only에서는 이 블록이 샘플 목록을 바꾸지 않습니다.
    n_explicit = len(all_items)
    uncertainty_audit: Dict[str, Any] = {
        "analysis": "training_policy_sensitivity",
        "policy": uncertainty_policy,
        "mapped_label": config.get("uncertainty_mapped_label"),
        "n_explicit_images": int(n_explicit),
        "n_added_images": 0,
        "evaluation_endpoint": "explicit_0_1_outer_test",
        "per_fold": [],
    }
    if uncertainty_policy != "explicit_only":
        if view not in ("AP", "PA"):
            raise ValueError("--uncertainty-policy u_zero/u_one 은 --view AP 또는 PA에서만 사용할 수 있습니다.")
        if not split_dir:
            raise ValueError("--uncertainty-policy u_zero/u_one 에는 고정 split인 --split-dir 이 필요합니다.")
        cohort_csv = getattr(args, "uncertain_cohort_csv", None) or os.path.join(
            split_dir, f"{view}_uncertain_trainval.csv"
        )
        cohort_csv = os.path.normpath(str(cohort_csv))
        if not os.path.isfile(cohort_csv):
            raise FileNotFoundError(
                f"uncertainty cohort CSV가 없습니다: {cohort_csv}\n"
                "먼저 build_uncertain_endpoint_labels.py --scope trainval 과 "
                "finalize_uncertain_trainval_cohort.py 를 실행하세요."
            )
        data_base_dir = config.get("data_base_dir") or infer_data_base_dir(data_root)
        config["data_base_dir"] = data_base_dir
        uncertain_root = getattr(args, "uncertain_data_root", None) or uncertain_trainval_data_root(
            data_base_dir, view, data_mode
        )
        uncertain_root = os.path.normpath(str(uncertain_root))
        mapped_label = int(UNCERTAINTY_POLICY_LABEL[uncertainty_policy])
        _fold_of_subject: Dict[int, int] = {}
        for _row in _read_split_csv(os.path.join(split_dir, f"{view}_fold_assignment.csv")):
            if _row.get("subject_id") in (None, ""):
                continue
            _fold_of_subject[int(_row["subject_id"])] = int(_row["fold"])
        for _row in _read_split_csv(cohort_csv):
            _sid = int(_row["subject_id"])
            _fold = int(_row["fold"])
            if _fold_of_subject.get(_sid) != _fold:
                raise RuntimeError(
                    f"cohort CSV의 fold가 고정 split과 다릅니다: subject {_sid}, "
                    f"cohort fold={_fold}, split fold={_fold_of_subject.get(_sid)}"
                )
        unc_items, unc_labels, unc_groups, unc_by_fold = load_policy_training_additions(
            uncertain_root=uncertain_root,
            cohort_csv=cohort_csv,
            data_mode=data_mode,
            view_tag=view_tag,
            mapped_label=mapped_label,
            source_image_root=source_image_root,
            strict_grouping=strict_grouping,
        )
        outer_csv_path, _fold_csv_path = _split_csv_paths(split_dir, view)
        outer_subjects = {
            int(row["subject_id"])
            for row in _read_split_csv(outer_csv_path)
            if row.get("subject_id") not in (None, "")
        }
        leaked_subjects = sorted({
            int(str(group)[3:])
            for group in unc_groups
            if int(str(group)[3:]) in outer_subjects
        })
        if leaked_subjects:
            raise RuntimeError(
                "[LEAKAGE] outer-test 환자의 uncertainty 영상이 학습 목록에 있습니다: "
                f"{leaked_subjects[:10]}"
            )
        explicit_dicom_ids = {
            str(row.get("dicom_id") or "")
            for row in _read_split_csv(outer_csv_path) + _read_split_csv(_fold_csv_path)
        }
        cohort_dicoms = {str(row.get("dicom_id") or "") for row in _read_split_csv(cohort_csv)}
        overlap_dicoms = sorted(explicit_dicom_ids & cohort_dicoms)
        if overlap_dicoms:
            raise RuntimeError(
                "uncertainty cohort가 기존 0/1 split과 dicom_id가 겹칩니다: "
                f"{overlap_dicoms[:5]}"
            )
        all_items = list(all_items) + list(unc_items)
        all_labels = list(all_labels) + list(unc_labels)
        all_groups = list(all_groups) + list(unc_groups)
        config["uncertain_cohort_csv"] = cohort_csv
        config["uncertain_data_root"] = uncertain_root
        uncertainty_audit.update({
            "cohort_csv": cohort_csv,
            "uncertain_data_root": uncertain_root,
            "n_added_images": int(len(unc_items)),
            "n_added_subjects": int(len(set(unc_groups))),
            "added_by_patient_fold": unc_by_fold,
            "mapped_label": mapped_label,
        })
        if rank == 0:
            print(
                f"🏷️ Uncertainty training policy={uncertainty_policy}: "
                f"-1 {len(unc_items):,}장을 {mapped_label}로 변환해 학습 목록에 추가 "
                f"(explicit {n_explicit:,}장은 그대로, 검증/outer test에는 넣지 않음)"
            )
            print(f"   cohort CSV : {cohort_csv}")
            print(f"   data root  : {uncertain_root}")
            print(f"   환자 fold별 장수 (그 fold 검증 환자라 해당 fold 학습에는 안 들어감): {unc_by_fold}")
        if rank == 0:
            run_meta["uncertainty_training_policy"] = uncertainty_audit
            try:
                with open(os.path.join(save_dir, "run_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(run_meta, f, indent=2)
            except Exception as meta_err:
                print(f"⚠️ run_meta.json 갱신 실패: {meta_err}")

    if bool(config.get('preload_images', True)):
        preloaded_images = preload_all_images_single(
            all_items, int(config['input_size']), rank=rank,
            npy_cache_dir=config.get('npy_cache_dir') or None,
            num_workers=int(config.get('preload_workers', 16)),
        )
    else:
        preloaded_images = None
        if rank == 0:
            print("ℹ️ Image preloading disabled (--no-preload-images). Images will be read from disk every epoch.")

    def _assert_disjoint_groups(name_a: str, idx_a, name_b: str, idx_b):
        ga = set(all_groups[int(i)] for i in idx_a)
        gb = set(all_groups[int(i)] for i in idx_b)
        inter = ga.intersection(gb)
        assert len(inter) == 0, (
            f"[LEAKAGE] Group overlap detected: {name_a} ∩ {name_b} = {len(inter)}\n"
            f"  Examples: {list(sorted(inter))[:10]}"
        )

    def _build_dataset(indices, train_mode: bool, split_name: Optional[str] = None):
        split_tag_name = str(split_name) if split_name is not None else ("train" if train_mode else "val")
        base_kwargs = {
            'norm_mean': config['norm_mean'],
            'norm_std': config['norm_std'],
            'split_name': split_tag_name,
            'preloaded_images': preloaded_images,
        }
        if train_mode:
            base_kwargs.update({
                'flip_prob': config['aug_flip_prob'],
                'rot_prob': config['aug_rot_prob'],
                'rot_deg': config['aug_rot_deg'],
                'bc_prob': config['aug_bc_prob'],
                'brightness': config['aug_brightness'],
                'contrast': config['aug_contrast'],
                'shift_prob': config['aug_shift_prob'],
                'shift_limit': config['aug_shift_limit'],
                'scale_prob': config['aug_scale_prob'],
                'scale_limit': config['aug_scale_limit'],
            })
        return DatasetCls(all_items, indices, config['input_size'], train=train_mode, **base_kwargs)

    def _log_drop_risk(indices, split_name):
        idxs = [int(i) for i in indices]
        total = len(idxs)
        if total == 0:
            return
        fail_all = 0
        fail_pos = 0
        fail_neg = 0
        pos_all = 0
        neg_all = 0
        for i in idxs:
            img_path = all_items[i][0]
            y = all_items[i][1]
            y_int = int(y)
            if y_int == 1:
                pos_all += 1
            else:
                neg_all += 1
            bad = not os.path.exists(img_path)
            if bad:
                fail_all += 1
                if y_int == 1:
                    fail_pos += 1
                else:
                    fail_neg += 1
        print(
            f"🧾 Drop-risk[{split_name}] total={total}, fail={fail_all} ({fail_all/max(1,total):.2%}), "
            f"pos_fail={fail_pos}/{max(1,pos_all)} ({fail_pos/max(1,pos_all):.2%}), "
            f"neg_fail={fail_neg}/{max(1,neg_all)} ({fail_neg/max(1,neg_all):.2%})"
        )

    def _build_model_instance():
        return ModelCls(
            model_name=config['model_name'],
            input_size=config['input_size'],
            dropout=config['dropout'],
            eva_x_ckpt_dir=config.get('eva_x_ckpt_dir'),
        ).to(device)

    def _raw_model_for_state_io(model_obj):
        raw_model = model_obj.module if hasattr(model_obj, 'module') else model_obj
        return getattr(raw_model, '_orig_mod', raw_model)

    def _load_state_into_existing_model(model_obj, state_dict):
        _load_state_dict_compat(_raw_model_for_state_io(model_obj), state_dict)
        return model_obj

    def _load_checkpoint_into_existing_model(model_obj, checkpoint_path: str):
        """Best checkpoint를 새 모델 생성 없이 기존 모델에 다시 로드합니다."""
        # CPU로 먼저 읽으면 GPU에 model 크기만큼 state_dict가 한 벌 더 생기지 않습니다.
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        _load_state_into_existing_model(model_obj, state_dict)
        del state_dict
        return model_obj

    def _shutdown_dataloader_workers(loader) -> None:
        """persistent DataLoader worker를 fold 경계에서 명시적으로 종료합니다."""
        if loader is None:
            return
        iterator = getattr(loader, '_iterator', None)
        if iterator is None:
            return
        try:
            iterator._shutdown_workers()
        except Exception as shutdown_err:
            if rank == 0:
                print(f"⚠️ DataLoader worker 종료 경고: {shutdown_err}")
        finally:
            try:
                loader._iterator = None
            except Exception:
                pass

    def _run_xai_suite(model_eval, sampled_examples, fold_dir, split_label):
        return

    def _load_saved_best_threshold(fold_dir: str, default: float = 0.5) -> float:
        thr_path = os.path.join(fold_dir, 'best_threshold.json')
        try:
            with open(thr_path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            return float(obj.get('threshold', default))
        except Exception:
            return float(default)

    # W&B
    _dp_default = False  # 기본값: DataParallel 비활성화 (단일 GPU 사용). 멀티 GPU 시 --force-dp로 활성화
    if args.force_dp:
        _dp_enabled = True
    elif args.no_dp:
        _dp_enabled = False
    else:
        _dp_enabled = _dp_default

    use_wandb = bool(getattr(args, 'wandb', False)) and _WANDB_OK
    if use_wandb:
        try:
            api_key = args.wandb_api_key or os.environ.get('WANDB_API_KEY')
            if api_key:
                login_kwargs = {'key': api_key}
                if args.wandb_host: login_kwargs['host'] = args.wandb_host
                if args.wandb_relogin: login_kwargs['relogin'] = True
                wandb.login(**login_kwargs)
        except Exception as e:
            print(f"[wandb] 로그인 실패: {e}")
            use_wandb = False
    if use_wandb:
        _cfg = dict(config)
        _cfg.update({'data_root': data_root, 'dp_enabled': _dp_enabled, 'arch': args.arch})
        mode_tag = ""
        run_name = f"{_cfg.get('model_name','model')}_{_cfg.get('input_size','NA')}px_{split_tag}{mode_tag}_{timestamp}"
        try:
            # Save W&B run files under this run's save_dir (results/timestamp_model_split)
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=_cfg, name=run_name, dir=save_dir)
        except Exception as e:
            print(f"[wandb] init 실패: {e}")
            use_wandb = False

    # Common vars
    global_step = 0
    use_outer_test = False

    # -------------
    # CV5 pathway
    # -------------
    use_outer_test = (split_mode == 'cv5' and float(args.outer_test_ratio) > 0.0)
    split_source = "generated"
    if split_dir:
        split_assignment = load_assignment_from_split_csvs(
            split_dir,
            view,
            n_folds=int(config['n_folds']),
        )
        split_source = "fixed_csv"
        if rank == 0:
            print(
                f"📂 Fixed patient split loaded from {split_dir} "
                f"(split_seed={split_seed}, training_seed={config['seed']})"
            )
    else:
        split_assignment = build_grouped_split_assignment(
            split_items,
            split_labels,
            split_groups,
            n_folds=int(config['n_folds']),
            outer_test_ratio=float(args.outer_test_ratio) if use_outer_test else 0.0,
            seed=split_seed,
        )
    outer_trainval_idx, outer_test_idx, active_fold_indices = map_grouped_split_assignment(
        all_groups,
        split_assignment,
    )

    if uncertainty_policy != "explicit_only":
        leaked_test = [int(i) for i in outer_test_idx if int(i) >= n_explicit]
        if leaked_test:
            raise RuntimeError(
                "[LEAKAGE] outer test에 uncertainty 영상이 들어 있습니다. "
                "평가 endpoint는 explicit 0/1만 사용해야 합니다."
            )
        filtered_folds = []
        per_fold_plan = []
        for fold_i, (train_idx, val_idx) in enumerate(active_fold_indices, start=1):
            val_groups = {all_groups[int(i)] for i in val_idx if int(i) < n_explicit}
            removed = [int(i) for i in val_idx if int(i) >= n_explicit]
            stray = [i for i in removed if all_groups[i] not in val_groups]
            if stray:
                raise RuntimeError(
                    f"[LEAKAGE] fold {fold_i}: 다른 fold 환자의 uncertainty 영상이 validation에 있습니다."
                )
            bad_train = [
                int(i) for i in train_idx
                if int(i) >= n_explicit and all_groups[int(i)] in val_groups
            ]
            if bad_train:
                raise RuntimeError(
                    f"[LEAKAGE] fold {fold_i}: validation 환자의 uncertainty 영상이 학습 목록에 있습니다."
                )
            val_explicit = np.asarray(
                [int(i) for i in val_idx if int(i) < n_explicit],
                dtype=np.int64,
            )
            filtered_folds.append((np.asarray(train_idx, dtype=np.int64), val_explicit))
            per_fold_plan.append({
                "fold": int(fold_i),
                "n_uncertain_in_train": int(sum(int(i) >= n_explicit for i in train_idx)),
                "n_uncertain_removed_from_val": int(len(removed)),
                "n_explicit_train": int(sum(int(i) < n_explicit for i in train_idx)),
                "n_explicit_val": int(len(val_explicit)),
            })
        active_fold_indices = filtered_folds
        uncertainty_audit["per_fold"] = per_fold_plan
        if rank == 0:
            added_msg = ", ".join(
                f"fold{row['fold']}={row['n_uncertain_in_train']}" for row in per_fold_plan
            )
            print(f"🏷️ fold별 학습에 추가되는 uncertainty 영상: {added_msg}")
            print("🏷️ validation과 outer test는 explicit 0/1만 남겼습니다.")

    active_group_set = set(all_groups)
    reference_group_set = (
        set(split_assignment["outer_trainval_groups"])
        | set(split_assignment["outer_test_groups"])
    )
    split_audit = {
        "split_seed": int(split_seed),
        "training_seed": int(config['seed']),
        "split_source": split_source,
        "split_dir": split_dir,
        "n_folds": int(config['n_folds']),
        "outer_test_ratio_requested": float(args.outer_test_ratio) if use_outer_test else 0.0,
        "outer_split_method": split_assignment["outer_method"],
        "reference": {
            "data_root": split_reference_data_root or data_root,
            "data_mode": config['split_reference_data_mode'] if split_reference_data_root else data_mode,
            "n_images": split_assignment["reference_n_images"],
            "n_subjects": split_assignment["reference_n_subjects"],
        },
        "active": {
            "data_root": data_root,
            "data_mode": data_mode,
            "n_images": int(len(all_items)),
            "n_subjects": int(len(active_group_set)),
            "reference_subjects_missing_from_active": sorted(reference_group_set - active_group_set),
            "outer_trainval_n_images": int(len(outer_trainval_idx)),
            "outer_trainval_n_subjects": int(len({all_groups[int(i)] for i in outer_trainval_idx})),
            "outer_test_n_images": int(len(outer_test_idx)),
            "outer_test_n_subjects": int(len({all_groups[int(i)] for i in outer_test_idx})),
        },
        "assignment": split_assignment,
        "uncertainty_training_policy": uncertainty_audit,
    }
    if rank == 0:
        with open(os.path.join(save_dir, 'split_assignment.json'), 'w', encoding='utf-8') as f:
            json.dump(split_audit, f, indent=2)

    if use_outer_test:
        _outer_ratio = float(args.outer_test_ratio)
        if rank == 0:
            print(f"\n🔒 Outer holdout Test split enabled: ratio={_outer_ratio:.2f}")
        _actual_test_ratio = len(outer_test_idx) / max(1, len(all_items))
        _tv_pos = int(sum(all_labels[i] for i in outer_trainval_idx))
        _tv_neg = int(len(outer_trainval_idx) - _tv_pos)
        _te_pos = int(sum(all_labels[i] for i in outer_test_idx))
        _te_neg = int(len(outer_test_idx) - _te_pos)
        if rank == 0:
            print(f"📦 Outer split method: {split_assignment['outer_method']}")
            print(f"📦 Outer splits - TrainVal({(1-_actual_test_ratio)*100:.1f}%): {len(outer_trainval_idx)} "
                  f"(pos={_tv_pos}, neg={_tv_neg}, prev={_tv_pos/max(1,len(outer_trainval_idx)):.3f})")
            print(f"📦 Outer splits - HoldoutTest({_actual_test_ratio*100:.1f}%): {len(outer_test_idx)} "
                  f"(pos={_te_pos}, neg={_te_neg}, prev={_te_pos/max(1,len(outer_test_idx)):.3f})")
        _assert_disjoint_groups("outer_trainval", outer_trainval_idx, "outer_test", outer_test_idx)
        if rank == 0:
            print("✅ Leakage guard: outer trainval/test group sets are disjoint")
    fold_results = []
    oof_labels_all, oof_probs_all, oof_dirs_all = [], [], []
    # A2 fix: per-fold threshold 를 기록해 두고, pooled OOF 에 각 fold 의
    # 자기 자신 threshold 를 적용 → pooled 데이터에서 threshold 를 재-튜닝하지 않음.
    # B2: patient-level 집계를 위해 subject_id(group) 와 fold 인덱스도 수집.
    oof_subject_ids_all: List[Any] = []
    oof_folds_all: List[int] = []
    oof_thresholds_per_fold: List[float] = []
    fold_model_paths = []
    split_iter = active_fold_indices
    requested_max_folds = config.get('max_folds', None)
    total_planned_folds = int(config['n_folds'])
    # 단일 GPU에서는 fold마다 native 백본을 파괴·재생성하지 않습니다.
    # HPC-X/UCX가 로드된 환경에서 두 번째 timm/EVA-X 모델 생성 시
    # signal 11이 발생할 수 있어, 최초 모델과 초기 state를 계속 재사용합니다.
    _reuse_model_across_folds = (
        not use_ddp
        and not _dp_enabled
        and not bool(config.get('use_compile', False))
    )
    _fold_model_cache = None
    _fold_initial_state = None
    if rank == 0 and requested_max_folds is not None:
        print(f"⚡ Quick CV mode: running first {requested_max_folds} / {total_planned_folds} folds")
    if rank == 0 and _reuse_model_across_folds:
        print("♻️ Single-GPU fold model reuse enabled (native model은 한 번만 생성)")

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        fpr_val = np.array([])
        tpr_val = np.array([])
        if requested_max_folds is not None and fold_idx >= int(requested_max_folds):
            if rank == 0:
                print(f"⏹️ Reached --max-folds={requested_max_folds}; stopping after {len(fold_results)} completed fold(s).")
            break
        # fold 시작 전 모든 rank 동기화: post-fold I/O(체크포인트·그래프 저장) 시간 차이로
        # 한 rank가 먼저 다음 fold DDP init(ALLGATHER)에 진입하는 것을 방지.
        if use_ddp:
            _ddp_barrier(local_rank)
        if rank == 0:
            print(f"\n{'='*60}\n📂 Fold {fold_idx+1}/{config['n_folds']}\n{'='*60}")
        _assert_disjoint_groups("fold_train", train_idx, "fold_val", val_idx)
        fold_dir = os.path.join(save_dir, f'fold_{fold_idx+1}'); os.makedirs(fold_dir, exist_ok=True)
        model_path = os.path.join(fold_dir, 'best_model.pth')

        _fold_done_path = os.path.join(fold_dir, 'fold_results.json')
        if os.path.isfile(_fold_done_path):
            if rank == 0:
                print(f"⏭️  Fold {fold_idx+1} 이미 완료 — fold_results.json 발견, 학습 건너뜀")
            _loaded_ok = False
            try:
                with open(_fold_done_path, 'r', encoding='utf-8') as _f_done:
                    _prev_fold = json.load(_f_done)
                fold_results.append(_prev_fold)
                if use_outer_test and os.path.isfile(model_path):
                    fold_model_paths.append(model_path)
                _loaded_ok = True
            except Exception as _load_done_err:
                if rank == 0:
                    print(f"⚠️ 완료 fold 로드 실패 ({_fold_done_path}): {_load_done_err} — 처음부터 다시 학습합니다.")
            if _loaded_ok:
                continue

        train_indices_orig = train_idx.tolist(); train_indices = train_indices_orig
        if config.get('downsample_negative', False):
            _labels_before = [all_labels[i] for i in train_indices_orig]
            _pos_before = int(sum(_labels_before)); _neg_before = int(len(_labels_before) - _pos_before)
            neg_per_pos = float(config.get('downsample_neg_per_pos', -1.0))
            if neg_per_pos > 0.0:
                _ds_seed = config['seed'] + fold_idx
                train_indices = downsample_negative_to_ratio(train_indices, all_labels, neg_per_pos=neg_per_pos, seed=_ds_seed)
                print(f"🔧 Downsample negative applied: neg_per_pos={neg_per_pos:.4f}, seed={_ds_seed} | train {len(train_indices_orig)} -> {len(train_indices)}")

        _labels_after = [all_labels[i] for i in train_indices]
        _pos_after = int(sum(_labels_after)); _neg_after = int(len(_labels_after) - _pos_after)
        print(f"📊 Train(after ds) - Neg: {_neg_after}, Pos: {_pos_after}, Ratio Neg:Pos = {(_neg_after / max(1, _pos_after)):.2f}:1")

        # Calculate imbalance ratio AFTER downsampling for loss function
        train_labels_fold = [all_labels[i] for i in train_indices]
        pos_count = sum(train_labels_fold); neg_count = len(train_labels_fold) - pos_count
        imbalance_ratio = neg_count / max(1, pos_count)

        if getattr(args, 'cv5_inner_811', False):
            inner_groups = [all_groups[i] for i in train_indices]
            gss_inner = GroupShuffleSplit(n_splits=1, train_size=1.0 - (1.0/9.0), random_state=config['seed'])
            tr_rel_in, va_rel_in = next(gss_inner.split(train_indices, [all_labels[i] for i in train_indices], groups=inner_groups))
            train_inner_idx = [train_indices[i] for i in tr_rel_in]
            val_inner_idx = [train_indices[i] for i in va_rel_in]
            if config.get('downsample_negative', False):
                neg_per_pos = float(config.get('downsample_neg_per_pos', -1.0))
                if neg_per_pos > 0.0:
                    _ds_seed_inner = config['seed'] + fold_idx + 1000
                    train_inner_idx = downsample_negative_to_ratio(train_inner_idx, all_labels, neg_per_pos=neg_per_pos, seed=_ds_seed_inner)
                    print(f"🔧 Downsample negative (cv5 inner, neg_per_pos={neg_per_pos:.4f}, seed={_ds_seed_inner}) | train -> {len(train_inner_idx)}")

            # 내부 검증도 explicit 0/1만 쓴다. 학습 환자 uncertainty는 학습에 남긴다.
            if uncertainty_policy != "explicit_only":
                held_out_uncertain = [i for i in val_inner_idx if int(i) >= n_explicit]
                if held_out_uncertain:
                    val_inner_idx = [i for i in val_inner_idx if int(i) < n_explicit]
                    train_inner_idx = list(train_inner_idx) + held_out_uncertain

            # Recalculate imbalance_ratio for inner train split
            train_inner_labels = [all_labels[i] for i in train_inner_idx]
            pos_count = sum(train_inner_labels); neg_count = len(train_inner_labels) - pos_count
            imbalance_ratio = neg_count / max(1, pos_count)
            print(f"📊 CV5 inner train - Neg: {neg_count}, Pos: {pos_count}, Imbalance ratio: {imbalance_ratio:.2f}:1")

            _log_drop_risk(train_inner_idx, f"fold{fold_idx+1}_train_inner")
            _log_drop_risk(val_inner_idx, f"fold{fold_idx+1}_val_inner")
            _log_drop_risk(val_idx.tolist(), f"fold{fold_idx+1}_test_outer")

            train_dataset = _build_dataset(train_inner_idx, train_mode=True, split_name="train_inner")
            val_dataset = _build_dataset(val_inner_idx, train_mode=False, split_name="val_inner")
            test_dataset = _build_dataset(val_idx.tolist(), train_mode=False, split_name="test_outer")
            if use_ddp:
                train_sampler_ddp = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler_ddp, drop_last=True, collate_fn=collate_skip_none, **{k: v for k, v in loader_kwargs.items() if k not in ('shuffle', 'batch_size')})
            else:
                train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_skip_none, **loader_kwargs)
            val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_skip_none, **eval_loader_kwargs)
            test_loader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_skip_none, **eval_loader_kwargs)
            if rank == 0:
                print(f"📊 CV5 inner split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test(outer): {len(test_dataset)}")
        else:
            print(f"📊 Class distribution (after downsampling) - Neg: {neg_count}, Pos: {pos_count}, Imbalance ratio: {imbalance_ratio:.2f}:1")

            _log_drop_risk(train_indices, f"fold{fold_idx+1}_train")
            _log_drop_risk(val_idx.tolist(), f"fold{fold_idx+1}_val")
            train_dataset = _build_dataset(train_indices, train_mode=True, split_name="train")
            val_dataset = _build_dataset(val_idx.tolist(), train_mode=False, split_name="val")
            if use_ddp:
                train_sampler_ddp = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler_ddp, drop_last=True, collate_fn=collate_skip_none, **{k: v for k, v in loader_kwargs.items() if k not in ('shuffle', 'batch_size')})
            else:
                train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_skip_none, **loader_kwargs)
            val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_skip_none, **eval_loader_kwargs)

        if uncertainty_policy != "explicit_only":
            train_for_weight = train_inner_idx if getattr(args, 'cv5_inner_811', False) else train_indices
            n_unc_loss = int(sum(int(i) >= n_explicit for i in train_for_weight))
            if any(int(i) >= n_explicit for i in val_idx.tolist()):
                raise RuntimeError(f"[LEAKAGE] fold {fold_idx+1} validation에 uncertainty 영상이 남아 있습니다.")
            if rank == 0:
                print(
                    f"🏷️ Fold {fold_idx+1} {uncertainty_policy}: "
                    f"학습 uncertainty {n_unc_loss}장, pos_weight={imbalance_ratio:.4f} "
                    f"(neg={int(neg_count)}, pos={int(pos_count)}, 이 fold 학습 라벨로 재계산)"
                )
                for row in uncertainty_audit.get("per_fold", []):
                    if int(row.get("fold", -1)) == fold_idx + 1:
                        row["train_pos"] = int(pos_count)
                        row["train_neg"] = int(neg_count)
                        row["pos_weight"] = float(imbalance_ratio)
                        row["n_uncertain_used_in_loss"] = n_unc_loss
                        break
                try:
                    with open(os.path.join(save_dir, "uncertainty_training_audit.json"), "w", encoding="utf-8") as f:
                        json.dump(uncertainty_audit, f, indent=2)
                except Exception as audit_err:
                    print(f"⚠️ uncertainty audit 저장 실패: {audit_err}")

        if rank == 0:
            print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        if _reuse_model_across_folds and _fold_model_cache is not None:
            model = _fold_model_cache
            _load_state_into_existing_model(model, _fold_initial_state)
            if rank == 0:
                print(f"♻️ Fold {fold_idx+1}: 기존 모델을 초기 pretrained state로 복원")
        else:
            model = _build_model_instance()
            if _reuse_model_across_folds:
                _fold_model_cache = model
                _fold_initial_state = {
                    key: value.detach().cpu().clone()
                    for key, value in _raw_model_for_state_io(model).state_dict().items()
                }
        # Backbone freeze (linear probe) — default ON
        if config.get('freeze_backbone', False):
            _raw_model = model.module if hasattr(model, 'module') else model
            _raw_model.freeze_backbone()
            if fold_idx == 0:  # print once (first fold only)
                _total = sum(p.numel() for p in model.parameters())
                _trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"🧊 Backbone FROZEN (linear probe): trainable {_trainable:,} / total {_total:,} params ({_trainable/_total*100:.1f}%)")
        else:
            if fold_idx == 0:
                _total = sum(p.numel() for p in model.parameters())
                print(f"🔥 Backbone UNFROZEN (full fine-tuning): all {_total:,} params trainable")
        if bool(config.get('use_compile', False)) and device.type == 'cuda':
            try:
                _prepare_torch_compile_dynamo()
                model = torch.compile(model)
                if rank == 0:
                    print(f"⚡ torch.compile(model) enabled")
            except Exception as e:
                if rank == 0:
                    print(f"⚠️ torch.compile failed ({e}), continuing without compile")
        if use_ddp:
            # ConvNeXtV2 depthwise conv(7x7, groups=C) 의 backward gradient는
            # size=1 차원 때문에 is_contiguous()==True 이면서 DDP bucket과 stride가
            # 다를 수 있습니다. 파라미터 data를 contiguous로 바꿔도 gradient는
            # 매 backward마다 다시 생기므로, DDP 래핑 전에 hook을 걸어야 합니다.
            _n_made_contiguous = 0
            for _p in model.parameters():
                if not _p.data.is_contiguous():
                    _p.data = _p.data.contiguous()
                    _n_made_contiguous += 1
            _n_stride_hooks = _register_ddp_grad_stride_hooks(model)
            if rank == 0 and (_n_made_contiguous > 0 or _n_stride_hooks > 0):
                print(
                    f"🔧 DDP prep: {_n_made_contiguous} non-contiguous param(s) → contiguous, "
                    f"{_n_stride_hooks} depthwise-conv grad stride hook(s)"
                )

            # 현재 모델 forward는 backbone feature와 head를 모두 사용합니다.
            # frozen backbone은 requires_grad=False라 DDP unused-parameter 탐색이 필요 없습니다.
            _find_unused = False
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                find_unused_parameters=_find_unused,
                # stride hook이 bucket과 같은 레이아웃을 만들어 주면
                # bucket view 재사용이 가능해 복사 비용을 줄입니다.
                gradient_as_bucket_view=True,
            )
            if rank == 0:
                print(f"🧩 Using DistributedDataParallel (world_size={world_size}, find_unused_parameters={_find_unused}, gradient_as_bucket_view=True)")
        elif _dp_enabled and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            try:
                if rank == 0:
                    print(f"🧩 Using DataParallel on {torch.cuda.device_count()} GPUs")
                model = nn.DataParallel(model)
            except Exception as e:
                if rank == 0:
                    print(f"⚠️ DataParallel init failed ({e}). Fallback to single GPU.")

        # Loss function selection
        loss_type = config.get('loss_type', 'weighted_bce')
        if loss_type == 'focal':
            criterion = FocalLoss(alpha=config['focal_alpha'], gamma=config['focal_gamma'])
            print(f"🔥 Loss: Focal Loss (alpha={config['focal_alpha']}, gamma={config['focal_gamma']})")
        elif loss_type == 'weighted_ce':
            class_weights = torch.tensor([1.0, imbalance_ratio], dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config['label_smoothing'])
            print(f"⚖️ Loss: Weighted CrossEntropyLoss (weights: [1.0, {imbalance_ratio:.2f}], label_smoothing={config['label_smoothing']})")
        elif loss_type == 'bce':
            criterion = BCELossWrapper(pos_weight=None)
            print(f"🎯 Loss: Binary CrossEntropy (BCEWithLogitsLoss)")
        elif loss_type == 'weighted_bce':
            pos_weight = config.get('bce_pos_weight', None) or imbalance_ratio
            criterion = BCELossWrapper(pos_weight=pos_weight).to(device)
            print(f"⚖️ Loss: Weighted Binary CrossEntropy (pos_weight={pos_weight:.4f} = N_neg/N_pos, training fold)")
        else:  # 'ce' or default
            criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
            print(f"📊 Loss: CrossEntropyLoss (label_smoothing={config['label_smoothing']})")

        if config.get('learning_rate') is None or config.get('weight_decay') is None:
            raise ValueError(
                f"learning_rate 또는 weight_decay가 설정되지 않았습니다. "
                f"lr={config.get('learning_rate')}, wd={config.get('weight_decay')}. "
                f"--arch 프리셋을 지정하거나 --learning-rate / --weight-decay CLI 인수를 사용하세요."
            )
        _opt_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(_opt_params, lr=config['learning_rate'], weight_decay=config['weight_decay'])
        
        # Scheduler selection based on config
        scheduler_type = config.get('scheduler_type', 'plateau')
        warmup_epochs = config.get('warmup_epochs', 0)
        _cosine_tmax = max(1, config['epochs'] - warmup_epochs)
        if scheduler_type == 'cosine':
            _main_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_cosine_tmax, eta_min=1e-7)
            _sched_desc = f"CosineAnnealingLR(T_max={_cosine_tmax}, eta_min=1e-7)"
        elif scheduler_type == 'cosine_restart':
            _main_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=1e-7)
            _sched_desc = "CosineAnnealingWarmRestarts(T_0=15, T_mult=1)"
        elif scheduler_type == 'step':
            _main_sched = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
            _sched_desc = "StepLR(step_size=15, gamma=0.5)"
        else:  # 'plateau' or default
            _main_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            _sched_desc = "ReduceLROnPlateau(factor=0.5, patience=3)"
        # LinearLR warmup 적용 (plateau 제외: step()에 metric이 필요해 SequentialLR 비호환)
        if warmup_epochs > 0 and scheduler_type != 'plateau':
            _warmup_sched = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_epochs
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[_warmup_sched, _main_sched], milestones=[warmup_epochs]
            )
            print(f"📊 Scheduler: LinearLR warmup({warmup_epochs}ep, 0.001x→1x) → {_sched_desc}")
        else:
            scheduler = _main_sched
            print(f"📊 Scheduler: {_sched_desc}")
            if warmup_epochs > 0:
                print(f"   ↳ LR warmup: manual linear ({warmup_epochs}ep, 0.001x→1x)")
        
        writer = SummaryWriter(os.path.join(fold_dir, 'tensorboard')) if rank == 0 else None

        history = {
            'train_loss': [], 'train_acc': [], 'train_auc': [], 'train_auprc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_auprc': [], 'val_f1': [],
            'train_sensitivity': [], 'val_sensitivity': []
        }

        # Model selection (AP/PA 공통):
        # validation AUROC 기준으로 best checkpoint 선정
        # A) 주요 지표 큰 개선(>=0.001)
        # B) 주요 지표 소폭 개선(>=0.001) + Val Loss 개선
        _primary_metric_name = "AUROC"
        best_val_primary = -1.0
        best_val_auc = -1.0
        best_val_auprc_at_best = -1.0
        best_val_loss = float('inf')
        patience_counter = 0
        best_metrics = None; best_threshold = 0.5

        if rank == 0:
            print("\n📚 학습 시작...")
            print("="*60)
        use_fp32 = bool(config.get('use_fp32', False))
        use_bf16 = bool(config.get('use_bf16', True)) and not use_fp32
        scaler = torch.amp.GradScaler('cuda') if (device.type == 'cuda' and not use_bf16 and not use_fp32) else None
        if device.type == 'cuda' and rank == 0:
            if use_fp32:
                print("🔬 Precision: float32 (autocast 없음, 메모리↑)")
            elif use_bf16:
                print("🔬 Mixed precision: bfloat16 (Ampere+ 권장)")
            else:
                print("🔬 Mixed precision: fp16 + GradScaler (CNN 수치 안정성)")

        # Resume: 이전 체크포인트에서 학습 재개
        _start_epoch = 0
        _resume_dir = getattr(args, 'resume_dir', None) or save_dir
        if _resume_dir:
            _fold_resume_dir = os.path.join(_resume_dir, f'fold_{fold_idx+1}')
            _ckpt_file = os.path.join(_fold_resume_dir, 'last_checkpoint.pth')
            _best_file = os.path.join(_fold_resume_dir, 'best_model.pth')
            _best_meta_file = os.path.join(_fold_resume_dir, 'best_threshold.json')
            _ckpt_loaded = False
            if os.path.isfile(_ckpt_file):
                try:
                    _ckpt = torch.load(_ckpt_file, map_location=device)
                    _raw_m = model.module if hasattr(model, 'module') else model
                    _raw_m = getattr(_raw_m, '_orig_mod', _raw_m)
                    _load_state_dict_compat(_raw_m, _ckpt['model_state_dict'])
                    optimizer.load_state_dict(_ckpt['optimizer_state_dict'])
                    if _ckpt.get('scheduler_state_dict') is not None:
                        scheduler.load_state_dict(_ckpt['scheduler_state_dict'])
                    if scaler is not None and _ckpt.get('scaler_state_dict') is not None:
                        scaler.load_state_dict(_ckpt['scaler_state_dict'])
                    best_val_primary = float(_ckpt.get('best_val_primary', _ckpt.get('best_val_auprc', -1.0)))
                    best_val_auc = float(_ckpt.get('best_val_auc', best_val_primary))
                    best_val_auprc_at_best = float(_ckpt.get('best_val_auprc_at_best', -1.0))
                    best_val_loss = float(_ckpt.get('best_val_loss', float('inf')))
                    best_threshold = float(_ckpt.get('best_threshold', 0.5))
                    patience_counter = int(_ckpt.get('patience_counter', 0))
                    global_step = int(_ckpt.get('global_step', 0))
                    _start_epoch = int(_ckpt['epoch']) + 1
                    _ckpt_loaded = True
                    if rank == 0:
                        print(f"🔄 Resume: {_ckpt_file} → epoch {_start_epoch} 부터 재개 (best AUROC={best_val_primary:.4f})")
                except Exception as _ckpt_err:
                    if rank == 0:
                        print(f"⚠️ last_checkpoint.pth 로드 실패(손상 가능): {_ckpt_err}")
            if not _ckpt_loaded and os.path.isfile(_best_file):
                try:
                    _raw_m = model.module if hasattr(model, 'module') else model
                    _raw_m = getattr(_raw_m, '_orig_mod', _raw_m)
                    _load_state_dict_compat(_raw_m, torch.load(_best_file, map_location=device))
                    if os.path.isfile(_best_meta_file):
                        with open(_best_meta_file, 'r', encoding='utf-8') as _bmf:
                            _bm = json.load(_bmf)
                        best_val_primary = float(_bm.get('best_metric_value', _bm.get('auc', -1.0)))
                        best_val_auc = float(_bm.get('auc', best_val_primary))
                        best_val_auprc_at_best = float(_bm.get('auprc', -1.0))
                        best_val_loss = float(_bm.get('val_loss_at_best', float('inf')))
                        best_threshold = float(_bm.get('threshold', 0.5))
                    if rank == 0:
                        print(f"🔄 Resume fallback: {_best_file} 가중치 로드 → epoch 1부터 재학습 (best AUROC={best_val_primary:.4f})")
                        if os.path.isfile(_ckpt_file):
                            print("   (last_checkpoint.pth 는 사용하지 않음 — optimizer/scheduler 상태는 초기화)")
                except Exception as _best_err:
                    if rank == 0:
                        print(f"⚠️ best_model.pth 로드 실패: {_best_err} — 처음부터 학습합니다.")
            elif not _ckpt_loaded and rank == 0:
                print(f"⚠️ Resume 체크포인트 미발견: {_ckpt_file} — 처음부터 학습합니다.")

        for epoch in range(_start_epoch, config['epochs']):
            if use_ddp and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            global_step += 1
            if rank == 0:
                print(f"\nEpoch {epoch+1}/{config['epochs']}")
            train_loss, train_acc, train_auc, train_auprc, train_sensitivity = train_epoch_fn(
                model, train_loader, criterion, optimizer, device, scaler=scaler, use_bf16=use_bf16
            )
            # NaN 가중치 감지: gradient explosion으로 모델이 망가졌는지 확인
            _nan_params = sum(1 for p in model.parameters() if p.data.isnan().any() or p.data.isinf().any())
            if _nan_params > 0 and rank == 0:
                print(f"🚨 경고: 모델 가중치에 NaN/Inf 발견 ({_nan_params}개 파라미터) — 학습률을 낮추거나 max_grad_norm을 줄이세요!")
            val_loss, val_acc, val_auc, val_auprc, val_sensitivity, val_labels, val_preds, val_probs, val_dirs = validate_epoch_fn(
                model, val_loader, criterion, device, use_bf16=use_bf16, use_fp32=use_fp32
            )
            # DDP: 각 rank가 전체 val set을 독립적으로 inference하므로 bfloat16 비결정성으로
            # val_loss / val_auc / val_auprc가 rank 간 미세하게 달라질 수 있음.
            # rank 0의 값을 broadcast해 patience_counter와 early-stopping 결정을
            # 모든 rank에서 항상 동일하게 유지 → fold 경계에서의 NCCL 교착 방지.
            if use_ddp and dist.is_initialized():
                _sync_m = torch.tensor(
                    [train_loss, val_loss, val_auc, val_auprc, val_sensitivity],
                    dtype=torch.float32, device=device,
                )
                dist.broadcast(_sync_m, src=0)
                train_loss      = float(_sync_m[0].item())
                val_loss        = float(_sync_m[1].item())
                val_auc         = float(_sync_m[2].item())
                val_auprc       = float(_sync_m[3].item())
                val_sensitivity = float(_sync_m[4].item())
            # validation 결과가 비었을 경우 (모든 배치 NaN/Inf 스킵) 안전하게 처리
            if len(val_labels) == 0 and rank == 0:
                print(f"⚠️ Epoch {epoch+1}: validation 결과가 비어 있음 (모든 배치 NaN/Inf). 모델 가중치 폭발 의심.")
            tuned_thr = tune_threshold(val_labels, val_probs, mode=config.get('thr_mode','youden'), min_specificity=config.get('min_specificity',0.9), min_sensitivity=config.get('min_sensitivity',0.8))
            val_metrics_opt = compute_metrics_with_threshold(val_labels, val_probs, tuned_thr)
            val_f1_opt = val_metrics_opt.get('f1_score', 0.0)
            # AP/PA 공통: validation AUROC 기준으로 best 모델 선정
            val_primary = val_auc
            _primary_metric_name = "AUROC"

            history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
            history['train_auc'].append(train_auc); history['train_auprc'].append(train_auprc)
            history['train_sensitivity'].append(train_sensitivity)
            history['val_loss'].append(val_loss); history['val_acc'].append(val_metrics_opt['accuracy'])
            history['val_auc'].append(val_auc); history['val_auprc'].append(val_auprc)
            history['val_f1'].append(val_f1_opt); history['val_sensitivity'].append(val_metrics_opt['sensitivity'])

            if writer is not None:
                writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
                writer.add_scalars('AUC', {'train': train_auc, 'val': val_auc}, epoch)
                writer.add_scalars('AUPRC', {'train': train_auprc, 'val': val_auprc}, epoch)
            if use_wandb and rank == 0:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_loss, 'train/auc': train_auc, 'train/auprc': train_auprc,
                    'val/loss': val_loss, 'val/acc_opt': val_metrics_opt['accuracy'], 'val/auc': val_auc,
                    'val/auprc': val_auprc, 'val/f1_opt': val_f1_opt,
                    'val/sens_opt': val_metrics_opt['sensitivity'], 'val/spec_opt': val_metrics_opt['specificity'],
                    'val/threshold_opt': val_metrics_opt['threshold']
                }, step=global_step, commit=False)

            # Scheduler step (ReduceLROnPlateau needs metric, others don't)
            if scheduler_type == 'plateau':
                if epoch < warmup_epochs:
                    lr_scale = (epoch + 1) / max(1, warmup_epochs)
                    for _pg in optimizer.param_groups:
                        _pg['lr'] = config['learning_rate'] * lr_scale
                else:
                    scheduler.step(val_loss)
            else:
                scheduler.step()  # SequentialLR이 warmup/main 전환을 자동 처리
            if rank == 0:
                print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}, AUPRC: {train_auprc:.4f}")
                print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_metrics_opt['accuracy']:.4f}, AUC: {val_auc:.4f}, AUPRC: {val_auprc:.4f}, Sens: {val_metrics_opt['sensitivity']:.4f}, Thr: {tuned_thr:.3f}")

            # --- Trend-based overfitting detection ---
            trend_overfit_reasons_cv = []
            n_hist_cv = len(history['val_loss'])
            if n_hist_cv >= 4:
                vl_cv = history['val_loss']
                tl_cv = history['train_loss']
                # Check 1: val_loss 연속 3회 이상 상승
                consec_rise_cv = 0
                for i in range(n_hist_cv - 1, max(n_hist_cv - 4, 0), -1):
                    if vl_cv[i] > vl_cv[i - 1]:
                        consec_rise_cv += 1
                    else:
                        break
                if consec_rise_cv >= 3:
                    trend_overfit_reasons_cv.append(f"val_loss {consec_rise_cv}회 연속 상승")
                # Check 2: train_loss↓ 인데 val_loss 정체/상승 3회 연속
                diverge_cv = 0
                for i in range(n_hist_cv - 1, max(n_hist_cv - 4, 0), -1):
                    if tl_cv[i] < tl_cv[i - 1] and vl_cv[i] >= vl_cv[i - 1] - 1e-4:
                        diverge_cv += 1
                    else:
                        break
                if diverge_cv >= 3:
                    trend_overfit_reasons_cv.append(f"train↓ val→ 괴리 {diverge_cv}회 연속")
            trend_overfitted_cv = len(trend_overfit_reasons_cv) > 0
            if trend_overfitted_cv and rank == 0:
                print(f"📈 과적합 추세 감지: {', '.join(trend_overfit_reasons_cv)}")

            did_best_commit = False
            # Best model: A/B hybrid rule, with overfitting guard
            # Strategy 1: Loss ratio — if val_loss is >10x train_loss, model is memorizing
            # NOTE: AUPRC gap check removed — train is downsampled but val is not,
            #       so train AUPRC is structurally higher due to different class prevalence,
            #       not overfitting. Comparing them directly is misleading.
            # Strategy 2(A): AUROC must improve by at least 0.001
            # Strategy 2(B): AUROC must improve by at least 0.001 AND val_loss must not increase
            # Strategy 3: Trend-based — val_loss 연속 상승 or train↓ val→ 괴리
            loss_ratio_cv = val_loss / (train_loss + 1e-8)
            overfitted_cv = (loss_ratio_cv > 10.0) or trend_overfitted_cv
            min_delta = 0.001
            soft_delta = 0.001
            min_loss_drop_ratio = 0.0  # val loss 감소하기만 하면 됨 (이전 best 대비 증가만 아니면 됨)
            max_loss_increase_ratio = 0.50
            primary_gain_cv = val_primary - best_val_primary
            if np.isfinite(best_val_loss):
                val_loss_drop_ratio_cv = (best_val_loss - val_loss) / (best_val_loss + 1e-8)
            else:
                val_loss_drop_ratio_cv = 0.0
            val_usable = len(val_labels) > 0 and np.isfinite(val_primary)
            first_best_cv = (best_val_primary < 0.0) and val_usable
            loss_exploded_cv = (not first_best_cv) and (val_loss_drop_ratio_cv < -max_loss_increase_ratio)
            hard_improve_cv = val_usable and (primary_gain_cv >= min_delta) and (not loss_exploded_cv)
            soft_improve_cv = val_usable and (primary_gain_cv >= soft_delta) and (val_loss_drop_ratio_cv >= min_loss_drop_ratio)
            should_update_best_cv = val_usable and not (_nan_params > 0) and (first_best_cv or hard_improve_cv or soft_improve_cv)

            if (first_best_cv or hard_improve_cv or soft_improve_cv) and not should_update_best_cv and rank == 0:
                if _nan_params > 0:
                    print(f"⚠️ Best checkpoint 저장 건너뜀: 모델 NaN/Inf ({_nan_params}개 파라미터)")
                elif not val_usable:
                    print("⚠️ Best checkpoint 저장 건너뜀: validation 결과 없음 (NaN/Inf 배치)")

            if loss_exploded_cv and (primary_gain_cv >= min_delta) and rank == 0:
                print(f"⚠️ Val loss exploded ({val_loss_drop_ratio_cv*100:+.1f}% > -{max_loss_increase_ratio*100:.0f}% limit) despite {_primary_metric_name} gain={primary_gain_cv:.4f}. Skipping save.")

            if should_update_best_cv:
                if overfitted_cv:
                    if rank == 0:
                        reasons = []
                        if loss_ratio_cv > 10.0:
                            reasons.append(f"Loss ratio {loss_ratio_cv:.1f}x > 10x")
                        reasons.extend(trend_overfit_reasons_cv)
                        print(f"⚠️ Overfitting detected ({', '.join(reasons)}). Skipping save.")
                else:
                    best_val_primary = val_primary
                    best_val_auc = val_auc
                    best_val_auprc_at_best = val_auprc
                    best_val_loss = val_loss
                    best_threshold = tuned_thr
                    best_metrics = val_metrics_opt.copy()
                    patience_counter = 0
                    did_best_commit = True
                    if rank == 0:
                        try:
                            torch.save(_get_state_dict_for_save(model), model_path)
                        except Exception as _save_err:
                            print(f"🚨 CRITICAL: Best model 저장 실패 ({model_path}): {_save_err}")
                            print("   → 디스크 공간·권한을 확인하세요. 이 fold의 결과가 손실될 수 있습니다.")
                        # B3 수정: fold validation 에서 4가지 주요 운영점을 모두 계산해 저장.
                        #   - 논문/리뷰어가 어떤 지점을 선호하든(F1, Youden, Spec=0.80, Spec=0.90)
                        #     추가 학습 없이 재보고 가능하게 함.
                        _val_ops: Dict[str, Any] = {}
                        try:
                            _op_thr_f1 = tune_threshold(val_labels, val_probs, mode='f1')
                            _op_thr_yd = tune_threshold(val_labels, val_probs, mode='youden')
                            _op_thr_s80 = find_threshold_for_spec(np.array(val_labels, dtype=int), np.array(val_probs, dtype=float), target_spec=0.80)
                            _op_thr_s90 = find_threshold_for_spec(np.array(val_labels, dtype=int), np.array(val_probs, dtype=float), target_spec=0.90)
                            for _name, _thr in (('f1', _op_thr_f1), ('youden', _op_thr_yd), ('spec80', _op_thr_s80), ('spec90', _op_thr_s90)):
                                try:
                                    _m = compute_metrics_with_threshold(val_labels, val_probs, _thr)
                                    _val_ops[_name] = {
                                        'threshold': float(_thr),
                                        'sensitivity': float(_m['sensitivity']),
                                        'specificity': float(_m['specificity']),
                                        'f1_score': float(_m['f1_score']),
                                        'mcc': float(_m['mcc']),
                                    }
                                except Exception:
                                    pass
                        except Exception:
                            _val_ops = {}
                        with open(os.path.join(fold_dir, 'best_threshold.json'), 'w') as f:
                            json.dump(
                                {
                                    'threshold': best_threshold,
                                    'mode': config.get('thr_mode'),
                                    'min_specificity': config.get('min_specificity'),
                                    'min_sensitivity': config.get('min_sensitivity'),
                                    'best_metric': _primary_metric_name,
                                    'best_metric_value': float(best_val_primary),
                                    'auc': float(val_auc),
                                    'auprc': float(val_auprc),
                                    'val_loss_at_best': float(val_loss),
                                    'train_auprc_at_best': float(train_auprc),
                                    'operating_points': _val_ops,
                                    'note': (
                                        "operating_points 에는 f1/youden/spec80/spec90 threshold 와 "
                                        "그 지점의 Sens/Spec/F1/MCC 가 모두 포함되어 있어, "
                                        "논문 표 작성 시 어떤 운영점을 선택해도 재학습 없이 재보고 가능."
                                    ),
                                },
                                f,
                                indent=2,
                            )
                        if first_best_cv:
                            save_rule_cv = "init"
                        elif hard_improve_cv:
                            save_rule_cv = f"A({_primary_metric_name}>=+0.001)"
                        else:
                            save_rule_cv = f"B({_primary_metric_name}>=+0.001 & val_loss↓)"
                        _best_log_metrics = f"AUROC: {best_val_primary:.4f}, AUPRC: {val_auprc:.4f}"
                        print(f"✅ Best model saved ({_best_log_metrics}, Loss: {val_loss:.4f}, Sens: {val_metrics_opt['sensitivity']:.4f}, Spec: {val_metrics_opt['specificity']:.4f}, Thr: {best_threshold:.3f})")
                        print(f"   Rule: {save_rule_cv}, {_primary_metric_name} gain={primary_gain_cv:.4f}, Val loss drop={val_loss_drop_ratio_cv*100:.1f}%")
                        y_pred_best = (np.array(val_probs) >= best_threshold).astype(int)
                        save_confusion_matrix(val_labels, y_pred_best, os.path.join(fold_dir, 'confusion_matrix.png'))
                        save_roc_curve(val_labels, val_probs, os.path.join(fold_dir, 'roc_curve.png'))
                        try:
                            save_example_predictions_grid(
                                val_labels, val_probs, val_dirs, best_threshold,
                                os.path.join(fold_dir, 'example_predictions_val.png'),
                                data_root, per_row=4,
                            )
                        except Exception as e:
                            print(f"⚠️ Val 예측 예시 그리드 생성 실패: {e}")
                        try:
                            fpr_val, tpr_val, _ = roc_curve(val_labels, val_probs)
                        except Exception:
                            fpr_val, tpr_val = np.array([]), np.array([])
                        save_pr_curve(val_labels, val_probs, os.path.join(fold_dir, 'pr_curve.png'))
                        prev = calc_prevalence(val_labels)
                        lift = (val_auprc / prev) if prev > 0 else float('nan')
                        brier = calc_brier(val_labels, val_probs)
                        ece = calc_ece(val_labels, val_probs, n_bins=10)
                        ci, cs = calc_calibration_intercept_slope(val_labels, val_probs)
                        thr_arr, dca_rows, nb_all, nb_none = decision_curve(val_labels, val_probs)
                        cal_path = os.path.join(fold_dir, 'calibration.png')
                        dca_path = os.path.join(fold_dir, 'dca.png')
                        try:
                            save_calibration_plot(val_labels, val_probs, cal_path, n_bins=10)
                            save_dca_plot(thr_arr, dca_rows, nb_all, nb_none, dca_path)
                        except Exception:
                            pass
                        if use_wandb:
                            try:
                                wandb.log({
                                    'images/confusion_matrix': wandb.Image(os.path.join(fold_dir, 'confusion_matrix.png')),
                                    'images/roc_curve': wandb.Image(os.path.join(fold_dir, 'roc_curve.png')),
                                    'images/pr_curve': wandb.Image(os.path.join(fold_dir, 'pr_curve.png')),
                                    'images/calibration': wandb.Image(cal_path) if os.path.exists(cal_path) else None,
                                    'images/dca': wandb.Image(dca_path) if os.path.exists(dca_path) else None,
                                }, step=global_step, commit=False)
                            except Exception:
                                pass
                            wandb.log({'epoch': epoch + 1, 'best/val_primary': best_val_primary, 'best/val_auc': best_val_auc, 'best/val_auprc': best_val_auprc_at_best,
                                        'best/prevalence': prev, 'best/lift': lift,
                                        'calibration/brier': brier, 'calibration/ece': ece,
                                        'calibration/intercept': ci, 'calibration/slope': cs}, step=global_step)

            if use_wandb and rank == 0 and not did_best_commit:
                try:
                    wandb.log({'epoch': epoch + 1}, step=global_step)
                except Exception:
                    pass

            # 매 에폭 끝에 체크포인트 저장 (crash recovery)
            if rank == 0:
                _ckpt_path = os.path.join(fold_dir, 'last_checkpoint.pth')
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': _get_state_dict_for_save(model),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                        'best_val_primary': best_val_primary,
                        'best_val_auc': best_val_auc,
                        'best_val_auprc_at_best': best_val_auprc_at_best,
                        'best_val_auprc': best_val_primary,
                        'best_val_loss': best_val_loss,
                        'best_threshold': best_threshold,
                        'patience_counter': patience_counter,
                        'global_step': global_step,
                    }, _ckpt_path)
                except Exception as _ckpt_err:
                    print(f"⚠️ 체크포인트 저장 실패: {_ckpt_err}")

            # epoch 끝 동기화: 위의 rank 0 전용 I/O(best model·그래프·체크포인트 저장)가
            # 끝나기 전에 다른 rank 가 다음 epoch 의 ALLREDUCE 로 넘어가지 못하게 막는다.
            if use_ddp:
                _ddp_barrier(local_rank)

            # Early stopping: validation AUROC 기준 (warmup 이후 적용)
            if epoch >= config['warmup_epochs']:
                # If best not updated in this epoch OR skipped due to overfitting -> increment patience
                if (not did_best_commit) or overfitted_cv:
                    patience_counter += 1
                    if patience_counter >= config['patience']:
                        if rank == 0:
                            print(f"\n⏰ Early stopping triggered at epoch {epoch+1}")
                            print(f"   Val AUROC not improving for {config['patience']} epochs (Best: {best_val_primary:.4f}, Current: {val_auc:.4f})")
                        break

        if writer is not None:
            writer.close()

        # 학습은 끝났으므로 OOF 평가 전에 큰 optimizer state를 먼저 반환합니다.
        # 특히 EVA-X에서 학습 모델+optimizer를 유지한 채 평가 모델을 하나 더 만들면
        # native UCX/CUDA 경로가 segmentation fault로 종료될 수 있습니다.
        optimizer = None
        scheduler = None
        scaler = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -----------------------------
        # OOF (Out-of-Fold) prediction collection
        # - In simple 5-fold CV, evaluate best model on val fold and collect for pooled OOF evaluation
        # -----------------------------
        if not getattr(args, 'cv5_inner_811', False):
            try:
                if os.path.exists(model_path):
                    model_eval = _load_checkpoint_into_existing_model(model, model_path)
                    print("♻️ Best checkpoint를 기존 모델에 로드해 OOF 평가 (추가 모델 생성 없음)")
                else:
                    model_eval = model

                oof_loss, oof_acc, oof_auc, oof_auprc_val, oof_sens, oof_labels, oof_preds, oof_probs, oof_dirs = validate_epoch_fn(
                    model_eval, val_loader, criterion, device, use_bf16=use_bf16, use_fp32=use_fp32
                )
                oof_labels_all.extend(list(oof_labels))
                oof_probs_all.extend(list(oof_probs))
                oof_dirs_all.extend(list(oof_dirs))
                # B2: val_idx 순서대로 subject_id(group) 를 함께 수집
                try:
                    _val_subj = [all_groups[_i] for _i in val_idx.tolist()]
                    oof_subject_ids_all.extend(_val_subj[:len(oof_labels)])
                except Exception:
                    oof_subject_ids_all.extend([None] * len(oof_labels))
                oof_folds_all.extend([int(fold_idx)] * len(oof_labels))
                oof_thresholds_per_fold.append(float(best_threshold))

                # Save fold val results (best model re-evaluation)
                oof_metrics_fold = compute_metrics_with_threshold(oof_labels, oof_probs, best_threshold)
                oof_calibration_fold = compute_primary_calibration_metrics(oof_labels, oof_probs, n_bins=10)
                with open(os.path.join(fold_dir, 'fold_val_results.json'), 'w') as fvf:
                    json.dump({
                        'metrics_at_best_thr': oof_metrics_fold,
                        'auc': float(oof_auc),
                        'auprc': float(oof_auprc_val),
                        'primary_calibration_metrics': oof_calibration_fold,
                    }, fvf, indent=2)

                # Save fold val plots
                y_pred_oof_fold = (np.array(oof_probs) >= best_threshold).astype(int)
                save_confusion_matrix(oof_labels, y_pred_oof_fold, os.path.join(fold_dir, 'confusion_matrix_val_best.png'))
                save_roc_curve(oof_labels, oof_probs, os.path.join(fold_dir, 'roc_curve_val_best.png'))
                save_pr_curve(oof_labels, oof_probs, os.path.join(fold_dir, 'pr_curve_val_best.png'))
                sampled_val_best = None
                try:
                    sampled_val_best = save_example_predictions_grid(
                        oof_labels, oof_probs, oof_dirs, best_threshold,
                        os.path.join(fold_dir, 'example_predictions_val_best.png'),
                        data_root, per_row=4,
                    )
                except Exception as e:
                    print(f"⚠️ Fold val_best 예측 예시 그리드 생성 실패: {e}")
                _run_xai_suite(model_eval, sampled_val_best, fold_dir, "val_best")

                print(f"📊 Fold OOF - AUC: {oof_auc:.4f}, AUPRC: {oof_auprc_val:.4f}, Sens: {oof_metrics_fold['sensitivity']:.4f}, Spec: {oof_metrics_fold['specificity']:.4f}")
            except Exception as e:
                print(f"⚠️ OOF collection failed: {e}")

        # -----------------------------
        # Fold held-out (outer) evaluation (only when cv5_inner_811 is used)
        # - When cv5_inner_811 is used, test_loader corresponds to the fold held-out split (val_idx).
        # - Evaluate the BEST checkpoint (model_path) and store fold-heldout metrics for paper-ready CV summary.
        # -----------------------------
        fold_outer = None
        te_labels = te_probs = None
        if getattr(args, 'cv5_inner_811', False):
            try:
                if os.path.exists(model_path):
                    model_eval = _load_checkpoint_into_existing_model(model, model_path)
                    print("♻️ Best checkpoint를 기존 모델에 로드해 held-out 평가 (추가 모델 생성 없음)")
                else:
                    model_eval = model

                global_step += 1
                te_loss, te_acc, te_auc, te_auprc, te_sens, te_labels, te_preds, te_probs, te_dirs = validate_epoch_fn(
                    model_eval, test_loader, criterion, device, use_bf16=use_bf16, use_fp32=use_fp32
                )
                test_metrics = compute_metrics_with_threshold(te_labels, te_probs, best_threshold)
                fold_outer = {
                    'auc': float(te_auc),
                    'auprc': float(te_auprc),
                    'metrics_at_best_thr': test_metrics,
                }

                # Save plots/files for held-out (outer) evaluation
                y_pred_test = (np.array(te_probs) >= best_threshold).astype(int)
                save_confusion_matrix(te_labels, y_pred_test, os.path.join(fold_dir, 'confusion_matrix_test.png'))
                save_roc_curve(te_labels, te_probs, os.path.join(fold_dir, 'roc_curve_test.png'))
                save_pr_curve(te_labels, te_probs, os.path.join(fold_dir, 'pr_curve_test.png'))
                cal_path_t = os.path.join(fold_dir, 'calibration_test.png')
                dca_path_t = os.path.join(fold_dir, 'dca_test.png')
                thr_arr, dca_rows, nb_all, nb_none = decision_curve(te_labels, te_probs)
                try:
                    save_calibration_plot(te_labels, te_probs, cal_path_t, n_bins=10)
                    save_dca_plot(thr_arr, dca_rows, nb_all, nb_none, dca_path_t)
                except Exception:
                    pass

                sampled_test = None
                try:
                    sampled_test = save_example_predictions_grid(
                        te_labels, te_probs, te_dirs, best_threshold,
                        os.path.join(fold_dir, 'example_predictions_test.png'),
                        data_root, per_row=4,
                    )
                except Exception as e:
                    print(f"⚠️ Fold held-out 예측 예시 그리드 생성 실패: {e}")

                _run_xai_suite(model_eval, sampled_test, fold_dir, "test")

                with open(os.path.join(fold_dir, 'fold_test_results.json'), 'w') as ftf:
                    json.dump(
                        {
                            'metrics_at_best_thr': test_metrics,
                            'auc': float(te_auc),
                            'auprc': float(te_auprc),
                        },
                        ftf,
                        indent=2,
                    )

                print(f"🧪 Fold held-out(outer) - AUC: {te_auc:.4f}, AUPRC: {te_auprc:.4f}, Thr(from inner): {best_threshold:.3f}")

                # OOF for outer-test ensemble threshold tuning should use fold held-out predictions
                if use_outer_test and te_labels is not None and te_probs is not None:
                    oof_labels_all.extend(list(te_labels))
                    oof_probs_all.extend(list(te_probs))
                    oof_dirs_all.extend(list(te_dirs))
                    try:
                        _te_subj = [all_groups[_i] for _i in val_idx.tolist()]
                        oof_subject_ids_all.extend(_te_subj[:len(te_labels)])
                    except Exception:
                        oof_subject_ids_all.extend([None] * len(te_labels))
                    oof_folds_all.extend([int(fold_idx)] * len(te_labels))
                    oof_thresholds_per_fold.append(float(best_threshold))

                if use_wandb:
                    try:
                        wandb.log(
                            {
                                'images/test_confusion_matrix': wandb.Image(os.path.join(fold_dir, 'confusion_matrix_test.png')),
                                'images/test_roc_curve': wandb.Image(os.path.join(fold_dir, 'roc_curve_test.png')),
                                'images/test_pr_curve': wandb.Image(os.path.join(fold_dir, 'pr_curve_test.png')),
                                'images/test_calibration': wandb.Image(cal_path_t) if os.path.exists(cal_path_t) else None,
                                'images/test_dca': wandb.Image(dca_path_t) if os.path.exists(dca_path_t) else None,
                                'test/auc': float(te_auc),
                                'test/auprc': float(te_auprc),
                                'test/sens_at_best': float(test_metrics.get('sensitivity', 0.0)),
                                'test/spec_at_best': float(test_metrics.get('specificity', 0.0)),
                                'test/threshold_from_inner': float(best_threshold),
                            },
                            step=global_step,
                        )
                    except Exception:
                        pass
            except Exception as e:
                print(f"⚠️ Fold held-out evaluation skipped: {e}")

        fold_result = {
            'fold': fold_idx + 1,
            'config': config,
            'best_selection_metric': 'AUROC',
            'best_auc': best_val_auc,
            'best_auprc': best_val_auprc_at_best,
            'best_threshold': best_threshold,
            'best_metrics': best_metrics,
            'outer_test': fold_outer,
            'history': history,
            'roc_curve_val': {
                'fpr': [float(x) for x in fpr_val.tolist()] if fpr_val.size > 0 else [],
                'tpr': [float(x) for x in tpr_val.tolist()] if tpr_val.size > 0 else []
            }
        }
        fold_results.append(fold_result)
        with open(os.path.join(fold_dir, 'fold_results.json'), 'w') as f:
            fold_result_to_save = dict(fold_result)
            fold_result_to_save['run_meta'] = run_meta
            json.dump(fold_result_to_save, f, indent=4)
        print(f"\n✅ Fold {fold_idx+1} 완료!")
        if best_metrics:
            print(f"   Best AUROC: {best_val_auc:.4f}, AUPRC: {best_val_auprc_at_best:.4f}")
            print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
            print(f"   Sensitivity: {best_metrics['sensitivity']:.4f}")
            print(f"   Specificity: {best_metrics['specificity']:.4f}")

        # Collect fold models for outer test ensemble evaluation
        if use_outer_test:
            fold_model_paths.append(model_path)

        # 윈도우 좀비 워커 프로세스 강제 종료 및 메모리 반환
        import gc
        if 'train_loader' in locals():
            _shutdown_dataloader_workers(train_loader)
            del train_loader
        if 'val_loader' in locals():
            _shutdown_dataloader_workers(val_loader)
            del val_loader
        if 'test_loader' in locals():
            _shutdown_dataloader_workers(test_loader)
            del test_loader
        if 'model_eval' in locals():
            del model_eval
        if 'model' in locals() and not _reuse_model_across_folds:
            del model
        gc.collect()

        # 다음 fold 전에 GPU 캐시 비우기 (OOM 방지)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # CV summary
    executed_folds = len(fold_results)
    print(f"\n{'='*60}\n📊 CV 결과 요약 (executed {executed_folds}/{total_planned_folds} folds)\n{'='*60}")

    # Per-fold validation metrics at best checkpoint (AUROC for selection; AUROC+AUPRC both reported)
    fold_auc_scores = [r.get('best_auc') for r in fold_results if r.get('best_auc') is not None]
    fold_auc_scores = [float(x) for x in fold_auc_scores if x is not None and float(x) >= 0.0]
    fold_auprc_scores = [r.get('best_auprc') for r in fold_results if r.get('best_auprc') is not None]
    fold_auprc_scores = [float(x) for x in fold_auprc_scores if x is not None and float(x) >= 0.0]

    if fold_auc_scores:
        print(f"\nPer-fold AUROC (best checkpoint): {[f'{x:.4f}' for x in fold_auc_scores]}")
        print(f"Mean AUROC: {np.mean(fold_auc_scores):.4f} +/- {np.std(fold_auc_scores):.4f}")
    if fold_auprc_scores:
        print(f"Per-fold AUPRC (best checkpoint): {[f'{x:.4f}' for x in fold_auprc_scores]}")
        print(f"Mean AUPRC: {np.mean(fold_auprc_scores):.4f} +/- {np.std(fold_auprc_scores):.4f}")

    # OOF (Out-of-Fold) pooled evaluation with Bootstrap 95% CI
    oof_summary = {}
    if oof_labels_all and oof_probs_all:
        oof_failed = False
        oof_error_msg = ""
        try:
            oof_labels_arr = np.array(oof_labels_all)
            oof_probs_arr = np.array(oof_probs_all)
            oof_auprc_pooled = average_precision_score(oof_labels_arr, oof_probs_arr)
            oof_auc_pooled = roc_auc_score(oof_labels_arr, oof_probs_arr)
            oof_primary_calibration = compute_primary_calibration_metrics(
                oof_labels_arr, oof_probs_arr, n_bins=10
            )

            # -----------------------------------------------------------------
            # A2 수정: pooled OOF 에서 threshold 를 재-튜닝하지 않음.
            #   대신 각 fold 가 자기 validation 에서 결정한 best_threshold 를
            #   그 fold 의 OOF 예측에 적용 (샘플별 threshold).
            #   → threshold-dependent 지표의 '순환 튜닝' 편향 제거.
            #   AUROC/AUPRC 는 threshold-free 이므로 pooled OOF 에서 둘 다 보고.
            # -----------------------------------------------------------------
            if oof_folds_all and oof_thresholds_per_fold and len(oof_folds_all) == len(oof_labels_all):
                _thr_per_sample = np.array(
                    [oof_thresholds_per_fold[f] if 0 <= f < len(oof_thresholds_per_fold)
                     else float(np.mean(oof_thresholds_per_fold)) for f in oof_folds_all],
                    dtype=float,
                )
                y_pred_oof = (oof_probs_arr >= _thr_per_sample).astype(int)
                _threshold_strategy = 'per_fold'
                oof_thr_reported = float(np.mean(oof_thresholds_per_fold))
            else:
                # Fallback: per-fold threshold 가 없으면 전역 threshold 하나만 튜닝해서 사용
                # (이전 동작과 호환). 이 경우 A2 편향이 남음을 출력으로 경고.
                oof_thr_reported = float(tune_threshold(
                    oof_labels_all, oof_probs_all,
                    mode=config.get('thr_mode', 'youden'),
                    min_specificity=config.get('min_specificity', 0.9),
                    min_sensitivity=config.get('min_sensitivity', 0.8),
                ))
                _thr_per_sample = np.full(len(oof_labels_arr), oof_thr_reported, dtype=float)
                y_pred_oof = (oof_probs_arr >= oof_thr_reported).astype(int)
                _threshold_strategy = 'pooled_retune_fallback'
                print("⚠️ per-fold threshold 정보가 없어 pooled OOF 에 단일 threshold 를 재-튜닝했습니다 (A2 편향 남음).")

            oof_metrics = compute_metrics_from_pred(oof_labels_arr, y_pred_oof, oof_probs_arr)
            oof_mcc = oof_metrics['mcc']
            n_boot = 2000
            oof_ci = bootstrap_metric_cis_per_sample_threshold(
                oof_labels_arr, oof_probs_arr, _thr_per_sample,
                n_boot=n_boot, seed=config['seed'],
            )
            auprc_ci_lo, auprc_ci_hi = oof_ci['auprc']
            auc_ci_lo, auc_ci_hi = oof_ci['auc']

            print(f"\n{'='*60}")
            print(f"📊 OOF Pooled Results (n={len(oof_labels_arr)}, pos={int(oof_labels_arr.sum())}, neg={int(len(oof_labels_arr) - oof_labels_arr.sum())})")
            print(f"{'='*60}")
            print(f"   AUROC: {oof_auc_pooled:.4f} (95% CI: {auc_ci_lo:.4f} - {auc_ci_hi:.4f})")
            print(f"   AUPRC: {oof_auprc_pooled:.4f} (95% CI: {auprc_ci_lo:.4f} - {auprc_ci_hi:.4f})")
            print(f"   Brier: {oof_primary_calibration['brier_score']:.4f}, "
                  f"ECE: {oof_primary_calibration['ece']:.4f}, "
                  f"Cal intercept/slope: {oof_primary_calibration['calibration_intercept']:.4f} / "
                  f"{oof_primary_calibration['calibration_slope']:.4f}")
            print(f"   Threshold strategy: {_threshold_strategy} (mean fold-thr={oof_thr_reported:.3f})")
            if oof_thresholds_per_fold:
                print(f"   Per-fold thresholds: {[f'{t:.3f}' for t in oof_thresholds_per_fold]}")
            print(f"   Accuracy: {oof_metrics['accuracy']:.4f} (95% CI: {oof_ci['accuracy'][0]:.4f} - {oof_ci['accuracy'][1]:.4f})")
            print(f"   Sensitivity: {oof_metrics['sensitivity']:.4f} (95% CI: {oof_ci['sensitivity'][0]:.4f} - {oof_ci['sensitivity'][1]:.4f})")
            print(f"   Specificity: {oof_metrics['specificity']:.4f} (95% CI: {oof_ci['specificity'][0]:.4f} - {oof_ci['specificity'][1]:.4f})")
            print(f"   Precision: {oof_metrics['precision']:.4f} (95% CI: {oof_ci['precision'][0]:.4f} - {oof_ci['precision'][1]:.4f})")
            print(f"   F1: {oof_metrics['f1_score']:.4f} (95% CI: {oof_ci['f1_score'][0]:.4f} - {oof_ci['f1_score'][1]:.4f})")
            print(f"   MCC: {oof_mcc:.4f} (95% CI: {oof_ci['mcc'][0]:.4f} - {oof_ci['mcc'][1]:.4f})")
            if _threshold_strategy == 'per_fold':
                print("   (threshold-dependent metrics use per-fold thresholds — no pooled re-tuning)")
            else:
                print("   ⚠️ threshold-dependent metrics may be optimistic (threshold was retuned on pooled OOF).")

            # ----- Key operating points: Spec=0.80 & Youden-Optimal (OOF, 95% CI) -----
            # 이 2개 운영점은 pooled OOF 에서 직접 튜닝된 것(탐색적 비교용)이므로 threshold-dep 지표는 optimistic 임을 명시.
            thr_spec80 = find_threshold_for_spec(oof_labels_arr, oof_probs_arr, target_spec=0.80)
            thr_youden = tune_threshold(oof_labels_all, oof_probs_all, mode='youden')
            metrics_spec80 = compute_metrics_with_threshold(oof_labels_all, oof_probs_all, thr_spec80)
            metrics_youden = compute_metrics_with_threshold(oof_labels_all, oof_probs_all, thr_youden)
            ci_spec80 = bootstrap_oof_metric_cis(oof_labels_arr, oof_probs_arr, thr_spec80, n_boot=n_boot, seed=config['seed'])
            ci_youden = bootstrap_oof_metric_cis(oof_labels_arr, oof_probs_arr, thr_youden, n_boot=n_boot, seed=config['seed'])
            model_display = (args.arch or 'model').replace('_', '-')
            if model_display.lower() == 'eva-x-small':
                model_display = 'EVA-X-S'
            elif model_display.lower() == 'dinov3':
                model_display = 'DINOv3-S'
            elif model_display.lower() == 'vit':
                model_display = 'DeiT III-S'
            elif model_display.lower() == 'convnextv2':
                model_display = 'ConvNeXt V2'
            print(f"\n   --- Model performance at key operating points (OOF predictions, 95% CI) ---")
            print(f"   Model      | Spec=0.80 Point          | Youden-Optimal Point")
            print(f"   -----------|--------------------------|-----------------------")
            print(f"              | Sens / Spec / MCC        | Sens / Spec / MCC")
            print(f"   {model_display:10} | {metrics_spec80['sensitivity']:.3f} / {metrics_spec80['specificity']:.3f} / {metrics_spec80['mcc']:.3f}   | {metrics_youden['sensitivity']:.3f} / {metrics_youden['specificity']:.3f} / {metrics_youden['mcc']:.3f}")
            print(f"              | ({ci_spec80['sensitivity'][0]:.3f}\u2013{ci_spec80['sensitivity'][1]:.3f})           | ({ci_youden['sensitivity'][0]:.3f}\u2013{ci_youden['sensitivity'][1]:.3f})")
            print(f"   (Thresholds: Spec=0.80 on OOF, Youden max J=sens+spec-1)")

            # Save OOF plots
            oof_dir = os.path.join(save_dir, 'oof_pooled')
            os.makedirs(oof_dir, exist_ok=True)
            save_confusion_matrix(oof_labels_all, y_pred_oof.tolist(), os.path.join(oof_dir, 'confusion_matrix_oof.png'))
            save_roc_curve(oof_labels_all, oof_probs_arr.tolist(), os.path.join(oof_dir, 'roc_curve_oof.png'))
            save_pr_curve(oof_labels_all, oof_probs_arr.tolist(), os.path.join(oof_dir, 'pr_curve_oof.png'))
            try:
                save_calibration_plot(oof_labels_all, oof_probs_arr.tolist(), os.path.join(oof_dir, 'calibration_oof.png'), n_bins=10)
                thr_arr, dca_rows, nb_all, nb_none = decision_curve(oof_labels_all, oof_probs_arr.tolist())
                save_dca_plot(thr_arr, dca_rows, nb_all, nb_none, os.path.join(oof_dir, 'dca_oof.png'))
            except Exception:
                pass
            try:
                save_example_predictions_grid(
                    oof_labels_all, oof_probs_arr.tolist(),
                    oof_dirs_all,
                    oof_thr_reported,
                    os.path.join(oof_dir, 'example_predictions_oof.png'),
                    data_root, per_row=4,
                )
            except Exception:
                pass

            # -----------------------------------------------------------------
            # B2 수정: Patient(subject)-level 집계 메트릭
            #   임상 의사결정 단위는 환자이므로, 환자당 확률(=이미지별 확률의 평균)로
            #   AUROC/AUPRC/Sens/Spec/F1/MCC 를 함께 보고.
            # -----------------------------------------------------------------
            patient_summary = None
            if oof_subject_ids_all and any(s is not None for s in oof_subject_ids_all):
                try:
                    pat_labels, pat_probs, pat_ids = aggregate_patient_level(
                        oof_labels_all, oof_probs_arr.tolist(), oof_subject_ids_all, agg='mean'
                    )
                    if len(pat_labels) > 0 and len(set(pat_labels)) > 1:
                        pat_labels_arr = np.array(pat_labels, dtype=int)
                        pat_probs_arr = np.array(pat_probs, dtype=float)
                        pat_auc = float(roc_auc_score(pat_labels_arr, pat_probs_arr))
                        pat_auprc = float(average_precision_score(pat_labels_arr, pat_probs_arr))
                        # patient-level threshold: 이미지-level per-fold thr 의 평균 (환자 앙상블 특성상 단일 기준이 필요)
                        pat_thr = oof_thr_reported
                        pat_y_pred = (pat_probs_arr >= pat_thr).astype(int)
                        pat_metrics = compute_metrics_from_pred(pat_labels_arr, pat_y_pred, pat_probs_arr)
                        pat_ci = bootstrap_oof_metric_cis(
                            pat_labels_arr, pat_probs_arr, pat_thr,
                            n_boot=n_boot, seed=config['seed'],
                        )
                        print(f"\n   --- Patient-level (mean prob per subject, n_patients={len(pat_ids)}) ---")
                        print(f"   AUROC: {pat_auc:.4f} (95% CI: {pat_ci['auc'][0]:.4f} - {pat_ci['auc'][1]:.4f})")
                        print(f"   AUPRC: {pat_auprc:.4f} (95% CI: {pat_ci['auprc'][0]:.4f} - {pat_ci['auprc'][1]:.4f})")
                        print(f"   Sens/Spec/F1/MCC @ thr={pat_thr:.3f}: "
                              f"{pat_metrics['sensitivity']:.3f} / {pat_metrics['specificity']:.3f} / "
                              f"{pat_metrics['f1_score']:.3f} / {pat_metrics['mcc']:.3f}")
                        patient_summary = {
                            'n_patients': int(len(pat_ids)),
                            'n_pos_patients': int(pat_labels_arr.sum()),
                            'n_neg_patients': int(len(pat_labels_arr) - pat_labels_arr.sum()),
                            'agg': 'mean',
                            'threshold': float(pat_thr),
                            'auc': pat_auc,
                            'auprc': pat_auprc,
                            'metrics': pat_metrics,
                            'ci_95': pat_ci,
                        }
                        try:
                            save_confusion_matrix(
                                pat_labels, pat_y_pred.tolist(),
                                os.path.join(oof_dir, 'confusion_matrix_patient_level.png'),
                            )
                            save_roc_curve(pat_labels, pat_probs_arr.tolist(), os.path.join(oof_dir, 'roc_curve_patient_level.png'))
                            save_pr_curve(pat_labels, pat_probs_arr.tolist(), os.path.join(oof_dir, 'pr_curve_patient_level.png'))
                        except Exception as _pp_err:
                            print(f"⚠️ Patient-level plot 실패: {_pp_err}")
                except Exception as _pat_err:
                    print(f"⚠️ Patient-level 집계 실패: {_pat_err}")

            oof_summary = {
                'n_samples': int(len(oof_labels_arr)),
                'n_pos': int(oof_labels_arr.sum()),
                'n_neg': int(len(oof_labels_arr) - oof_labels_arr.sum()),
                'primary_calibration_metrics': oof_primary_calibration,
                'oof_auprc': float(oof_auprc_pooled),
                'oof_auprc_95ci': [float(auprc_ci_lo), float(auprc_ci_hi)],
                'oof_auc': float(oof_auc_pooled),
                'oof_auc_95ci': [float(auc_ci_lo), float(auc_ci_hi)],
                'oof_accuracy_95ci': [float(oof_ci['accuracy'][0]), float(oof_ci['accuracy'][1])],
                'oof_sensitivity_95ci': [float(oof_ci['sensitivity'][0]), float(oof_ci['sensitivity'][1])],
                'oof_specificity_95ci': [float(oof_ci['specificity'][0]), float(oof_ci['specificity'][1])],
                'oof_precision_95ci': [float(oof_ci['precision'][0]), float(oof_ci['precision'][1])],
                'oof_f1_95ci': [float(oof_ci['f1_score'][0]), float(oof_ci['f1_score'][1])],
                'oof_mcc_95ci': [float(oof_ci['mcc'][0]), float(oof_ci['mcc'][1])],
                'oof_threshold_reported': float(oof_thr_reported),
                'oof_threshold_strategy': str(_threshold_strategy),
                'oof_thresholds_per_fold': [float(t) for t in oof_thresholds_per_fold],
                'oof_metrics': oof_metrics,
                'oof_mcc': float(oof_mcc),
                'n_bootstrap': n_boot,
                'patient_level': patient_summary,
                # Key operating points (for table: Spec=0.80 & Youden)
                'key_operating_points': {
                    'model_display': model_display,
                    'spec80': {
                        'threshold': float(thr_spec80),
                        'sensitivity': float(metrics_spec80['sensitivity']),
                        'specificity': float(metrics_spec80['specificity']),
                        'mcc': float(metrics_spec80['mcc']),
                        'sensitivity_95ci': [float(ci_spec80['sensitivity'][0]), float(ci_spec80['sensitivity'][1])],
                        'specificity_95ci': [float(ci_spec80['specificity'][0]), float(ci_spec80['specificity'][1])],
                        'mcc_95ci': [float(ci_spec80['mcc'][0]), float(ci_spec80['mcc'][1])],
                    },
                    'youden': {
                        'threshold': float(thr_youden),
                        'sensitivity': float(metrics_youden['sensitivity']),
                        'specificity': float(metrics_youden['specificity']),
                        'mcc': float(metrics_youden['mcc']),
                        'sensitivity_95ci': [float(ci_youden['sensitivity'][0]), float(ci_youden['sensitivity'][1])],
                        'specificity_95ci': [float(ci_youden['specificity'][0]), float(ci_youden['specificity'][1])],
                        'mcc_95ci': [float(ci_youden['mcc'][0]), float(ci_youden['mcc'][1])],
                    },
                },
            }

            # Save OOF sample-level predictions CSV
            try:
                import csv
                csv_path = os.path.join(oof_dir, 'oof_predictions.csv')
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvf:
                    writer = csv.writer(csvf)
                    writer.writerow(['img_path', 'label', 'prob'])
                    for d_i, l_i, p_i in zip(oof_dirs_all, oof_labels_all, oof_probs_arr.tolist()):
                        writer.writerow([str(d_i), int(l_i), float(p_i)])
                print(f"📄 OOF sample-level predictions saved: {csv_path}")
            except Exception as e:
                print(f"[WARN] Failed to save oof_predictions.csv: {e}")

            # Save OOF results JSON
            with open(os.path.join(oof_dir, 'oof_results.json'), 'w') as f:
                json.dump({
                    'run_meta': run_meta,
                    'primary_metrics_policy': {
                        'checkpoint_selection': 'validation_auroc',
                        'final_evaluation': ['auroc', 'auprc', 'brier_score', 'ece', 'calibration_intercept', 'calibration_slope'],
                        'threshold_metrics': 'secondary_report_only',
                    },
                    **oof_summary,
                    'per_fold_auprc': fold_auprc_scores,
                    'per_fold_auc': fold_auc_scores,
                    'mean_fold_auprc': float(np.mean(fold_auprc_scores)) if fold_auprc_scores else None,
                    'std_fold_auprc': float(np.std(fold_auprc_scores)) if fold_auprc_scores else None,
                }, f, indent=2)

            if use_wandb:
                try:
                    wandb.log({
                        'images/oof_confusion_matrix': wandb.Image(os.path.join(oof_dir, 'confusion_matrix_oof.png')),
                        'images/oof_roc_curve': wandb.Image(os.path.join(oof_dir, 'roc_curve_oof.png')),
                        'images/oof_pr_curve': wandb.Image(os.path.join(oof_dir, 'pr_curve_oof.png')),
                        'oof/auprc': float(oof_auprc_pooled),
                        'oof/auprc_ci_lo': float(auprc_ci_lo),
                        'oof/auprc_ci_hi': float(auprc_ci_hi),
                        'oof/auc': float(oof_auc_pooled),
                        'oof/auc_ci_lo': float(auc_ci_lo),
                        'oof/auc_ci_hi': float(auc_ci_hi),
                        'oof/accuracy_ci_lo': float(oof_ci['accuracy'][0]),
                        'oof/accuracy_ci_hi': float(oof_ci['accuracy'][1]),
                        'oof/sensitivity_ci_lo': float(oof_ci['sensitivity'][0]),
                        'oof/sensitivity_ci_hi': float(oof_ci['sensitivity'][1]),
                        'oof/specificity_ci_lo': float(oof_ci['specificity'][0]),
                        'oof/specificity_ci_hi': float(oof_ci['specificity'][1]),
                        'oof/precision_ci_lo': float(oof_ci['precision'][0]),
                        'oof/precision_ci_hi': float(oof_ci['precision'][1]),
                        'oof/f1_ci_lo': float(oof_ci['f1_score'][0]),
                        'oof/f1_ci_hi': float(oof_ci['f1_score'][1]),
                        'oof/mcc_ci_lo': float(oof_ci['mcc'][0]),
                        'oof/mcc_ci_hi': float(oof_ci['mcc'][1]),
                        'oof/threshold': float(oof_thr_reported),
                        'oof/sensitivity': float(oof_metrics['sensitivity']),
                        'oof/specificity': float(oof_metrics['specificity']),
                        'oof/f1': float(oof_metrics['f1_score']),
                        'oof/mcc': float(oof_mcc),
                    })
                except Exception:
                    pass
        except Exception as e:
            import traceback as _tb
            oof_failed = True
            oof_error_msg = _tb.format_exc()
            print(f"⚠️ OOF pooled evaluation failed, continue without OOF summary: {e}")

    # -----------------------------------------------------------------
    # Final Outer (Holdout) Test evaluation — recommended IEEE/RSNA 방식
    #   1) OOF 예측에서 global threshold 를 결정 (학습·튜닝 데이터 내부에서만)
    #   2) 독립된 holdout test 에 fold 앙상블(평균 확률) 한 번 적용
    #   3) AUROC/AUPRC/Sens/Spec/F1/MCC + bootstrap 95% CI 저장
    # -----------------------------------------------------------------
    outer_test_summary = None
    if use_outer_test and rank == 0:
        if not (oof_labels_all and oof_probs_all):
            print("⚠️ Outer test 평가 건너뜀: OOF 예측이 비어 있어 global threshold 를 튜닝할 수 없습니다.")
        elif not fold_model_paths:
            print("⚠️ Outer test 평가 건너뜀: 저장된 fold best_model 이 없습니다.")
        else:
            outer_dir = os.path.join(save_dir, 'outer_test')
            os.makedirs(outer_dir, exist_ok=True)
            print(f"\n{'-'*60}")
            print(f"🧪 Holdout Outer Test 평가 ({len(outer_test_idx)}장) — {len(fold_model_paths)}-fold 앙상블")
            tuned_thr_outer = tune_threshold(
                oof_labels_all, oof_probs_all,
                mode=config.get('thr_mode', 'youden'),
                min_specificity=config.get('min_specificity', 0.9),
                min_sensitivity=config.get('min_sensitivity', 0.8),
            )
            print(f"🔧 OOF 기반 global threshold: {tuned_thr_outer:.3f} (mode={config.get('thr_mode','youden')})")

            # Build outer test dataset/loader ONCE (모델마다 재생성하지 않음)
            test_dataset_outer = _build_dataset(
                outer_test_idx, train_mode=False, split_name="outer_test"
            )
            test_loader_outer = DataLoader(
                test_dataset_outer, shuffle=False,
                collate_fn=collate_skip_none, **eval_loader_kwargs,
            )
            _ot_fp32 = bool(config.get('use_fp32', False))
            _ot_bf16 = bool(config.get('use_bf16', True)) and not _ot_fp32

            test_probs_ens = None
            test_labels_ref = None
            test_dirs_ref = None
            per_fold_outer_probs: List[List[float]] = []
            valid_models = 0
            model_eval = None
            try:
                # 한 모델만 만든 뒤 fold checkpoint를 순서대로 덮어써서 평가합니다.
                # EVA-X native loader 반복 호출과 GPU 메모리 중복을 피합니다.
                if _fold_model_cache is not None:
                    model_eval = _fold_model_cache
                    print("♻️ Fold 학습 모델을 outer-test 앙상블 평가에도 재사용")
                else:
                    model_eval = _build_model_instance()
                for mp in fold_model_paths:
                    try:
                        _load_checkpoint_into_existing_model(model_eval, mp)
                        tl, ta, tc_auc, tc_auprc, ts, y_true_t, y_pred_t, y_prob_t, y_dirs_t = validate_epoch_fn(
                            model_eval, test_loader_outer, criterion, device,
                            use_bf16=_ot_bf16, use_fp32=_ot_fp32,
                        )
                        _probs_arr = np.array(y_prob_t, dtype=float)
                        per_fold_outer_probs.append(_probs_arr.tolist())
                        if test_probs_ens is None:
                            test_probs_ens = _probs_arr.copy()
                            test_labels_ref = np.array(y_true_t, dtype=int)
                            test_dirs_ref = list(y_dirs_t)
                        else:
                            test_probs_ens = test_probs_ens + _probs_arr
                        valid_models += 1
                    except Exception as e:
                        print(f"⚠️ Outer test: skipped model {mp}: {e}")
            finally:
                if model_eval is not None:
                    del model_eval
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if test_probs_ens is None or test_labels_ref is None or valid_models <= 0:
                print("⚠️ Outer test ensemble 계산 가능한 모델이 없습니다. 평가를 건너뜁니다.")
            else:
                n_models = int(valid_models)
                test_probs_ens = (test_probs_ens / float(n_models)).astype(float)
                test_probs_list = test_probs_ens.tolist()
                test_labels_list = test_labels_ref.tolist()

                # Primary metrics (threshold-free + threshold-based)
                test_auc = roc_auc_score(test_labels_ref, test_probs_ens) if len(set(test_labels_list)) > 1 else 0.0
                test_auprc = average_precision_score(test_labels_ref, test_probs_ens) if len(set(test_labels_list)) > 1 else 0.0
                outer_primary_calibration = compute_primary_calibration_metrics(
                    test_labels_ref, test_probs_ens, n_bins=10
                )
                test_metrics_outer = compute_metrics_with_threshold(
                    test_labels_list, test_probs_list, tuned_thr_outer
                )

                # Bootstrap 95% CI (논문용)
                try:
                    test_ci = bootstrap_oof_metric_cis(
                        test_labels_ref, test_probs_ens, tuned_thr_outer,
                        n_boot=2000, seed=config['seed'],
                    )
                except Exception as _ci_err:
                    print(f"⚠️ Outer test bootstrap CI 계산 실패: {_ci_err}")
                    test_ci = None

                # Key operating points (Spec=0.80 & Youden, OOF-derived thresholds)
                try:
                    thr_spec80_outer = find_threshold_for_spec(
                        np.array(oof_labels_all, dtype=int),
                        np.array(oof_probs_all, dtype=float),
                        target_spec=0.80,
                    )
                    thr_youden_outer = tune_threshold(oof_labels_all, oof_probs_all, mode='youden')
                    metrics_spec80_outer = compute_metrics_with_threshold(test_labels_list, test_probs_list, thr_spec80_outer)
                    metrics_youden_outer = compute_metrics_with_threshold(test_labels_list, test_probs_list, thr_youden_outer)
                except Exception:
                    thr_spec80_outer = thr_youden_outer = None
                    metrics_spec80_outer = metrics_youden_outer = None

                # Plots
                try:
                    y_pred_outer = (test_probs_ens >= tuned_thr_outer).astype(int).tolist()
                    save_confusion_matrix(
                        test_labels_list, y_pred_outer,
                        os.path.join(outer_dir, 'confusion_matrix_outer_test.png'),
                    )
                    save_roc_curve(test_labels_list, test_probs_list, os.path.join(outer_dir, 'roc_curve_outer_test.png'))
                    save_pr_curve(test_labels_list, test_probs_list, os.path.join(outer_dir, 'pr_curve_outer_test.png'))
                    save_calibration_plot(
                        test_labels_list, test_probs_list,
                        os.path.join(outer_dir, 'calibration_outer_test.png'), n_bins=10,
                    )
                    thr_arr_t, dca_rows_t, nb_all_t, nb_none_t = decision_curve(test_labels_list, test_probs_list)
                    save_dca_plot(thr_arr_t, dca_rows_t, nb_all_t, nb_none_t, os.path.join(outer_dir, 'dca_outer_test.png'))
                    if test_dirs_ref is not None:
                        save_example_predictions_grid(
                            test_labels_list, test_probs_list, test_dirs_ref, tuned_thr_outer,
                            os.path.join(outer_dir, 'example_predictions_outer_test.png'),
                            data_root, per_row=4,
                        )
                except Exception as _plot_err:
                    print(f"⚠️ Outer test 시각화 일부 실패: {_plot_err}")

                # CSV: sample-level predictions
                try:
                    import csv
                    csv_path = os.path.join(outer_dir, 'outer_test_predictions.csv')
                    with open(csv_path, 'w', newline='', encoding='utf-8') as csvf:
                        cw = csv.writer(csvf)
                        cw.writerow(['img_path', 'label', 'prob_ensemble'] + [f'prob_fold{i+1}' for i in range(len(per_fold_outer_probs))])
                        _dirs_iter = test_dirs_ref if test_dirs_ref is not None else [''] * len(test_labels_list)
                        for i_row in range(len(test_labels_list)):
                            row = [
                                str(_dirs_iter[i_row]) if i_row < len(_dirs_iter) else '',
                                int(test_labels_list[i_row]),
                                float(test_probs_list[i_row]),
                            ]
                            for _fp in per_fold_outer_probs:
                                row.append(float(_fp[i_row]) if i_row < len(_fp) else float('nan'))
                            cw.writerow(row)
                    print(f"📄 Outer test sample-level predictions saved: {csv_path}")
                except Exception as _csv_err:
                    print(f"[WARN] Failed to save outer_test_predictions.csv: {_csv_err}")

                outer_test_summary = {
                    'n_samples': int(len(test_labels_ref)),
                    'n_pos': int(test_labels_ref.sum()),
                    'n_neg': int(len(test_labels_ref) - test_labels_ref.sum()),
                    'prevalence': float(test_labels_ref.mean()) if len(test_labels_ref) else 0.0,
                    'n_models_ensembled': n_models,
                    'primary_calibration_metrics': outer_primary_calibration,
                    'threshold_from_oof': float(tuned_thr_outer),
                    'threshold_mode': str(config.get('thr_mode', 'youden')),
                    'auc': float(test_auc),
                    'auprc': float(test_auprc),
                    'metrics_at_threshold': test_metrics_outer,
                    'ci_95': test_ci,
                    'key_operating_points': None,
                }
                if metrics_spec80_outer is not None and metrics_youden_outer is not None:
                    outer_test_summary['key_operating_points'] = {
                        'spec80': {
                            'threshold': float(thr_spec80_outer),
                            'sensitivity': float(metrics_spec80_outer['sensitivity']),
                            'specificity': float(metrics_spec80_outer['specificity']),
                            'mcc': float(metrics_spec80_outer['mcc']),
                        },
                        'youden': {
                            'threshold': float(thr_youden_outer),
                            'sensitivity': float(metrics_youden_outer['sensitivity']),
                            'specificity': float(metrics_youden_outer['specificity']),
                            'mcc': float(metrics_youden_outer['mcc']),
                        },
                    }

                # Save JSON
                try:
                    with open(os.path.join(outer_dir, 'cv5_outer_test_results.json'), 'w', encoding='utf-8') as otf:
                        json.dump({
                            'run_meta': run_meta,
                            'primary_metrics_policy': {
                                'checkpoint_selection': 'validation_auroc',
                                'final_evaluation': ['auroc', 'auprc', 'brier_score', 'ece', 'calibration_intercept', 'calibration_slope'],
                                'threshold_metrics': 'secondary_report_only',
                            },
                            **outer_test_summary,
                        }, otf, indent=2)
                except Exception as _json_err:
                    print(f"⚠️ Outer test JSON 저장 실패: {_json_err}")

                print(f"✅ Holdout Outer Test - AUC: {test_auc:.4f}, AUPRC: {test_auprc:.4f}, "
                      f"Brier: {outer_primary_calibration['brier_score']:.4f}, "
                      f"ECE: {outer_primary_calibration['ece']:.4f}, "
                      f"Cal intercept/slope: {outer_primary_calibration['calibration_intercept']:.4f} / "
                      f"{outer_primary_calibration['calibration_slope']:.4f}, "
                      f"Acc: {test_metrics_outer['accuracy']:.4f}, Sens: {test_metrics_outer['sensitivity']:.4f}, "
                      f"Spec: {test_metrics_outer['specificity']:.4f}, F1: {test_metrics_outer['f1_score']:.4f}, "
                      f"MCC: {test_metrics_outer.get('mcc', 0):.4f}")
                if test_ci:
                    print(f"   95% CI - AUC: [{test_ci['auc'][0]:.4f}, {test_ci['auc'][1]:.4f}], "
                          f"AUPRC: [{test_ci['auprc'][0]:.4f}, {test_ci['auprc'][1]:.4f}], "
                          f"Sens: [{test_ci['sensitivity'][0]:.4f}, {test_ci['sensitivity'][1]:.4f}], "
                          f"Spec: [{test_ci['specificity'][0]:.4f}, {test_ci['specificity'][1]:.4f}]")

                if use_wandb:
                    try:
                        _wb_log = {
                            'outer_test/auc': float(test_auc),
                            'outer_test/auprc': float(test_auprc),
                            'outer_test/sensitivity': float(test_metrics_outer['sensitivity']),
                            'outer_test/specificity': float(test_metrics_outer['specificity']),
                            'outer_test/f1': float(test_metrics_outer['f1_score']),
                            'outer_test/mcc': float(test_metrics_outer.get('mcc', 0)),
                            'outer_test/accuracy': float(test_metrics_outer['accuracy']),
                            'outer_test/threshold': float(tuned_thr_outer),
                            'outer_test/n_samples': int(len(test_labels_ref)),
                            'images/outer_test_confusion_matrix': wandb.Image(os.path.join(outer_dir, 'confusion_matrix_outer_test.png')),
                            'images/outer_test_roc_curve': wandb.Image(os.path.join(outer_dir, 'roc_curve_outer_test.png')),
                            'images/outer_test_pr_curve': wandb.Image(os.path.join(outer_dir, 'pr_curve_outer_test.png')),
                        }
                        _cal_p = os.path.join(outer_dir, 'calibration_outer_test.png')
                        _dca_p = os.path.join(outer_dir, 'dca_outer_test.png')
                        if os.path.exists(_cal_p):
                            _wb_log['images/outer_test_calibration'] = wandb.Image(_cal_p)
                        if os.path.exists(_dca_p):
                            _wb_log['images/outer_test_dca'] = wandb.Image(_dca_p)
                        wandb.log(_wb_log)
                    except Exception:
                        pass

    summary = {
        'run_meta': run_meta,
        'config': config,
        'primary_metrics_policy': {
            'checkpoint_selection': 'validation_auroc',
            'final_evaluation': ['auroc', 'auprc', 'brier_score', 'ece', 'calibration_intercept', 'calibration_slope'],
            'threshold_metrics': 'secondary_report_only',
        },
        'executed_folds': executed_folds,
        'requested_n_folds': total_planned_folds,
        'fold_results': fold_results,
        'best_selection_metric': 'AUROC',
        'per_fold_auc': fold_auc_scores,
        'mean_fold_auc': float(np.mean(fold_auc_scores)) if fold_auc_scores else 0.0,
        'std_fold_auc': float(np.std(fold_auc_scores)) if fold_auc_scores else 0.0,
        'per_fold_auprc': fold_auprc_scores,
        'mean_fold_auprc': float(np.mean(fold_auprc_scores)) if fold_auprc_scores else 0.0,
        'std_fold_auprc': float(np.std(fold_auprc_scores)) if fold_auprc_scores else 0.0,
        'oof_pooled': oof_summary if oof_summary else None,
        'outer_test': outer_test_summary,
        'outer_test_enabled': bool(use_outer_test),
        'outer_test_ratio': float(args.outer_test_ratio) if use_outer_test else 0.0,
        'outer_test_n_samples': int(len(outer_test_idx)) if use_outer_test else 0,
        'uncertainty_training_policy': uncertainty_audit,
    }
    if rank == 0:
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"\n✅ 학습 완료! 결과 저장 위치: {save_dir}")
        if use_wandb and rank == 0:
            wandb.finish()
    if use_ddp:
        dist.destroy_process_group()
        if rank != 0 and sys.stdout != sys.__stdout__:
            sys.stdout.close()
            sys.stdout = sys.__stdout__


def _maybe_auto_launch_ddp() -> None:
    """옵션이 켜졌을 때만 GPU 여러 장에서 torch.distributed.run으로 재실행 (기본은 재실행하지 않음)."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        return
    want_auto = '--auto-ddp' in sys.argv
    if not want_auto:
        for _env in ('CXR_SINGLE_AUTO_DDP', 'SIAMESE_AUTO_DDP'):
            v = os.environ.get(_env, '').strip().lower()
            if v in ('1', 'true', 'yes', 'on'):
                want_auto = True
                break
    if not want_auto:
        return
    if '--help' in sys.argv or '-h' in sys.argv:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return
    n = int(torch.cuda.device_count())
    script = os.path.abspath(sys.argv[0])
    # 자식 프로세스에서 무한 재실행 방지: --auto-ddp 는 제거해 전달
    child_argv = [a for a in sys.argv[1:] if a != '--auto-ddp']
    cmd = [sys.executable, '-m', 'torch.distributed.run', f'--nproc_per_node={n}', script] + child_argv
    print(
        f'🔁 --auto-ddp: GPU {n}장으로 torch.distributed.run 재실행합니다.',
        flush=True,
    )
    os.execvp(sys.executable, cmd)


if __name__ == "__main__":
    _maybe_auto_launch_ddp()
    # GPU 정보 출력 (rank 0이거나 단일 프로세스일 때만)
    if os.environ.get('RANK', '0') == '0':
        print("CUDA available:", torch.cuda.is_available())
        print("GPU count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"[{i}] {p.name} | total {p.total_memory/1024**3:.1f} GB")
    main()


