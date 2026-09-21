#!/usr/bin/env python3
"""
CXR 폐 세그멘테이션 데이터셋 생성 (PA / AP / both 선택 가능).

--seg-mode 옵션으로 두 가지 마스크 소스를 선택할 수 있습니다:
  medsam3  (기본): MedSAM3 모델로 폐 마스크 추론
  chexmask        : ChexMask CSV(RLE)에서 폐 마스크 복원 (모델 추론 불필요)

--view 옵션으로 처리할 촬영 방향을 선택합니다:
  PA   (기본): Posterior-Anterior 영상만 처리
  AP          : Anterior-Posterior 영상만 처리
  both        : PA + AP 모두 처리

입력:
- 패키지 루트의 pneumonia_labels.json (image_abs_path, view_position 포함)

출력(medsam3 + PA, 기본):
- <DATA_BASE_DIR>/cxr_medsam3_lung_seg_pa/
  - lung_mask_npy/.../*.npy
  - masked_cxr/.../*.jpg
  - meta_json/.../*.json
  - manifest_pa_medsam3_ok.json
  - manifest_pa_medsam3_errors.json

출력(chexmask + AP 예시):
- <DATA_BASE_DIR>/cxr_chexmask_lung_seg_ap/
  - manifest_ap_chexmask_ok.json
  - manifest_ap_chexmask_errors.json

동작 (medsam3):
1) labels JSON에서 지정한 view_position 이고 pneumonia 라벨이 0 또는 1인 항목만 선택
   (null / -1 제외)
2) MedSAM3로 폐 마스크 추론
3) 마스크(.npy)와 배경 제거된 CXR(.jpg) 저장
4) 산출물 manifest JSON 저장 (성공/에러 분리)

동작 (chexmask):
1) 동일한 view 필터링
2) ChexMask CSV 로드 → {dicom_id: row} lookup 구성
3) dicom_id 매칭 → Left+Right Lung RLE 디코딩 → .npy + masked CXR 저장
4) 산출물 manifest JSON 저장 (성공/에러 분리)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from torchvision.ops import nms


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_BASE_DIR = Path(os.environ.get("DATA_BASE_DIR", REPO_ROOT)).resolve()

DEFAULT_LABEL_JSON = REPO_ROOT / "pneumonia_labels.json"
DEFAULT_OUT_ROOT = DATA_BASE_DIR / "cxr_medsam3_lung_seg"
DEFAULT_CHEXMASK_OUT_ROOT = DATA_BASE_DIR / "cxr_chexmask_lung_seg"
DEFAULT_CHEXMASK_CSV = Path(os.environ.get("CHEXMASK_CSV", REPO_ROOT / "data" / "ChexMask_MIMIC-CXR-JPG.csv"))
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "medsam3_full_lora_config.yaml"
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "checkpoints" / "best_lora_weights.pt"
BPE_VOCAB_PATH = REPO_ROOT / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
SAM3_INPUT_SIZE = 1008


def _resolve_bpe_vocab_path() -> Optional[str]:
    if BPE_VOCAB_PATH.is_file():
        return str(BPE_VOCAB_PATH)
    try:
        from importlib.resources import files

        installed = files("sam3").joinpath("assets", "bpe_simple_vocab_16e6.txt.gz")
        if installed.is_file():
            return str(installed)
    except Exception:
        pass
    return None


def _setup_python_path() -> None:
    """MedSAM3/SAM3 import 경로 설정."""
    sam3_pythonpath = REPO_ROOT / "sam3"
    medsam3_root = REPO_ROOT / "MedSAM3"
    if sam3_pythonpath.is_dir():
        p = str(sam3_pythonpath)
        if p not in sys.path:
            sys.path.insert(0, p)
    if medsam3_root.is_dir():
        p = str(medsam3_root)
        if p not in sys.path:
            sys.path.append(p)


def build_model(config_path: str, weights_path: str, device: str = "cuda"):
    """
    MedSAM3 모델 + LoRA 가중치를 직접 로드합니다.
    (batch_lung_segment.py 의존 없이 이 스크립트 단독으로 동작)
    """
    _setup_python_path()
    from sam3.model_builder import build_sam3_image_model
    from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {dev}")

    print("SAM3 모델 로딩 중...")
    bpe_path = _resolve_bpe_vocab_path()
    model = build_sam3_image_model(
        device=dev.type,
        compile=False,
        load_from_HF=True,
        bpe_path=bpe_path,
        eval_mode=True,
    )

    print("LoRA 적용 중...")
    lora_cfg = config["lora"]
    lora_config = LoRAConfig(
        rank=lora_cfg["rank"],
        alpha=lora_cfg["alpha"],
        dropout=0.0,
        target_modules=lora_cfg["target_modules"],
        apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
        apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
        apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
        apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
        apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
        apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
    )
    model = apply_lora_to_model(model, lora_config)

    print(f"LoRA 가중치 로딩: {weights_path}")
    load_lora_weights(model, weights_path)
    model.to(dev)
    model.eval()

    from sam3.train.transforms.basic_for_api import (
        ComposeAPI,
        RandomResizeAPI,
        ToTensorAPI,
        NormalizeAPI,
    )
    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=SAM3_INPUT_SIZE,
                max_size=SAM3_INPUT_SIZE,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    print("모델 준비 완료!\n")
    return model, transform, dev


@torch.no_grad()
def segment_lung(model, transform, device, pil_image: Image.Image, threshold: float = 0.3, nms_iou: float = 0.5):
    """단일 이미지에 대해 lung 마스크 추론."""
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        Image as SAMImage,
        FindQueryLoaded,
        InferenceMetadata,
    )
    from sam3.train.data.collator import collate_fn_api
    from sam3.model.utils.misc import copy_data_to_device

    w, h = pil_image.size
    sam_image = SAMImage(data=pil_image, objects=[], size=[h, w])
    query = FindQueryLoaded(
        query_text="lung",
        image_id=0,
        object_ids_output=[],
        is_exhaustive=True,
        query_processing_order=0,
        inference_metadata=InferenceMetadata(
            coco_image_id=0,
            original_image_id=0,
            original_category_id=1,
            original_size=[w, h],
            object_id=0,
            frame_index=0,
        ),
    )
    datapoint = Datapoint(find_queries=[query], images=[sam_image])
    datapoint = transform(datapoint)
    batch = collate_fn_api([datapoint], dict_key="input")["input"]
    batch = copy_data_to_device(batch, device, non_blocking=True)

    outputs = model(batch)
    last_output = outputs[-1]
    pred_logits = last_output["pred_logits"]
    pred_boxes = last_output["pred_boxes"]
    pred_masks = last_output.get("pred_masks", None)

    scores = pred_logits.sigmoid()[0].max(dim=-1)[0]
    keep = scores > threshold

    combined_mask = np.zeros((h, w), dtype=bool)
    boxes_result = []
    scores_result = []

    if keep.sum() > 0:
        boxes_cxcywh = pred_boxes[0, keep]
        kept_scores = scores[keep]
        cx, cy, bw, bh = boxes_cxcywh.unbind(-1)
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        keep_nms = nms(boxes_xyxy, kept_scores, nms_iou)
        boxes_xyxy = boxes_xyxy[keep_nms]
        kept_scores = kept_scores[keep_nms]

        if pred_masks is not None:
            masks_small = pred_masks[0, keep][keep_nms].sigmoid() > 0.5
            masks_resized = F.interpolate(
                masks_small.unsqueeze(0).float(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0) > 0.5
            combined_mask = masks_resized.any(dim=0).cpu().numpy()

        boxes_result = boxes_xyxy.cpu().numpy().tolist()
        scores_result = kept_scores.cpu().numpy().tolist()

    return combined_mask, boxes_result, scores_result


def _decode_chexmask_rle(rle_str: str, height: int, width: int) -> Optional[np.ndarray]:
    """
    ChexMask CSV의 RLE 문자열을 bool 마스크 배열로 디코딩합니다.

    RLE 형식: "픽셀인덱스 런길이 픽셀인덱스 런길이 ..."
    - 픽셀인덱스: 0-based 행우선(row-major) 평면 인덱스 (row*W + col)
    - 런길이: 해당 인덱스부터 연속 True 픽셀 수

    반환: shape=(height, width) bool 배열, 실패 시 None
    """
    if not isinstance(rle_str, str) or not rle_str.strip():
        return None
    try:
        tokens = list(map(int, rle_str.split()))
        if len(tokens) % 2 != 0:
            return None
        mask = np.zeros(height * width, dtype=bool)
        for i in range(0, len(tokens), 2):
            start = tokens[i]
            length = tokens[i + 1]
            end = min(start + length, height * width)
            mask[start:end] = True
        return mask.reshape(height, width)
    except Exception as e:
        print(f"⚠️ ChexMask RLE 디코딩 실패: {e}")
        return None


def _load_chexmask_mask(row: dict, target_h: int, target_w: int) -> Optional[np.ndarray]:
    """
    ChexMask CSV 한 행에서 Left+Right Lung 마스크를 합쳐 bool 배열로 반환합니다.

    - CSV의 Height/Width 기준으로 RLE 복원 후 target_h×target_w로 리사이즈
    - 리사이즈는 PIL NEAREST 보간(마스크이므로 보간 불필요)
    - 반환: shape=(target_h, target_w) bool 배열, 실패 시 None
    """
    try:
        H = int(row.get("Height", 0) or 0)
        W = int(row.get("Width", 0) or 0)
        if H <= 0 or W <= 0:
            return None

        combined = np.zeros(H * W, dtype=bool).reshape(H, W)
        for col_name in ("Left Lung", "Right Lung"):
            rle_str = row.get(col_name, "")
            if not isinstance(rle_str, str) or not rle_str.strip():
                continue
            decoded = _decode_chexmask_rle(rle_str, H, W)
            if decoded is not None:
                combined |= decoded

        if not combined.any():
            return None

        if (H, W) != (target_h, target_w):
            pil = Image.fromarray((combined.astype(np.uint8)) * 255)
            pil = pil.resize((target_w, target_h), Image.NEAREST)
            combined = np.array(pil) > 0

        return combined
    except Exception as e:
        print(f"⚠️ ChexMask 마스크 생성 실패: {e}")
        return None


def _load_chexmask_lookup(csv_path: Path) -> dict:
    """
    ChexMask CSV를 읽어 {dicom_id → 행 dict} lookup 테이블을 반환합니다.
    CSV가 수백 MB일 수 있으므로 전체를 한 번에 로드 후 dict으로 캐싱합니다.
    """
    import pandas as pd

    print(f"📥 ChexMask CSV 로드 중: {csv_path}")
    try:
        df = pd.read_csv(str(csv_path), dtype=str)
    except Exception as e:
        raise RuntimeError(f"ChexMask CSV 로드 실패: {csv_path}\n  오류: {e}")

    if "dicom_id" not in df.columns:
        raise ValueError(f"ChexMask CSV에 'dicom_id' 컬럼이 없습니다: {list(df.columns)}")

    lookup: dict = {}
    for _, row in df.iterrows():
        dicom_id = str(row["dicom_id"]).strip()
        if dicom_id:
            lookup[dicom_id] = row.to_dict()

    print(f"✅ ChexMask lookup 완료: {len(lookup):,}개 dicom_id 로드")
    return lookup


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_pa_record(rec: Dict) -> bool:
    view = str(rec.get("view_position") or "").strip().upper()
    return view == "PA"


def _is_target_view_record(rec: Dict, target_views: List[str]) -> bool:
    """view_position 이 target_views(대문자 리스트) 중 하나인지 확인."""
    view = str(rec.get("view_position") or "").strip().upper()
    return view in target_views


def _safe_rel_from_record(rec: Dict) -> str:
    """
    출력 폴더 내부 상대경로를 결정.
    - image_rel_path가 있으면 활용
    - 없으면 dicom_id.jpg fallback
    """
    rel = str(rec.get("image_rel_path") or "").strip()
    if rel:
        return rel
    dicom_id = str(rec.get("dicom_id") or "unknown")
    return f"{dicom_id}.jpg"


def _safe_src_path(rec: Dict, source_image_root: Optional[Path] = None) -> Optional[Path]:
    src = str(rec.get("image_abs_path") or "").strip()
    if src:
        p = Path(src)
        if p.is_file():
            return p
    if source_image_root is not None:
        rel = str(rec.get("image_rel_path") or "").strip()
        if rel:
            candidate = (source_image_root / rel.lstrip("/\\")).resolve()
            if candidate.is_file():
                return candidate
    return None


def _save_masked_image(src_img: Image.Image, mask_bool: np.ndarray, out_path: Path) -> None:
    arr = np.array(src_img.convert("RGB"))
    if arr.shape[:2] != mask_bool.shape:
        raise ValueError(f"shape mismatch: image={arr.shape[:2]}, mask={mask_bool.shape}")
    out = arr.copy()
    out[~mask_bool] = 0
    _ensure_dir(out_path.parent)
    Image.fromarray(out).save(out_path, quality=95)


def _normalize_pneumonia_label(v):
    if v is None:
        return None
    try:
        # 0.0 / 1.0 / numpy scalar 모두 int(0/1)로 정규화
        return int(v)
    except Exception:
        return None


def _pneumonia_is_usable_for_segmentation(v, keep_uncertain: bool = False) -> bool:
    """
    세그멘테이션 대상으로 포함할 pneumonia 라벨인지.
    - null / NaN 제외
    - 기본은 -1 제외
    - --keep-uncertain 이면 CheXpert uncertain(-1)도 포함 (마스크 생성만, 학습 아님)
    - 그 외(0/1 등 정수로 해석 가능한 값)만 포함
    """
    if v is None:
        return False
    try:
        if isinstance(v, float) and np.isnan(v):
            return False
    except Exception:
        pass
    try:
        iv = int(float(v))
    except Exception:
        return False
    if iv == -1:
        return bool(keep_uncertain)
    return True


def _take_first_n(items, n: int):
    if n <= 0:
        return items
    return items[:n]


def _balanced_sample(records: List[Dict], n_per_class: int) -> List[Dict]:
    """
    pneumonia 라벨 기준으로 양성(1)·음성(0) 각 n_per_class 개씩 뽑아 반환.
    실제 가용 샘플이 n_per_class 보다 적으면 있는 것만 전부 사용.
    """
    pos, neg = [], []
    for r in records:
        label = _normalize_pneumonia_label(r.get("pneumonia"))
        if label == 1:
            pos.append(r)
        elif label == 0:
            neg.append(r)

    sampled_pos = pos[:n_per_class]
    sampled_neg = neg[:n_per_class]

    print(f"[INFO] balance-per-class={n_per_class}")
    print(f"[INFO]   pneumonia=1 pool={len(pos):,}  → sampled={len(sampled_pos):,}")
    print(f"[INFO]   pneumonia=0 pool={len(neg):,}  → sampled={len(sampled_neg):,}")

    return sampled_pos + sampled_neg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CXR 폐 세그멘테이션 데이터셋 빌더. "
            "--seg-mode medsam3(기본) 또는 chexmask, "
            "--view PA(기본) / AP / both 선택."
        )
    )
    parser.add_argument(
        "--seg-mode",
        type=str,
        default="medsam3",
        choices=["medsam3", "chexmask"],
        help=(
            "폐 마스크 소스 선택 (기본: medsam3)\n"
            "  medsam3  : MedSAM3 모델 추론으로 폐 마스크 생성\n"
            "  chexmask : ChexMask CSV(RLE)에서 폐 마스크 복원 (모델 불필요)"
        ),
    )
    parser.add_argument(
        "--view",
        type=str,
        default="PA",
        choices=["PA", "AP", "both"],
        help=(
            "처리할 촬영 방향 (기본: PA)\n"
            "  PA   : Posterior-Anterior 영상만 처리\n"
            "  AP   : Anterior-Posterior 영상만 처리\n"
            "  both : PA + AP 모두 처리"
        ),
    )
    parser.add_argument(
        "--chexmask-csv",
        type=str,
        default=None,
        help=f"--seg-mode chexmask 시 사용할 ChexMask CSV 경로 (기본: {DEFAULT_CHEXMASK_CSV})",
    )
    parser.add_argument("--labels-json", type=str, default=str(DEFAULT_LABEL_JSON))
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help=(
            "출력 루트 디렉터리 (미지정 시 seg-mode와 --view에 따라 자동 결정)\n"
            f"  PA+medsam3  기본: {DEFAULT_OUT_ROOT}_pa\n"
            f"  PA+chexmask 기본: {DEFAULT_CHEXMASK_OUT_ROOT}_pa\n"
            "  PA/AP/both 는 위 경로에 _pa / _ap / _pa_ap 접미사 추가"
        ),
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS_PATH))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="0이면 전체 (--balance-per-class와 동시 사용 불가)")
    parser.add_argument(
        "--balance-per-class",
        type=int,
        default=0,
        metavar="N",
        help=(
            "pneumonia 양성(1)·음성(0) 각 N개씩 균형 샘플링. 0이면 비활성(기본). "
            "--limit과 동시에 사용할 수 없음."
        ),
    )
    parser.add_argument(
        "--keep-uncertain",
        action="store_true",
        help="CheXpert Pneumonia=-1(uncertain) 영상도 마스크 생성 대상에 포함합니다. 학습에는 쓰지 않습니다.",
    )
    parser.add_argument("--resume", action="store_true", default=True, help="이미 생성된 샘플 건너뜀")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--source-image-root",
        type=str,
        default=None,
        help=(
            "image_abs_path가 유효하지 않을 때 image_rel_path와 결합할 MIMIC-CXR 루트 "
            "(예: /path/mimic-cxr-jpg/2.1.0)"
        ),
    )
    args = parser.parse_args()

    if args.limit > 0 and args.balance_per_class > 0:
        parser.error("--limit 과 --balance-per-class 는 동시에 사용할 수 없습니다.")

    seg_mode = args.seg_mode
    view_arg = args.view.upper()  # PA / AP / BOTH

    # view 인자 → 실제 필터 집합 및 파일명 태그 결정
    if view_arg == "BOTH":
        target_views: List[str] = ["PA", "AP"]
        view_tag = "pa_ap"
    else:
        target_views = [view_arg]          # ["PA"] 또는 ["AP"]
        view_tag = view_arg.lower()        # "pa" 또는 "ap"

    print(f"[INFO] Segmentation mode: {seg_mode}")
    print(f"[INFO] View filter      : {view_arg} → {target_views}")

    labels_json = Path(args.labels_json).resolve()
    if not labels_json.is_file():
        raise FileNotFoundError(f"labels json 파일이 없습니다: {labels_json}")

    source_image_root = (
        Path(args.source_image_root).resolve()
        if getattr(args, "source_image_root", None)
        else None
    )
    if source_image_root is not None:
        print(f"[INFO] Source image root : {source_image_root}")

    # --out-root 미지정 시 seg-mode + view에 따라 기본값 자동 선택
    if args.out_root is not None:
        out_root = Path(args.out_root).resolve()
    else:
        base_root = DEFAULT_CHEXMASK_OUT_ROOT if seg_mode == "chexmask" else DEFAULT_OUT_ROOT
        if view_arg == "PA":
            out_root = base_root.with_name(base_root.name + "_pa").resolve()
        elif view_arg == "AP":
            out_root = base_root.with_name(base_root.name + "_ap").resolve()
        else:  # both
            out_root = base_root.with_name(base_root.name + "_pa_ap").resolve()

    _ensure_dir(out_root)
    mask_root = out_root / "lung_mask_npy"
    masked_root = out_root / "masked_cxr"
    meta_root = out_root / "meta_json"

    # manifest 파일명: {view_tag}_{seg_mode}
    seg_tag = "chexmask" if seg_mode == "chexmask" else "medsam3"
    manifest_ok_out = out_root / f"manifest_{view_tag}_{seg_tag}_ok.json"
    manifest_err_out = out_root / f"manifest_{view_tag}_{seg_tag}_errors.json"

    with open(labels_json, "r", encoding="utf-8") as f:
        all_records = json.load(f)
    if not isinstance(all_records, list):
        raise RuntimeError("labels json은 list(JSON 배열) 형식이어야 합니다.")

    view_all = [r for r in all_records if isinstance(r, dict) and _is_target_view_record(r, target_views)]
    view_usable_all = [
        r
        for r in view_all
        if _pneumonia_is_usable_for_segmentation(
            r.get("pneumonia"),
            keep_uncertain=bool(args.keep_uncertain),
        )
    ]
    n_view_excluded_pneumonia = len(view_all) - len(view_usable_all)

    if args.balance_per_class > 0:
        pa_records = _balanced_sample(view_usable_all, args.balance_per_class)
    else:
        pa_records = _take_first_n(view_usable_all, int(args.limit) if args.limit and args.limit > 0 else 0)

    print(f"[INFO] input records: {len(all_records):,}")
    print(f"[INFO] {view_arg} records (all): {len(view_all):,}")
    uncertain_note = "include -1" if args.keep_uncertain else "exclude null/-1"
    print(f"[INFO] {view_arg} records (pneumonia usable; {uncertain_note}): {len(view_usable_all):,}")
    if args.limit and args.limit > 0:
        print(f"[INFO] {view_arg} records (after --limit={int(args.limit)}): {len(pa_records):,}")
    if n_view_excluded_pneumonia:
        print(f"[INFO] excluded {view_arg} by pneumonia null/-1: {n_view_excluded_pneumonia:,}")
    print(f"[INFO] final records to process: {len(pa_records):,}")

    if not pa_records:
        print(f"[WARN] 처리할 {view_arg} 샘플이 없습니다 (pneumonia null/-1 제외 후 비었거나 limit 결과). 종료합니다.")
        with open(manifest_ok_out, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        with open(manifest_err_out, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return

    # ── 모드별 초기화 ─────────────────────────────────────────────────────────
    model, transform, device = None, None, None
    chexmask_lookup: dict = {}

    if seg_mode == "chexmask":
        chexmask_csv_path = Path(args.chexmask_csv).resolve() if args.chexmask_csv else DEFAULT_CHEXMASK_CSV
        chexmask_lookup = _load_chexmask_lookup(chexmask_csv_path)
    else:
        print("[INFO] MedSAM3 모델 로딩...")
        model, transform, device = build_model(
            str(Path(args.config).resolve()),
            str(Path(args.weights).resolve()),
            device=str(args.device),
        )
    # ─────────────────────────────────────────────────────────────────────────

    out_manifest_ok: List[Dict] = []
    out_manifest_err: List[Dict] = []
    n_ok, n_skip_resume, n_skip_missing, n_skip_no_mask, n_err = 0, 0, 0, 0, 0
    FLUSH_INTERVAL = 100  # 100개 처리마다 manifest 중간 저장

    desc = f"{view_arg} lung segmentation [{seg_mode}]"
    for i, rec in enumerate(tqdm(pa_records, desc=desc)):
        rel_jpg = _safe_rel_from_record(rec)
        rel_path = Path(rel_jpg)
        stem = rel_path.stem
        dicom_id = str(rec.get("dicom_id") or "")

        src_path = _safe_src_path(rec, source_image_root=source_image_root)
        if src_path is None:
            n_skip_missing += 1
            continue

        out_mask = (mask_root / rel_path).with_name(f"{stem}_lung_mask.npy")
        out_masked = masked_root / rel_path
        out_meta = (meta_root / rel_path).with_suffix(".json")

        if args.resume and out_mask.is_file() and out_masked.is_file() and out_meta.is_file():
            n_skip_resume += 1
            try:
                with open(out_meta, "r", encoding="utf-8") as _f:
                    out_manifest_ok.append(json.load(_f))
            except Exception:
                pass
            continue

        try:
            src_img = Image.open(src_path).convert("RGB")
            cxr_w, cxr_h = src_img.size

            # ── 마스크 취득: seg-mode 분기 ────────────────────────────────
            if seg_mode == "chexmask":
                if dicom_id not in chexmask_lookup:
                    n_skip_no_mask += 1
                    continue
                mask_bool = _load_chexmask_mask(
                    chexmask_lookup[dicom_id],
                    target_h=cxr_h,
                    target_w=cxr_w,
                )
                if mask_bool is None:
                    n_skip_no_mask += 1
                    continue
                boxes: list = []
                scores: list = []
                chexmask_row = chexmask_lookup[dicom_id]
                dice_mean = chexmask_row.get("Dice RCA (Mean)", None)
                dice_max = chexmask_row.get("Dice RCA (Max)", None)
            else:
                mask_bool, boxes, scores = segment_lung(
                    model,
                    transform,
                    device,
                    src_img,
                    threshold=float(args.threshold),
                    nms_iou=float(args.nms_iou),
                )
                dice_mean, dice_max = None, None
            # ─────────────────────────────────────────────────────────────

            _ensure_dir(out_mask.parent)
            np.save(out_mask, mask_bool.astype(bool))
            _save_masked_image(src_img, mask_bool.astype(bool), out_masked)

            lung_pixels = int(mask_bool.sum())
            total_pixels = int(mask_bool.size)
            meta: Dict = {
                "source_image_abs_path": str(src_path),
                "source_image_rel_path": str(rec.get("image_rel_path") or ""),
                "subject_id": rec.get("subject_id"),
                "study_id": rec.get("study_id"),
                "dicom_id": dicom_id,
                "view_position": rec.get("view_position"),
                "pneumonia": _normalize_pneumonia_label(rec.get("pneumonia")),
                "lung_mask_npy": str(out_mask),
                "masked_cxr_jpg": str(out_masked),
                "lung_pixel_count": lung_pixels,
                "total_pixel_count": total_pixels,
                "lung_area_ratio": float(lung_pixels / total_pixels) if total_pixels > 0 else 0.0,
                "seg_mode": seg_mode,
            }
            if seg_mode == "chexmask":
                meta["chexmask_dice_mean"] = float(dice_mean) if dice_mean is not None else None
                meta["chexmask_dice_max"] = float(dice_max) if dice_max is not None else None
            else:
                meta["num_detections"] = int(len(scores))
                meta["scores"] = [float(s) for s in scores]
                meta["boxes_xyxy"] = boxes.tolist() if hasattr(boxes, "tolist") else boxes

            _ensure_dir(out_meta.parent)
            with open(out_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            out_manifest_ok.append(meta)
            n_ok += 1
        except Exception as e:
            n_err += 1
            out_manifest_err.append(
                {
                    "source_image_abs_path": str(src_path),
                    "source_image_rel_path": str(rec.get("image_rel_path") or ""),
                    "subject_id": rec.get("subject_id"),
                    "study_id": rec.get("study_id"),
                    "dicom_id": dicom_id,
                    "view_position": rec.get("view_position"),
                    "pneumonia": _normalize_pneumonia_label(rec.get("pneumonia")),
                    "seg_mode": seg_mode,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_repr": repr(e),
                }
            )

        # 중간 저장: 프로세스가 중단되어도 진행 상황 보존
        if (i + 1) % FLUSH_INTERVAL == 0:
            with open(manifest_ok_out, "w", encoding="utf-8") as f:
                json.dump(out_manifest_ok, f, ensure_ascii=False, indent=2)
            with open(manifest_err_out, "w", encoding="utf-8") as f:
                json.dump(out_manifest_err, f, ensure_ascii=False, indent=2)

    with open(manifest_ok_out, "w", encoding="utf-8") as f:
        json.dump(out_manifest_ok, f, ensure_ascii=False, indent=2)
    with open(manifest_err_out, "w", encoding="utf-8") as f:
        json.dump(out_manifest_err, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] 폐 세그멘테이션 데이터셋 생성 완료")
    print(f"  - seg_mode       : {seg_mode}")
    print(f"  - view           : {view_arg}")
    print(f"  - out_root       : {out_root}")
    print(f"  - success        : {n_ok:,}")
    print(f"  - skipped(resume): {n_skip_resume:,}")
    print(f"  - skipped(no src): {n_skip_missing:,}")
    if seg_mode == "chexmask":
        print(f"  - skipped(no chexmask): {n_skip_no_mask:,}")
    print(f"  - errors         : {n_err:,}")
    print(f"  - manifest_ok    : {manifest_ok_out}")
    print(f"  - manifest_err   : {manifest_err_out}")


if __name__ == "__main__":
    main()
