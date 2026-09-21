#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
폐 마스크를 이용해 Pneumonia 분류용 CXR을 크롭하는 스크립트

--seg-mode 옵션으로 두 가지 마스크 소스를 선택할 수 있습니다:
  medsam3  (기본): MedSAM3가 생성한 .npy 마스크 파일을 사용
  chexmask        : ChexMask CSV(RLE 인코딩)에서 폐 마스크를 복원하여 사용

--view 옵션으로 처리할 촬영 방향을 선택합니다:
  PA   (기본): Posterior-Anterior 영상만 처리
  AP          : Anterior-Posterior 영상만 처리
  both        : PA + AP 모두 처리

처리 순서 (두 모드 공통):
1. 원본 CXR 이미지 로드
2. 폐 마스크(PIL Image)에서 폐 영역 바운딩박스 계산
3. 바운딩박스로 이미지 크롭 (상/하/좌/우 여백 추가)
4. 검은 테두리 제거 (trim)
5. 최종 이미지 저장

아이디어 (아주 단순 버전으로 설명):
- 각 X-ray 이미지마다 "폐가 있는 위치"를 표시한 마스크가 있습니다.
  · medsam3 모드: .npy 파일(True/False 배열)
  · chexmask 모드: CSV의 RLE 숫자열 → Left/Right 폐를 합쳐서 하나의 마스크 생성
- 이 마스크에서 True(=폐 영역) 픽셀만 골라서, 그 부분을 딱 둘러싸는 네모(바운딩박스)를 찾습니다.
- 그 바운딩박스를 상/하/좌/우로 일정 픽셀만큼 더 넓혀서 이미지를 잘라서 저장합니다.

입력 (medsam3 + PA, 기본):
- cxr_medsam3_lung_seg_pa/manifest_pa_medsam3_ok.json
  - 안에 source_image_abs_path, lung_mask_npy, pneumonia 라벨 등이 들어 있음
- cxr_medsam3_lung_seg_pa/lung_mask_npy/files/.../{dicom_id}_lung_mask.npy

입력 (chexmask 모드):
- manifest: cxr_chexmask_lung_seg_{view_tag}/manifest_{view_tag}_chexmask_ok.json
- ChexMask_MIMIC-CXR-JPG.csv (dicom_id별 Left Lung / Right Lung RLE 컬럼 포함)

출력 (medsam3 + PA, 기본):
- out-root (기본: <DATA_BASE_DIR>/cxr_medsam3_lung_seg_cropped_pa)
  └── files/pXX/pXXXXXXXX/sXXXXXXXX/
        {dicom_id}.jpg              (크롭+trim된 CXR)
        {dicom_id}_orig.jpg         (크롭 전 원본 CXR, 마스크·bbox 오버레이)
  └── manifest_pa_medsam3_ok.json

출력 (chexmask + AP 예시):
- out-root (기본: <DATA_BASE_DIR>/cxr_chexmask_lung_seg_cropped_ap)
  └── 동일 폴더 구조
  └── manifest_ap_chexmask_ok.json

주의:
- medsam3 마스크(.npy)는 이미 CXR과 동일 크기이므로 별도 리사이즈 불필요.
- chexmask RLE 마스크는 CSV의 Height/Width 기준으로 복원 후 원본 CXR 크기에 맞게 리사이즈합니다.
- *_orig.jpg 파일은 크롭/trim 전 원본에 마스크·bbox를 덧씌운 디버그 이미지입니다.

사용 예시:

# [medsam3] PA 기본 크롭
> python -m pip install -r requirements.txt
> python preprocessing/build_cxr_lung_crop_dataset.py --seg-mode medsam3

# [chexmask] AP 크롭
> python preprocessing/build_cxr_lung_crop_dataset.py --seg-mode chexmask --view AP

# [chexmask] PA + AP 모두 크롭
> python preprocessing/build_cxr_lung_crop_dataset.py --seg-mode chexmask --view both

# [chexmask] CSV 경로 직접 지정
> python preprocessing/build_cxr_lung_crop_dataset.py --seg-mode chexmask --chexmask-csv /path/to/ChexMask_MIMIC-CXR-JPG.csv

# 처음부터 전체 재처리
> python preprocessing/build_cxr_lung_crop_dataset.py --seg-mode chexmask --no-resume

pad 값(상하좌우 여유 픽셀 수)은 --pad 옵션으로 자유롭게 바꿀 수 있습니다.
"""

import os
import json
import argparse
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile, ImageDraw, ImageOps, ImageFont
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def _normalize_frac(v: Optional[float]) -> Optional[float]:
    """
    비율 입력을 정규화합니다.
    - None -> None
    - 0.05 -> 0.05 (5%%)
    - 5.0  -> 0.05 (5%% 라고 입력한 경우를 허용)
    - 0 또는 0.0 -> None (비율 모드 비활성화로 해석)
    """
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if f < 0:
        return None
    if f == 0.0:
        return None
    if f > 1.0:
        f = f / 100.0
    return float(f)


def _resolve_pad_px(pad_px: Optional[int], pad_frac: Optional[float], dim: int) -> int:
    """
    pad를 픽셀 단위로 결정합니다.
    - pad_frac가 있으면 dim(가로/세로)에 대한 비율로 환산
    - 없으면 pad_px(픽셀)를 그대로 사용
    """
    if pad_frac is not None:
        f = _normalize_frac(pad_frac)
        if f is not None:
            return max(0, int(round(float(f) * float(max(1, int(dim))))))
    try:
        return max(0, int(pad_px or 0))
    except Exception:
        return 0


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DATA_BASE_DIR = os.path.abspath(os.environ.get("DATA_BASE_DIR", PACKAGE_ROOT))

# cxr_medsam3_lung_seg 폴더: MedSAM3로 생성한 폐 마스크(.npy) + manifest 위치
DEFAULT_SEG_ROOT = os.path.join(DATA_BASE_DIR, "cxr_medsam3_lung_seg")
DEFAULT_MANIFEST = os.path.join(DEFAULT_SEG_ROOT, "manifest_pa_medsam3_ok.json")
DEFAULT_OUT_ROOT = os.path.join(DATA_BASE_DIR, "cxr_medsam3_lung_seg_cropped")

# ChexMask 모드용 기본 경로
DEFAULT_CHEXMASK_CSV = os.environ.get(
    "CHEXMASK_CSV",
    os.path.join(PACKAGE_ROOT, "data", "ChexMask_MIMIC-CXR-JPG.csv"),
)
DEFAULT_CHEXMASK_SEG_ROOT = os.path.join(DATA_BASE_DIR, "cxr_chexmask_lung_seg")
DEFAULT_CHEXMASK_OUT_ROOT = os.path.join(DATA_BASE_DIR, "cxr_chexmask_lung_seg_cropped")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def decode_chexmask_rle(rle_str: str, height: int, width: int) -> Optional[np.ndarray]:
    """
    ChexMask CSV의 RLE(Run-Length Encoding) 문자열을 bool 마스크 배열로 디코딩합니다.

    ChexMask RLE 형식:
    - "픽셀인덱스1 런길이1 픽셀인덱스2 런길이2 ..."
    - 픽셀인덱스: 0부터 시작하는 행-우선(row-major) 평면 인덱스 (즉, row*W + col)
    - 런길이: 해당 인덱스부터 연속으로 True인 픽셀 수

    반환: shape=(height, width)인 bool 배열, 실패 시 None
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


def load_chexmask_as_pil(
    row: dict,
    target_size: Optional[Tuple[int, int]] = None,
) -> Optional[Image.Image]:
    """
    ChexMask CSV 한 행(row)에서 Left Lung + Right Lung 마스크를 합쳐 PIL Image로 반환합니다.

    - Left Lung, Right Lung 각각의 RLE를 디코딩하여 OR 합산
    - CSV의 Height/Width로 복원 후, target_size(W, H)가 주어지면 원본 CXR 크기에 맞게 리사이즈
    - 리사이즈는 PIL NEAREST 보간 사용 (마스크이므로 보간 불필요)

    반환: PIL Image (L 모드, 255=폐 영역 / 0=배경), 실패 시 None
    """
    try:
        H = int(row.get("Height", 0))
        W = int(row.get("Width", 0))
        if H <= 0 or W <= 0:
            return None

        combined = np.zeros(H * W, dtype=bool).reshape(H, W)

        for col_name in ("Left Lung", "Right Lung"):
            rle_str = row.get(col_name, "")
            if not isinstance(rle_str, str) or not rle_str.strip():
                continue
            decoded = decode_chexmask_rle(rle_str, H, W)
            if decoded is not None:
                combined |= decoded

        if not combined.any():
            return None

        mask_pil = Image.fromarray((combined.astype(np.uint8)) * 255)

        if target_size is not None:
            tw, th = target_size
            if (tw, th) != (W, H):
                mask_pil = mask_pil.resize((tw, th), Image.NEAREST)

        return mask_pil
    except Exception as e:
        print(f"⚠️ ChexMask 마스크 생성 실패: {e}")
        return None


def load_chexmask_lookup(csv_path: str) -> Dict[str, dict]:
    """
    ChexMask CSV를 읽어 {dicom_id → 행 dict} 형태의 lookup 테이블을 반환합니다.

    lookup 테이블 구조:
    {
        "f4a185f1-db2de1fd-...": {
            "Height": 3056,
            "Width": 2544,
            "Left Lung": "1110713 9 ...",
            "Right Lung": "1102454 5 ...",
            ...
        },
        ...
    }

    주의: CSV가 수백 MB일 수 있으므로 최초 1회 로드 후 dict으로 캐싱해서 사용하세요.
    """
    import pandas as pd

    print(f"📥 ChexMask CSV 로드 중: {csv_path}")
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        raise RuntimeError(f"ChexMask CSV 로드 실패: {csv_path}\n  오류: {e}")

    if "dicom_id" not in df.columns:
        raise ValueError(f"ChexMask CSV에 'dicom_id' 컬럼이 없습니다: {list(df.columns)}")

    lookup: Dict[str, dict] = {}
    for _, row in df.iterrows():
        dicom_id = str(row["dicom_id"]).strip()
        if dicom_id:
            lookup[dicom_id] = row.to_dict()

    print(f"✅ ChexMask lookup 완료: {len(lookup):,}개 dicom_id 로드")
    return lookup


def load_npy_mask_as_pil(npy_path: str) -> Optional[Image.Image]:
    """
    medsam3가 생성한 폐 마스크 .npy 파일(bool 배열)을 PIL Image(L 모드)로 변환합니다.

    .npy 파일 안에는 True/False로 구성된 2D 배열이 들어 있습니다.
    - True  → 255 (폐 영역)
    - False → 0   (배경)

    반환: PIL Image (L 모드), 실패 시 None
    """
    try:
        arr = np.load(npy_path)
        if arr.ndim == 3:
            # 혹시 (H, W, C) 형태라면 첫 채널만 사용
            arr = arr[..., 0]
        mask_uint8 = (arr.astype(bool).astype(np.uint8)) * 255
        return Image.fromarray(mask_uint8)
    except Exception as e:
        print(f"⚠️ .npy 마스크 로드 실패 ({npy_path}): {e}")
        return None


def _is_valid_image_file(path: str) -> bool:
    """
    파일이 존재하고, PIL로 열 수 있으며 유효한 크기의 이미지인지 확인합니다.
    생성 도중 끊겨서 깨진(truncated) 파일은 False를 반환합니다.
    """
    if not path or not os.path.isfile(path):
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            w, h = img.size
        return w >= 1 and h >= 1
    except Exception:
        return False


def filter_mask_small_components(
    mask_img: Image.Image,
    keep_top_k: int = 2,
    min_area: int = 20,
) -> Image.Image:
    """
    마스크에서 '아주 작은 점/조각' 같은 노이즈를 제거합니다.

    방식:
    - 4-이웃 연결요소(connected component)로 덩어리를 찾고
    - 면적이 큰 덩어리 상위 keep_top_k개만 남깁니다.
    - 단, min_area(픽셀)보다 작은 덩어리는 무조건 제거합니다.

    scipy.ndimage.label을 사용해 C 레벨에서 처리하므로
    대형 마스크에서도 안정적으로 동작합니다.
    """
    from scipy import ndimage

    m = np.array(mask_img.convert("L")) > 0
    h, w = m.shape
    if h == 0 or w == 0:
        return mask_img

    struct4 = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=bool)
    labeled, n_comps = ndimage.label(m, structure=struct4)

    if n_comps == 0:
        return mask_img

    # 각 레이블의 면적을 계산하고 면적 순으로 정렬
    areas = [(int((labeled == i).sum()), i) for i in range(1, n_comps + 1)]
    areas.sort(reverse=True)

    keep_top_k = max(1, int(keep_top_k))
    min_area = max(1, int(min_area))

    out = np.zeros((h, w), dtype=bool)
    kept = 0
    for area, label_id in areas:
        if kept >= keep_top_k:
            break
        if area < min_area:
            continue
        out[labeled == label_id] = True
        kept += 1

    out_img = Image.fromarray((out.astype(np.uint8) * 255))
    return out_img


def _connected_components_4(mask_bool: np.ndarray) -> list:
    """
    4-이웃 연결요소를 찾아 각 컴포넌트의 통계를 반환합니다.
    반환 원소: {"area", "xmin", "xmax", "ymin", "ymax", "cx"}

    scipy.ndimage.label을 사용해 C 레벨에서 처리하므로
    수백만 픽셀 규모의 대형 마스크도 안정적으로 처리합니다.
    """
    from scipy import ndimage

    # 4-connectivity: 상하좌우만 이웃으로 봄 (대각선 제외)
    struct4 = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=bool)
    labeled, n_comps = ndimage.label(mask_bool, structure=struct4)

    comps = []
    for label_id in range(1, n_comps + 1):
        ys, xs = np.where(labeled == label_id)
        if ys.size == 0:
            continue
        comps.append(
            {
                "area": int(ys.size),
                "xmin": int(xs.min()),
                "xmax": int(xs.max()),
                "ymin": int(ys.min()),
                "ymax": int(ys.max()),
                "cx": float(xs.mean()),
            }
        )
    return comps


## NOTE:
# 논문용 파이프라인 단순화/방어력 강화를 위해 spatial=mask 모드는 제거했습니다.
# 따라서 mask 확장/브릿지/홀필/픽셀 마스킹 관련 유틸은 더 이상 사용하지 않습니다.


def check_mask_quality(mask_img: Image.Image, img_size: Tuple[int, int], min_area_ratio: float = 0.10) -> Tuple[bool, str]:
    """
    마스크 품질을 검사하여 "빈 마스크" 또는 "너무 작은 폐 마스크"를 간단히 감지합니다.

    Returns:
        (is_good, reason)
        - is_good=False 인 경우 reason 예: empty_mask, mask_size_mismatch, too_small_ratio_0.034
    """
    try:
        mask_arr = (np.array(mask_img.convert("L")) > 0)
    except Exception:
        return False, "mask_read_error"

    h, w = mask_arr.shape
    exp_w, exp_h = int(img_size[0]), int(img_size[1])
    if (w != exp_w) or (h != exp_h):
        # 이 스크립트에서는 mask를 img_orig 크기로 resize 한 후 검사하는 것이 정상입니다.
        return False, f"mask_size_mismatch_{w}x{h}_vs_{exp_w}x{exp_h}"

    # 1) 텅 빈 마스크
    mask_area = int(mask_arr.sum())
    if mask_area == 0:
        return False, "empty_mask"

    # 2) 면적 비율 체크(너무 작으면 segmentation 실패일 확률 ↑)
    img_area = int(w * h)
    ratio = float(mask_area / max(img_area, 1))
    if ratio < float(min_area_ratio):
        return False, f"too_small_ratio_{ratio:.3f}"

    return True, "ok"


def _center_crop_bbox(img_size: Tuple[int, int], crop_ratio: float = 0.8, full_height: bool = False) -> Tuple[int, int, int, int]:
    """안전한 fallback 용 중앙 크롭 bbox를 생성합니다. (left, top, right, bottom)"""
    w, h = int(img_size[0]), int(img_size[1])
    r = float(crop_ratio)
    r = max(0.1, min(1.0, r))
    crop_w = int(w * r)
    crop_h = int(h * r)
    left = max(0, (w - crop_w) // 2)
    top = 0 if full_height else max(0, (h - crop_h) // 2)
    right = min(w, left + crop_w)
    bottom = h if full_height else min(h, top + crop_h)
    # 최소 1px 보장
    right = max(left + 1, right)
    bottom = max(top + 1, bottom)
    return (left, top, right, bottom)


def compute_bbox_from_mask(
    mask_img: Image.Image,
    pad_top: int = 32,
    pad_bottom: int = 32,
    pad_lr: int = 32,
    bbox_mode: str = "auto",  # "mask" | "auto"
    bbox_min_area_ratio: float = 0.35,
    bbox_min_width_ratio: float = 0.50,
    bbox_min_height_ratio: float = 0.55,
    single_comp_wide_threshold: float = 0.60,
    tall_bbox_aspect_threshold: float = 1.5,
    tall_bbox_min_width_frac: float = 0.5,
) -> Optional[Tuple[int, int, int, int]]:
    """
    마스크 이미지에서 0이 아닌 픽셀을 모두 찾고, 그 부분을 감싸는 바운딩박스를 계산합니다.

    마스크 bbox를 기준으로 위/아래/좌/우를 크롭합니다.
    - 위쪽(upper): 폐가 시작하는 지점보다 pad_top 만큼 위로 확장
    - 아래쪽(lower): 폐가 끝나는 지점보다 pad_bottom 만큼 아래로 확장
    - 좌/우(left/right): 마스크 기준으로 pad_lr 만큼 확장

    반환값은 (left, upper, right, lower) 형식이며,
    - left, upper: 포함
    - right, lower: PIL 규칙에 따라 'exclusive' (한 칸 뒤 픽셀까지 자름)
    """
    # 마스크는 흑백(L)로 변환해서 사용
    m = mask_img.convert("L")
    arr = np.array(m)

    mask_bool = arr > 0

    # 0이 아닌 픽셀만 선택 (즉, 마스크가 켜져 있는 부분)
    ys, xs = np.where(mask_bool)
    if ys.size == 0 or xs.size == 0:
        # 마스크가 전부 0이면 바운딩박스를 만들 수 없음
        return None

    y_min_raw, y_max_raw = int(ys.min()), int(ys.max())
    x_min_raw, x_max_raw = xs.min(), xs.max()

    h, w = arr.shape

    # 기본: 마스크 기반 좌/우 경계 (auto 확장 대상)
    x_min_use = int(x_min_raw)
    x_max_use = int(x_max_raw)

    # 옵션: 조건부 "큰 폐 기준 대칭 확장"
    # - 폐가 1개만 잡히거나
    # - 좌/우 폐 크기(면적/너비) 차이가 일정 이상이면
    #   큰 폐를 기준으로 작은 쪽을 대칭으로 더 열어줍니다.
    if str(bbox_mode).lower() == "auto":
        # NOTE: 연결요소 분석은 "원본 마스크"를 사용 (마스크 확장으로 인한 임의 merge를 최소화)
        comps = _connected_components_4(mask_bool)
        comps.sort(key=lambda c: c["area"], reverse=True)
        comps = comps[:2]

        # 연결요소가 2개면 좌/우로 분리, 1개면 중앙 기준으로 대칭 확장 후보
        if len(comps) == 2:
            # left/right 결정 (중심 x)
            c1, c2 = comps[0], comps[1]
            left = c1 if c1["cx"] <= c2["cx"] else c2
            right = c2 if left is c1 else c1

            left_area, right_area = left["area"], right["area"]
            left_w = left["xmax"] - left["xmin"] + 1
            right_w = right["xmax"] - right["xmin"] + 1
            left_h = left["ymax"] - left["ymin"] + 1
            right_h = right["ymax"] - right["ymin"] + 1

            small_area = min(left_area, right_area)
            large_area = max(left_area, right_area)
            small_w = min(left_w, right_w)
            large_w = max(left_w, right_w)
            small_h = min(left_h, right_h)
            large_h = max(left_h, right_h)

            area_ratio = small_area / max(large_area, 1)
            width_ratio = small_w / max(large_w, 1)
            height_ratio = small_h / max(large_h, 1)

            # "척추(중앙선)"을 두 폐 사이로 가정: left.xmax 와 right.xmin 사이의 중간
            if right["xmin"] > left["xmax"]:
                spine = int((left["xmax"] + right["xmin"]) // 2)
            else:
                # 두 폐가 붙어 있거나 겹치면(분리 실패) 이미지 중앙 fallback
                spine = w // 2

            if (
                (area_ratio < float(bbox_min_area_ratio))
                or (width_ratio < float(bbox_min_width_ratio))
                or (height_ratio < float(bbox_min_height_ratio))
            ):
                # 큰 쪽 폭을 기준으로 양쪽을 대칭으로 열기
                left_width = max(0, spine - int(left["xmin"]))
                right_width = max(0, int(right["xmax"]) - spine)
                max_width = max(left_width, right_width)
                x_min_use = max(0, spine - max_width)
                x_max_use = min(w - 1, spine + max_width)
            else:
                # 정상(균형)이라면 순수 마스크 bbox 사용
                x_min_use = int(x_min_raw)
                x_max_use = int(x_max_raw)
        elif len(comps) == 1:
            # 연결요소가 1개인 경우:
            # - 진짜 한쪽 폐만 잡힌(single lung) 케이스일 수도 있고
            # - 양쪽 폐가 마스크 오류로 얇게 붙어서 1개로 보이는(merged) 케이스일 수도 있습니다.
            # 너무 넓은(예: 전체 너비의 60% 이상) 덩어리는 merged 가능성이 높으므로 대칭 확장을 금지합니다.
            comp_w = int(x_max_raw - x_min_raw + 1)
            width_frac = float(comp_w / max(w, 1))
            if width_frac >= float(single_comp_wide_threshold):
                # 확장 금지: 순수 마스크 bbox 사용
                x_min_use = int(x_min_raw)
                x_max_use = int(x_max_raw)
            else:
                # 좁으면(single lung 가능성↑) 중앙(spine) 기준 대칭 확장
                spine = w // 2
                left_width = max(0, spine - int(x_min_raw))
                right_width = max(0, int(x_max_raw) - spine)
                max_width = max(left_width, right_width)
                x_min_use = max(0, spine - max_width)
                x_max_use = min(w - 1, spine + max_width)

        # 연결요소 2개인데 세로가 가로보다 압도적으로 길면 (한쪽 폐가 위/아래로 쪼개진 경우) 가로로 넓히기
        # 똑같은 크기의 박스를 없는 쪽으로만 붙임: 박스가 왼쪽에 치우치면 오른쪽으로만, 오른쪽에 치우치면 왼쪽으로만 확장
        if len(comps) == 2:
            box_h = int(y_max_raw - y_min_raw) + 1
            box_w_cur = int(x_max_use - x_min_use) + 1
            if box_w_cur > 0 and box_h > tall_bbox_aspect_threshold * box_w_cur:
                add_width = box_w_cur
                cx = (int(x_min_use) + int(x_max_use)) // 2
                if cx < w // 2:
                    # 박스가 왼쪽에 치우침 → 없는 쪽(오른쪽)으로만 확장
                    x_max_use = min(w - 1, int(x_max_use) + add_width)
                else:
                    # 박스가 오른쪽에 치우침 → 없는 쪽(왼쪽)으로만 확장
                    x_min_use = max(0, int(x_min_use) - add_width)

    # bottom은 "기존 방식 그대로": y_max_raw + pad_bottom
    y_max = min(h - 1, int(y_max_raw) + int(pad_bottom))

    # 기본 box 방식: bbox에 pad를 더함
    y_min = max(0, int(y_min_raw) - int(pad_top))
    x_min = max(0, int(x_min_use) - int(pad_lr))
    x_max = min(w - 1, int(x_max_use) + int(pad_lr))

    # PIL.Image.crop 은 (left, upper, right, lower) 에서
    # right, lower 는 "포함되지 않는" 인덱스이므로
    # - right: x_max + 1
    # - lower: y_max + 1 (exclusive)
    left = int(x_min)
    upper = int(y_min)
    right = int(x_max) + 1
    lower = int(y_max) + 1
    return left, upper, right, lower


def trim_zero_border_box(
    img: Image.Image, threshold: int = 0
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Return the trimmed image and the exclusive (left, upper, right, lower) box
    in the pre-trim crop coordinates."""
    g = img.convert("L")
    arr = np.array(g)
    mask = arr > threshold
    w, h = img.size
    if not mask.any():
        return img, (0, 0, int(w), int(h))

    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    y_min = int(rows.argmax())
    y_max = int(len(rows) - 1 - rows[::-1].argmax())
    x_min = int(cols.argmax())
    x_max = int(len(cols) - 1 - cols[::-1].argmax())
    box = (x_min, y_min, x_max + 1, y_max + 1)
    return img.crop(box), box


def trim_zero_border(img: Image.Image, threshold: int = 0) -> Image.Image:
    """
    이미지의 맨 바깥 테두리(상/하/좌/우)를 기준으로,
    픽셀 값이 전부 0(또는 threshold 이하)인 줄/열은 잘라냅니다.

    - 중요한 점: 한 줄이라도 0보다 큰 픽셀(=실제 내용)이 있으면 그 줄은 남깁니다.
    - 따라서 "내용이 있는 부분"은 절대 잘리지 않고,
      완전히 검은색(0)인 외곽 여백만 제거됩니다.
    """
    trimmed, _ = trim_zero_border_box(img, threshold=threshold)
    return trimmed


def square_padding_ltrb(size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Padding (left, top, right, bottom) added by pad_to_square."""
    w, h = int(size[0]), int(size[1])
    if w == h:
        return (0, 0, 0, 0)
    if w > h:
        diff = w - h
        top = diff // 2
        return (0, top, 0, diff - top)
    diff = h - w
    left = diff // 2
    return (left, 0, diff - left, 0)


def pad_to_square(img: Image.Image, fill=0) -> Image.Image:
    """
    이미지를 정사각형으로 만들기 위해, 짧은 변 방향으로 양쪽에 검은 여백을 추가합니다.
    - 원본 내용은 그대로 유지됩니다(늘리거나 줄이지 않음).
    """
    w, h = img.size
    if w == h:
        return img

    # PIL.ImageOps.expand: border=(left, top, right, bottom)
    if w > h:
        diff = w - h
        top = diff // 2
        bottom = diff - top
        border = (0, top, 0, bottom)
    else:
        diff = h - w
        left = diff // 2
        right = diff - left
        border = (left, 0, right, 0)

    # fill은 모드에 따라 튜플이 필요할 수 있음 (RGB 등)
    if img.mode in ("RGB", "RGBA") and not isinstance(fill, tuple):
        fill = (fill, fill, fill) if img.mode == "RGB" else (fill, fill, fill, 255)

    return ImageOps.expand(img, border=border, fill=fill)


def overlay_mask_on_image(
    base_img: Image.Image,
    mask_img: Image.Image,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    bbox_color: Tuple[int, int, int] = (255, 255, 0),
    bbox_width: int = 5,
    mask_bbox: Optional[Tuple[int, int, int, int]] = None,
    mask_bbox_color: Tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """
    base_img 위에 mask_img(0이 아닌 픽셀)를 반투명 컬러로 덧씌운 overlay 이미지를 만듭니다.
    - base_img, mask_img 크기가 다르면 mask_img를 base_img에 맞춰 NEAREST로 리사이즈합니다.
    - bbox: 최종 크롭 bbox (기본: 노란색, fallback bbox는 파란색으로도 사용 가능)
    - mask_bbox: 원본 마스크 기반 bbox (빨간색) - 대칭 확장 전 비교용
    """
    if base_img.size != mask_img.size:
        mask_img = mask_img.resize(base_img.size, Image.NEAREST)

    base_rgba = base_img.convert("RGBA")
    m = np.array(mask_img.convert("L")) > 0
    h, w = m.shape

    a = int(max(0.0, min(1.0, float(alpha))) * 255)
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    overlay[..., 3] = 0
    overlay[m, 3] = a

    overlay_img = Image.fromarray(overlay)
    out = Image.alpha_composite(base_rgba, overlay_img).convert("RGB")

    draw = ImageDraw.Draw(out)

    # 원본 마스크 bbox (빨간색, 점선처럼 얇게) - 대칭 확장 전
    if mask_bbox is not None:
        try:
            left, upper, right, lower = [int(v) for v in mask_bbox]
            right = max(left + 1, right - 1)
            lower = max(upper + 1, lower - 1)
            draw.rectangle([left, upper, right, lower], outline=mask_bbox_color, width=3)
        except Exception:
            pass

    # 최종 크롭 bbox (노란색, 두껍게) - 대칭 확장 후
    if bbox is not None:
        try:
            left, upper, right, lower = [int(v) for v in bbox]
            # bbox는 (left, upper, right, lower)이며 right/lower는 exclusive 규칙.
            # 시각화는 inclusive에 가깝게 보여주기 위해 -1 보정
            right = max(left + 1, right - 1)
            lower = max(upper + 1, lower - 1)
            draw.rectangle([left, upper, right, lower], outline=bbox_color, width=int(bbox_width))
        except Exception:
            pass

    return out


def _pick_bbox_color(bbox_info: Optional[Dict[str, object]]) -> Tuple[int, int, int]:
    """
    bbox 색상을 결정합니다.
    - 마스크 기반 크롭: 노란색
    - 마스크 불량 감지 fallback 크롭: 파란색
    """
    reason = str((bbox_info or {}).get("reason") or "").lower()
    if reason.startswith("fallback_center_") or ("fallback_center" in reason):
        return (0, 120, 255)  # blue-ish
    return (255, 255, 0)      # yellow


def draw_bbox_info_label(
    img_rgb: Image.Image,
    bbox_mode: str,
    bbox_info: Optional[Dict[str, object]],
    font_size: int = 24,
) -> Image.Image:
    """좌상단에 bbox 정보(대칭확장 여부 포함)를 텍스트로 표시합니다."""
    out = img_rgb.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=int(font_size))
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", size=int(font_size))
        except Exception:
            font = ImageFont.load_default()

    symmetric = bool((bbox_info or {}).get("symmetric", False))
    tall_narrow_widened = bool((bbox_info or {}).get("tall_narrow_widened", False))
    reason = (bbox_info or {}).get("reason", "")
    area_ratio = (bbox_info or {}).get("area_ratio", None)
    width_ratio = (bbox_info or {}).get("width_ratio", None)
    height_ratio = (bbox_info or {}).get("height_ratio", None)

    parts = [f"bbox_mode={bbox_mode}", f"symmetric={'YES' if symmetric else 'NO'}", f"tall_narrow_widened={'YES' if tall_narrow_widened else 'NO'}"]
    if reason:
        parts.append(f"reason={reason}")
    if area_ratio is not None:
        parts.append(f"area_ratio={area_ratio:.3f}")
    if width_ratio is not None:
        parts.append(f"width_ratio={width_ratio:.3f}")
    if height_ratio is not None:
        parts.append(f"height_ratio={height_ratio:.3f}")
    label = " | ".join(parts)

    # 반투명 배경 박스처럼 보이게 검은 사각형 먼저 그리기(완전 투명은 PIL 기본만으로 어려워서 단색)
    try:
        bbox = draw.textbbox((10, 10), label, font=font)
        draw.rectangle([bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4], fill=(0, 0, 0))
    except Exception:
        pass
    try:
        draw.text((10, 10), label, fill="white", font=font, stroke_width=2, stroke_fill="black")
    except TypeError:
        draw.text((10, 10), label, fill="white", font=font)
    return out


def compute_bbox_from_mask_info(
    mask_img: Image.Image,
    pad_top: int = 32,
    pad_bottom: int = 32,
    pad_lr: int = 32,
    bbox_mode: str = "auto",  # "mask" | "auto"
    bbox_min_area_ratio: float = 0.35,
    bbox_min_width_ratio: float = 0.50,
    bbox_min_height_ratio: float = 0.55,
    single_comp_wide_threshold: float = 0.60,
    tall_bbox_aspect_threshold: float = 1.5,
    tall_bbox_min_width_frac: float = 0.5,
) -> Tuple[Optional[Tuple[int, int, int, int]], Dict[str, object]]:
    """
    compute_bbox_from_mask의 확장 버전: bbox와 함께 '대칭확장 사용 여부/이유' 같은 정보를 반환합니다.
    
    info에 포함되는 정보:
    - symmetric: 대칭 확장 사용 여부
    - reason: 이유 (single_component, imbalance_two_components, balanced_two_components, ...)
    - area_ratio, width_ratio: 좌/우 폐 비율 (해당 시)
    - mask_bbox: 원본 마스크 기반 bbox (대칭 확장 전, 패딩 적용 후) - 비교용
    """
    info: Dict[str, object] = {
        "symmetric": False,
        "reason": "",
        "area_ratio": None,
        "width_ratio": None,
        "height_ratio": None,
        "mask_bbox": None,  # 원본 마스크 기반 bbox (대칭 확장 전)
        "tall_narrow_widened": False,  # 세로로 긴 박스 가로 확장 적용 여부
    }

    m = mask_img.convert("L")
    arr = np.array(m)
    mask_bool = arr > 0
    ys, xs = np.where(mask_bool)
    if ys.size == 0 or xs.size == 0:
        return None, info

    y_min_raw, y_max_raw = int(ys.min()), int(ys.max())
    x_min_raw, x_max_raw = int(xs.min()), int(xs.max())
    h, w = arr.shape

    x_min_use = x_min_raw
    x_max_use = x_max_raw

    if str(bbox_mode).lower() == "auto":
        # NOTE: 연결요소 분석은 "원본 마스크" 기준
        comps = _connected_components_4(mask_bool)
        comps.sort(key=lambda c: c["area"], reverse=True)
        comps = comps[:2]

        if len(comps) == 2:
            c1, c2 = comps[0], comps[1]
            left = c1 if c1["cx"] <= c2["cx"] else c2
            right = c2 if left is c1 else c1

            left_area, right_area = int(left["area"]), int(right["area"])
            left_w = int(left["xmax"] - left["xmin"] + 1)
            right_w = int(right["xmax"] - right["xmin"] + 1)
            left_h = int(left["ymax"] - left["ymin"] + 1)
            right_h = int(right["ymax"] - right["ymin"] + 1)

            small_area = min(left_area, right_area)
            large_area = max(left_area, right_area)
            small_w = min(left_w, right_w)
            large_w = max(left_w, right_w)
            small_h = min(left_h, right_h)
            large_h = max(left_h, right_h)

            area_ratio = small_area / max(large_area, 1)
            width_ratio = small_w / max(large_w, 1)
            height_ratio = small_h / max(large_h, 1)
            info["area_ratio"] = float(area_ratio)
            info["width_ratio"] = float(width_ratio)
            info["height_ratio"] = float(height_ratio)

            if int(right["xmin"]) > int(left["xmax"]):
                spine = int((int(left["xmax"]) + int(right["xmin"])) // 2)
            else:
                spine = w // 2

            if (
                (area_ratio < float(bbox_min_area_ratio))
                or (width_ratio < float(bbox_min_width_ratio))
                or (height_ratio < float(bbox_min_height_ratio))
            ):
                left_width = max(0, spine - int(left["xmin"]))
                right_width = max(0, int(right["xmax"]) - spine)
                max_width = max(left_width, right_width)
                x_min_use = max(0, spine - max_width)
                x_max_use = min(w - 1, spine + max_width)
                info["symmetric"] = True
                info["reason"] = "imbalance_two_components"
            else:
                info["reason"] = "balanced_two_components"
        elif len(comps) == 1:
            # 연결요소가 1개인 경우:
            # - 진짜 한쪽 폐만 잡힌(single lung) 케이스일 수도 있고
            # - 양쪽 폐가 마스크 오류로 얇게 붙어서 1개로 보이는(merged) 케이스일 수도 있습니다.
            # 너무 넓은(예: 전체 너비의 60% 이상) 덩어리는 merged 가능성이 높으므로 대칭 확장을 금지합니다.
            comp_w = int(x_max_raw - x_min_raw + 1)
            width_frac = float(comp_w / max(w, 1))
            if width_frac >= float(single_comp_wide_threshold):
                # 확장 금지: 순수 마스크 bbox 사용
                x_min_use = x_min_raw
                x_max_use = x_max_raw
                info["symmetric"] = False
                info["reason"] = f"single_component_wide_no_expand_w{width_frac:.2f}"
            else:
                spine = w // 2
                left_width = max(0, spine - x_min_raw)
                right_width = max(0, x_max_raw - spine)
                max_width = max(left_width, right_width)
                x_min_use = max(0, spine - max_width)
                x_max_use = min(w - 1, spine + max_width)
                info["symmetric"] = True
                info["reason"] = f"single_component_expand_w{width_frac:.2f}"
        else:
            info["reason"] = "no_components"

        # 연결요소 2개인데 세로가 가로보다 압도적으로 길면 (한쪽 폐가 위/아래로 쪼개진 경우) 가로로 넓히기
        # 똑같은 크기의 박스를 없는 쪽으로만 붙임: 박스가 왼쪽에 치우치면 오른쪽으로만, 오른쪽에 치우치면 왼쪽으로만 확장
        if len(comps) == 2:
            box_h = int(y_max_raw - y_min_raw) + 1
            box_w_cur = int(x_max_use - x_min_use) + 1
            if box_w_cur > 0 and box_h > tall_bbox_aspect_threshold * box_w_cur:
                add_width = box_w_cur
                cx = (int(x_min_use) + int(x_max_use)) // 2
                if cx < w // 2:
                    x_max_use = min(w - 1, int(x_max_use) + add_width)
                else:
                    x_min_use = max(0, int(x_min_use) - add_width)
                prev = str(info.get("reason") or "")
                info["reason"] = (prev + "+tall_narrow_widened").strip("+")
                info["tall_narrow_widened"] = True
    else:
        info["reason"] = "mask_only"

    # 원본 마스크 기반 bbox (대칭 확장 전, 패딩 적용 후) - 비교용
    y_min_mask = max(0, int(y_min_raw) - int(pad_top))
    y_max_mask = min(h - 1, int(y_max_raw) + int(pad_bottom))
    x_min_mask = max(0, int(x_min_raw) - int(pad_lr))
    x_max_mask = min(w - 1, int(x_max_raw) + int(pad_lr))
    info["mask_bbox"] = (int(x_min_mask), int(y_min_mask), int(x_max_mask) + 1, int(y_max_mask) + 1)

    # 최종 bbox (대칭 확장 적용 후)
    y_min = max(0, int(y_min_raw) - int(pad_top))
    y_max = min(h - 1, int(y_max_raw) + int(pad_bottom))
    x_min = max(0, int(x_min_use) - int(pad_lr))
    x_max = min(w - 1, int(x_max_use) + int(pad_lr))

    left = int(x_min)
    upper = int(y_min)
    right = int(x_max) + 1
    lower = int(y_max) + 1
    return (left, upper, right, lower), info


def crop_with_mask(
    src_img_path: str,
    mask_path: str,
    out_path: str,
    pad_top: int = 32,
    pad_bottom: int = 32,
    pad_lr: int = 32,
    pad_frac: Optional[float] = None,
    pad_top_frac: Optional[float] = None,
    pad_bottom_frac: Optional[float] = None,
    pad_lr_frac: Optional[float] = None,
    save_mask: bool = True,
    save_orig: bool = True,
    make_square: bool = False,
    mask_overlay_alpha: float = 0.35,
    mask_keep_topk: int = 2,
    mask_min_area: int = 20,
    bbox_mode: str = "auto",
    bbox_min_area_ratio: float = 0.35,
    bbox_min_width_ratio: float = 0.50,
    bbox_min_height_ratio: float = 0.55,
    single_comp_wide_threshold: float = 0.60,
    tall_bbox_aspect_threshold: float = 1.5,
    tall_bbox_min_width_frac: float = 0.5,
    save_orig_raw: bool = False,
    mask_quality_min_area_ratio: float = 0.10,
    fallback_center_ratio: float = 0.8,
    fallback_center_full_height: bool = False,
    mask_img_override: Optional[Image.Image] = None,
    write_files: bool = True,
    quiet: bool = False,
) -> Optional[Dict[str, object]]:
    """
    단일 이미지에 대해:
    1. 원본 이미지를 로드
    2. 마스크를 읽고 바운딩박스 계산
       - mask_img_override 가 주어지면 해당 PIL Image를 마스크로 사용 (medsam3 .npy 변환본 등)
       - 없으면 mask_path 에서 파일을 읽어 사용
    3. 이미지를 bbox로 크롭하여 저장
    4. save_mask=True면 마스크 오버레이도 저장
    5. save_orig=True면 원본 디버그 이미지도 저장

    성공하면 geometry dict, 실패/스킵 시 None.
    """
    if not os.path.exists(src_img_path):
        return None

    try:
        img_orig = Image.open(src_img_path)
        img_orig.load()
    except Exception:
        return None

    # 마스크 로드: override(PIL Image 직접)를 우선 사용, 없으면 파일에서 읽기
    if mask_img_override is not None:
        mask = mask_img_override.copy()
    else:
        if not os.path.exists(mask_path):
            return None
        try:
            mask = Image.open(mask_path)
            mask.load()
        except Exception:
            return None

    # pad를 "픽셀" 또는 "비율(%%)" 기반으로 최종 결정
    # - top/bottom은 높이(H) 기준, lr은 너비(W) 기준
    try:
        W, H = img_orig.size
    except Exception:
        W, H = (0, 0)
    pad_frac = _normalize_frac(pad_frac)
    pad_top_frac_eff = _normalize_frac(pad_top_frac) if pad_top_frac is not None else pad_frac
    pad_bottom_frac_eff = _normalize_frac(pad_bottom_frac) if pad_bottom_frac is not None else pad_frac
    pad_lr_frac_eff = _normalize_frac(pad_lr_frac) if pad_lr_frac is not None else pad_frac

    pad_top = _resolve_pad_px(pad_top, pad_top_frac_eff, dim=int(H))
    pad_bottom = _resolve_pad_px(pad_bottom, pad_bottom_frac_eff, dim=int(H))
    pad_lr = _resolve_pad_px(pad_lr, pad_lr_frac_eff, dim=int(W))
    
    # ✨ Step 0: 마스크 노이즈 제거(작은 점/조각 제거) - 업스케일 전에 수행
    # (업스케일 후에 하면 노이즈도 커져서 오히려 문제를 키울 수 있음)
    try:
        mask = filter_mask_small_components(mask, keep_top_k=mask_keep_topk, min_area=mask_min_area)
    except Exception:
        pass

    # ✨ Step 1: 마스크를 CXR 크기에 맞게 업스케일
    if img_orig.size != mask.size:
        mask = mask.resize(img_orig.size, Image.NEAREST)

    mask_orig = mask
    mask_full = mask_orig
    img = img_orig

    # ✨ Step 2: 마스크 품질 검사 + bbox 계산(또는 fallback)
    bbox = None
    bbox_info: Dict[str, object] = {
        "symmetric": False,
        "reason": "",
        "area_ratio": None,
        "width_ratio": None,
        "height_ratio": None,
        "mask_bbox": None,
    }

    # ✅ 불량 마스크면(라벨과 무관하게) 무조건 Center Crop fallback
    is_good_mask, reason = check_mask_quality(
        mask_orig,
        img_orig.size,
        min_area_ratio=mask_quality_min_area_ratio,
    )

    if (not is_good_mask):
        bbox = _center_crop_bbox(
            img_orig.size,
            crop_ratio=fallback_center_ratio,
            full_height=bool(fallback_center_full_height),
        )
        bbox_info["reason"] = f"fallback_center_{reason}"
        if not quiet:
            print(f"⚠️ Poor mask ({reason}) -> Fallback center crop: {os.path.basename(src_img_path)}")
    else:
        # 마스크가 멀쩡하면 원래대로 정밀 크롭 수행
        bbox, bbox_info = compute_bbox_from_mask_info(
            mask_orig,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_lr=pad_lr,
            bbox_mode=bbox_mode,
            bbox_min_area_ratio=bbox_min_area_ratio,
            bbox_min_width_ratio=bbox_min_width_ratio,
            bbox_min_height_ratio=bbox_min_height_ratio,
            single_comp_wide_threshold=single_comp_wide_threshold,
            tall_bbox_aspect_threshold=tall_bbox_aspect_threshold,
            tall_bbox_min_width_frac=tall_bbox_min_width_frac,
        )
    if bbox is None:
        return None

    # ✅ 보수적(안전) 처리:
    # 최종 bbox가 어떤 이유로든 원본 mask_bbox보다 작아지면 폐가 잘릴 수 있으므로 union으로 더 큰 bbox 사용
    try:
        mask_bbox = bbox_info.get("mask_bbox")
        if mask_bbox is not None:
            l1, u1, r1, d1 = [int(v) for v in bbox]
            l2, u2, r2, d2 = [int(v) for v in mask_bbox]
            left = min(l1, l2)
            upper = min(u1, u2)
            right = max(r1, r2)
            lower = max(d1, d2)
            W, H = img_orig.size
            left = max(0, min(int(W) - 1, int(left)))
            upper = max(0, min(int(H) - 1, int(upper)))
            right = max(left + 1, min(int(W), int(right)))
            lower = max(upper + 1, min(int(H), int(lower)))
            bbox = (left, upper, right, lower)
            prev_reason = str(bbox_info.get("reason") or "")
            if "union" not in prev_reason:
                bbox_info["reason"] = (prev_reason + "+union_mask_bbox").strip("+")
    except Exception:
        pass

    # ✨ Step 3: 이미지 크롭
    cropped = img.crop(bbox)

    # ✨ Step 4: 검은 테두리 제거 (trim)
    cropped_trimmed, trim_bbox = trim_zero_border_box(cropped, threshold=0)

    # ✨ Step 5: (옵션) 정사각형 패딩 - 짧은 변 양쪽에 검은 여백 추가
    padding = (0, 0, 0, 0)
    cropped_final = cropped_trimmed
    if make_square:
        padding = square_padding_ltrb(cropped_trimmed.size)
        cropped_final = pad_to_square(cropped_trimmed, fill=0)

    orig_w, orig_h = img_orig.size
    geom: Dict[str, object] = {
        "original_size": [int(orig_w), int(orig_h)],
        "crop_bbox": [int(v) for v in bbox],
        "trim_bbox": [int(v) for v in trim_bbox],
        "padding": [int(v) for v in padding],
        "final_crop_size": [int(cropped_final.size[0]), int(cropped_final.size[1])],
        "make_square": bool(make_square),
        "bbox_reason": str(bbox_info.get("reason") or ""),
    }

    if not write_files:
        geom["image"] = cropped_final
        return geom

    # 마스크는 업스케일만 하고 크롭/trim 모두 하지 않음!
    _ensure_dir(os.path.dirname(out_path))
    try:
        # ✨ Step 6: 크롭+trim(+옵션: square) CXR 저장
        cropped_final.save(out_path)

        base_dir = os.path.dirname(out_path)
        base_name = os.path.basename(out_path)
        name_without_ext, ext = os.path.splitext(base_name)

        # 원본 CXR도 함께 저장 (pre_orig.jpg / post_orig.jpg)
        # - orig.jpg에 mask + bbox + 대칭확장 여부까지 '통합'해서 저장
        # - 필요하면 원본(raw)도 *_orig_raw.jpg로 추가 저장 가능
        if save_orig:
            orig_out_path = os.path.join(base_dir, f"{name_without_ext}_orig{ext}")
            # 대칭 확장 전 원본 마스크 bbox (빨간색)
            mask_bbox_for_vis = bbox_info.get("mask_bbox") if bbox_info.get("symmetric") else None
            bbox_color_for_vis = _pick_bbox_color(bbox_info)
            orig_debug = overlay_mask_on_image(
                img_orig,
                mask_full,
                color=(0, 255, 0),
                alpha=mask_overlay_alpha,
                bbox=bbox,
                bbox_color=bbox_color_for_vis,
                bbox_width=5,
                mask_bbox=mask_bbox_for_vis,
                mask_bbox_color=(255, 0, 0),
            )
            orig_debug = draw_bbox_info_label(orig_debug, bbox_mode=bbox_mode, bbox_info=bbox_info, font_size=24)
            orig_debug.save(orig_out_path)
            if save_orig_raw:
                orig_raw_out_path = os.path.join(base_dir, f"{name_without_ext}_orig_raw{ext}")
                img_orig.save(orig_raw_out_path)

        # 마스크 저장 (pre.jpg → pre_mask.jpg 형식)
        # 기본은 별도 저장을 하지 않으며(통합 orig로 대체),
        # 과거에 생성된 *_mask.jpg는 save_mask=False일 때 자동 삭제합니다.
        mask_out_path = os.path.join(base_dir, f"{name_without_ext}_mask{ext}")
        if save_mask:
            bbox_color_for_vis = _pick_bbox_color(bbox_info)
            mask_overlay = overlay_mask_on_image(
                img_orig,
                mask_full,
                color=(0, 255, 0),
                alpha=mask_overlay_alpha,
                bbox=bbox,
                bbox_color=bbox_color_for_vis,
                bbox_width=5,
            )
            mask_overlay.save(mask_out_path)
        else:
            try:
                if os.path.exists(mask_out_path):
                    os.remove(mask_out_path)
            except Exception:
                pass

        return geom
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(
        description=(
            "폐 마스크를 이용해 Pneumonia 분류용 CXR을 크롭합니다. "
            "--seg-mode medsam3(기본) 또는 chexmask, "
            "--view PA(기본) / AP / both 선택."
        )
    )
    ap.add_argument(
        "--seg-mode",
        type=str,
        default="medsam3",
        choices=["medsam3", "chexmask"],
        help=(
            "폐 마스크 소스 선택 (기본: medsam3)\n"
            "  medsam3  : MedSAM3가 생성한 .npy 마스크 파일 사용\n"
            "  chexmask : ChexMask CSV(RLE)에서 폐 마스크 복원하여 사용"
        ),
    )
    ap.add_argument(
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
    ap.add_argument(
        "--chexmask-csv",
        type=str,
        default=None,
        help=(
            f"--seg-mode chexmask 시 사용할 ChexMask CSV 경로 "
            f"(기본: {DEFAULT_CHEXMASK_CSV})"
        ),
    )
    ap.add_argument(
        "--mask-keep-topk",
        type=int,
        default=2,
        help="마스크 노이즈 제거 시 남길 연결요소 개수(보통 폐 1~2개, 기본: 2)",
    )
    ap.add_argument(
        "--mask-min-area",
        type=int,
        default=20,
        help="마스크 노이즈 제거 시 최소 면적(픽셀). 이보다 작은 조각은 버립니다(기본: 20)",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help=(
            "입력 manifest JSON 경로. 미지정 시 --seg-mode와 --view에 따라 자동 결정.\n"
            f"  PA+medsam3  기본: {DEFAULT_SEG_ROOT}_pa/manifest_pa_medsam3_ok.json\n"
            f"  AP+chexmask 기본: {DEFAULT_CHEXMASK_SEG_ROOT}_ap/manifest_ap_chexmask_ok.json"
        ),
    )
    ap.add_argument(
        "--out-root",
        default=None,
        help=(
            "크롭된 이미지를 저장할 루트 (미지정 시 seg-mode와 --view에 따라 자동 결정)\n"
            f"  PA+medsam3  기본: {DEFAULT_OUT_ROOT}_pa\n"
            f"  PA+chexmask 기본: {DEFAULT_CHEXMASK_OUT_ROOT}_pa\n"
            "  PA/AP/both 는 위 경로에 _pa / _ap / _pa_ap 접미사 추가"
        ),
    )
    ap.add_argument(
        "--pad",
        type=int,
        default=None,
        help="(픽셀) 기본 pad 픽셀 수 (지정 시 pad_lr 기본값으로 사용). 비율 패딩을 끄려면 --pad-frac 0 사용",
    )
    ap.add_argument(
        "--pad-top",
        type=int,
        default=None,
        help="(픽셀) 위쪽(upper)에만 적용할 pad 픽셀 수. 비율 패딩을 끄려면 --pad-frac 0 사용",
    )
    ap.add_argument(
        "--pad-bottom",
        type=int,
        default=None,
        help="(픽셀) 아래쪽(lower)에만 적용할 pad 픽셀 수. 비율 패딩을 끄려면 --pad-frac 0 사용",
    )
    ap.add_argument(
        "--pad-lr",
        type=int,
        default=None,
        help="(픽셀) 좌우(left/right)에만 적용할 pad 픽셀 수 (지정 안 하면 --pad 값 사용). 비율 패딩을 끄려면 --pad-frac 0 사용",
    )
    ap.add_argument(
        "--pad-frac",
        type=float,
        default=0.05,
        help="(기본=5%%) pad 공통 비율. 상하좌우 비율 미지정 시 이 값 사용 (예: 0.05=5%%, 5=5%%). 0을 주면 비율 모드 끄고 픽셀 pad만 사용",
    )
    ap.add_argument(
        "--pad-top-frac",
        type=float,
        default=0.07,
        help="(기본=7%%) 위쪽 pad를 높이 대비 비율로 지정 (예: 0.06=6%%). --pad-frac보다 우선",
    )
    ap.add_argument(
        "--pad-bottom-frac",
        type=float,
        default=0.10,
        help="(기본=10%%) 아래쪽 pad를 높이 대비 비율로 지정 (예: 0.12=12%%). --pad-frac보다 우선",
    )
    ap.add_argument(
        "--pad-lr-frac",
        type=float,
        default=0.05,
        help="(기본=5%%) 좌/우 pad를 너비 대비 비율로 지정 (예: 0.05=5%%). --pad-frac보다 우선",
    )
    ap.add_argument(
        "--bbox-mode",
        type=str,
        default="auto",
        choices=["auto", "mask"],
        help="크롭 bbox 계산 방식 (auto=조건부 대칭 확장, mask=순수 마스크 bbox, 기본: auto)",
    )
    ap.add_argument(
        "--bbox-min-area-ratio",
        type=float,
        default=0.5,
        help="auto 모드에서 좌/우 폐 면적 비율(작은/큰)이 이 값보다 작으면 대칭 확장 (기본: 0.5)",
    )
    ap.add_argument(
        "--bbox-min-width-ratio",
        type=float,
        default=0.70,
        help="auto 모드에서 좌/우 폐 너비 비율(작은/큰)이 이 값보다 작으면 대칭 확장 (기본: 0.70)",
    )
    ap.add_argument(
        "--bbox-min-height-ratio",
        type=float,
        default=0.55,
        help="auto 모드에서 좌/우 폐 높이 비율(작은/큰)이 이 값보다 작으면 대칭 확장 (기본: 0.55)",
    )
    ap.add_argument(
        "--single-comp-wide-threshold",
        type=float,
        default=0.55,
        help="연결요소가 1개로 잡혔을 때, 덩어리 너비/전체너비가 이 값 이상이면 '두 폐가 붙음(merged)'으로 보고 대칭확장 금지 (기본: 0.55)",
    )
    ap.add_argument(
        "--tall-bbox-aspect-threshold",
        type=float,
        default=1.5,
        help="연결요소 2개일 때 세로/가로 비가 이 값보다 크면 '세로로 긴 박스'로 보고 가로 확장 (기본: 1.5)",
    )
    ap.add_argument(
        "--tall-bbox-min-width-frac",
        type=float,
        default=0.5,
        help="세로로 긴 박스 가로 확장 시 최소 너비 = 높이×이 비율 (기본: 0.5, 즉 너비≥높이의 50%%)",
    )
    ap.add_argument(
        "--mask-quality-min-area-ratio",
        type=float,
        default=0.10,
        help="불량 마스크 최소 면적 비율(마스크 픽셀/전체 픽셀). 이보다 작으면 불량 처리 (기본: 0.10)",
    )
    ap.add_argument(
        "--fallback-center-ratio",
        type=float,
        default=0.8,
        help="fallback 중앙 크롭 비율(0~1). 0.8이면 가로/세로 80%% 영역을 중앙에서 자름 (기본: 0.8)",
    )
    ap.add_argument(
        "--fallback-center-full-height",
        action="store_true",
        default=False,
        help="fallback 중앙 크롭 시 세로는 자르지 않고(=전체 높이 유지) 가로만 중앙 크롭합니다",
    )
    ap.add_argument(
        "--save-orig",
        action="store_true",
        default=True,
        help="크롭/trim 전 원본 CXR 이미지를 *_orig.jpg로 함께 저장 (기본: True, --no-save-orig로 비활성화)",
    )
    ap.add_argument(
        "--no-save-orig",
        action="store_false",
        dest="save_orig",
        help="원본 CXR 저장 기능 비활성화",
    )
    ap.add_argument(
        "--save-orig-raw",
        action="store_true",
        default=False,
        help="*_orig.jpg(통합 디버그) 외에, 원본(raw)도 *_orig_raw.jpg로 추가 저장",
    )
    ap.add_argument(
        "--save-mask",
        action="store_true",
        default=False,
        help="*_mask.jpg를 별도로 저장합니다 (기본: False; *_orig.jpg에 통합되어 별도 저장은 권장하지 않음)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="최대 처리할 이미지 개수 (0이면 전체 처리)",
    )
    ap.add_argument(
        "--mask-overlay-alpha",
        type=float,
        default=0.35,
        help="*_mask.jpg 저장 시 마스크 overlay 투명도 (0~1, 기본: 0.35)",
    )
    ap.add_argument(
        "--make-square",
        action="store_true",
        default=True,
        help="전처리 완료 후, 짧은 변 양쪽에 검은 여백을 추가해 정사각형으로 만듭니다 (기본: True)",
    )
    ap.add_argument(
        "--no-make-square",
        action="store_false",
        dest="make_square",
        help="정사각형 패딩을 하지 않습니다",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="이미 처리된 파일은 건너뛰고 이어서 처리합니다 (기본: True)",
    )
    ap.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="모든 파일을 처음부터 다시 처리합니다 (기존 파일 덮어쓰기)",
    )

    args = ap.parse_args()

    seg_mode = args.seg_mode  # "medsam3" 또는 "chexmask"
    view_arg = args.view.upper()  # PA / AP / BOTH

    # view 인자 → 파일명 태그 및 seg_root 접미사 결정
    if view_arg == "BOTH":
        view_tag = "pa_ap"
        view_suffix = "_pa_ap"
    elif view_arg == "AP":
        view_tag = "ap"
        view_suffix = "_ap"
    else:  # PA
        view_tag = "pa"
        view_suffix = "_pa"

    seg_tag = "chexmask" if seg_mode == "chexmask" else "medsam3"

    # --manifest 미지정 시 seg-mode + view로 자동 결정
    if args.manifest is not None:
        manifest_path = args.manifest
    elif seg_mode == "chexmask":
        seg_root = DEFAULT_CHEXMASK_SEG_ROOT + view_suffix
        manifest_path = os.path.join(seg_root, f"manifest_{view_tag}_{seg_tag}_ok.json")
    else:
        seg_root = DEFAULT_SEG_ROOT + view_suffix
        manifest_path = os.path.join(seg_root, f"manifest_{view_tag}_{seg_tag}_ok.json")

    # --out-root 미지정 시 seg-mode + view에 따라 기본값 자동 선택
    if args.out_root is not None:
        out_root = args.out_root
    elif seg_mode == "chexmask":
        out_root = DEFAULT_CHEXMASK_OUT_ROOT + view_suffix
    else:
        out_root = DEFAULT_OUT_ROOT + view_suffix

    print(f"🔧 Segmentation mode: {seg_mode}")
    print(f"🔧 View filter      : {view_arg}")

    # pad 값 설정
    base_pad = int(args.pad) if args.pad is not None else 0
    pad_top = int(args.pad_top) if args.pad_top is not None else base_pad
    pad_bottom = int(args.pad_bottom) if args.pad_bottom is not None else base_pad
    pad_lr = int(args.pad_lr) if args.pad_lr is not None else base_pad

    pad_frac = _normalize_frac(getattr(args, "pad_frac", None))
    pad_top_frac = _normalize_frac(getattr(args, "pad_top_frac", None))
    pad_bottom_frac = _normalize_frac(getattr(args, "pad_bottom_frac", None))
    pad_lr_frac = _normalize_frac(getattr(args, "pad_lr_frac", None))

    # manifest 로드 (manifest_pa_medsam3_ok.json - 두 모드 공통으로 이미지 경로·라벨 제공)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    print(f"📥 Loading manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"✅ Loaded {len(manifest):,} images from manifest")
    print(f"📂 Output root : {out_root}")

    # ChexMask 모드: CSV 로드 → {dicom_id: row} lookup 구성
    chexmask_lookup: Dict[str, dict] = {}
    if seg_mode == "chexmask":
        chexmask_csv_path = args.chexmask_csv if args.chexmask_csv else DEFAULT_CHEXMASK_CSV
        chexmask_lookup = load_chexmask_lookup(chexmask_csv_path)

    if args.limit and args.limit > 0:
        manifest = manifest[: args.limit]
        print(f"⚠️ limit={args.limit} 로 앞에서부터 {len(manifest):,}개 이미지만 처리합니다.")

    n_crop_ok = 0
    n_skipped = 0
    n_mask_missing = 0
    n_src_missing = 0
    n_chexmask_missing = 0
    out_manifest_entries = []

    if args.resume:
        print("🔄 Resume 모드: 이미 처리된 파일은 건너뜁니다")
    else:
        print("⚠️ 전체 재처리 모드: 기존 파일을 덮어씁니다")

    common_crop_kwargs = dict(
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_lr=pad_lr,
        pad_frac=pad_frac,
        pad_top_frac=pad_top_frac,
        pad_bottom_frac=pad_bottom_frac,
        pad_lr_frac=pad_lr_frac,
        save_mask=args.save_mask,
        save_orig=args.save_orig,
        make_square=args.make_square,
        mask_overlay_alpha=args.mask_overlay_alpha,
        mask_keep_topk=args.mask_keep_topk,
        mask_min_area=args.mask_min_area,
        bbox_mode=args.bbox_mode,
        bbox_min_area_ratio=args.bbox_min_area_ratio,
        bbox_min_width_ratio=args.bbox_min_width_ratio,
        bbox_min_height_ratio=args.bbox_min_height_ratio,
        single_comp_wide_threshold=args.single_comp_wide_threshold,
        tall_bbox_aspect_threshold=args.tall_bbox_aspect_threshold,
        tall_bbox_min_width_frac=args.tall_bbox_min_width_frac,
        save_orig_raw=args.save_orig_raw,
        mask_quality_min_area_ratio=args.mask_quality_min_area_ratio,
        fallback_center_ratio=args.fallback_center_ratio,
        fallback_center_full_height=args.fallback_center_full_height,
    )

    desc = f"Cropping CXR with {seg_mode} lung masks"
    for entry in tqdm(manifest, desc=desc):
        # manifest entry에서 필요한 정보 추출
        src_img_path = entry.get("source_image_abs_path", "")
        rel_path = entry.get("source_image_rel_path", "")
        dicom_id = entry.get("dicom_id", "")

        if not src_img_path or not os.path.isfile(src_img_path):
            n_src_missing += 1
            continue

        # 출력 경로: out_root 아래에 rel_path 구조 유지
        # 예: files/p10/p10000980/s50985099/{dicom_id}.jpg
        if rel_path:
            out_jpg_path = os.path.join(out_root, rel_path)
        else:
            out_jpg_path = os.path.join(out_root, f"{dicom_id}.jpg")

        # Resume 모드: 이미 유효한 파일이면 건너뜀
        if args.resume and _is_valid_image_file(out_jpg_path):
            n_skipped += 1
            out_entry = entry.copy()
            out_entry["cropped_cxr_jpg"] = os.path.abspath(out_jpg_path)
            out_manifest_entries.append(out_entry)
            continue

        # ── 마스크 로딩: seg-mode에 따라 분기 ────────────────────────────────
        if seg_mode == "chexmask":
            # ChexMask CSV에서 RLE 디코딩 → PIL mask
            if dicom_id not in chexmask_lookup:
                n_chexmask_missing += 1
                continue
            # 원본 CXR 크기를 먼저 읽어서 마스크 리사이즈에 활용
            try:
                with Image.open(src_img_path) as _img:
                    cxr_w, cxr_h = _img.size
            except Exception:
                n_src_missing += 1
                continue
            mask_pil = load_chexmask_as_pil(
                chexmask_lookup[dicom_id],
                target_size=(cxr_w, cxr_h),
            )
            if mask_pil is None:
                n_chexmask_missing += 1
                continue
        else:
            # medsam3 모드: .npy 마스크 파일 로드
            npy_mask_path = entry.get("lung_mask_npy", "")
            if not npy_mask_path or not os.path.isfile(npy_mask_path):
                n_mask_missing += 1
                continue
            mask_pil = load_npy_mask_as_pil(npy_mask_path)
            if mask_pil is None:
                n_mask_missing += 1
                continue
        # ────────────────────────────────────────────────────────────────────

        geom = crop_with_mask(
            src_img_path,
            "",
            out_jpg_path,
            mask_img_override=mask_pil,
            **common_crop_kwargs,
        )

        if geom:
            n_crop_ok += 1
            out_entry = entry.copy()
            out_entry["cropped_cxr_jpg"] = os.path.abspath(out_jpg_path)
            out_entry["original_size"] = geom["original_size"]
            out_entry["crop_bbox"] = geom["crop_bbox"]
            out_entry["trim_bbox"] = geom["trim_bbox"]
            out_entry["padding"] = geom["padding"]
            out_entry["final_crop_size"] = geom["final_crop_size"]
            out_entry["make_square"] = geom["make_square"]
            out_manifest_entries.append(out_entry)

    # 출력 manifest 파일명: view_tag + seg_tag 조합
    out_manifest_name = f"manifest_{view_tag}_{seg_tag}_ok.json"
    out_manifest_path = os.path.join(out_root, out_manifest_name)
    _ensure_dir(out_root)
    with open(out_manifest_path, "w", encoding="utf-8") as f:
        json.dump(out_manifest_entries, f, ensure_ascii=False, indent=2)
    print(f"📄 Output manifest saved: {out_manifest_path} ({len(out_manifest_entries):,} images)")

    print("✅ Cropping finished.")
    if n_skipped > 0:
        print(f"  - 건너뛴 이미지 (이미 처리됨): {n_skipped:,} 개")
    if seg_mode == "medsam3" and n_mask_missing > 0:
        print(f"  - .npy 마스크 없음/로드 실패(건너뜀): {n_mask_missing:,} 개")
    if seg_mode == "chexmask" and n_chexmask_missing > 0:
        print(f"  - ChexMask CSV 미매칭/디코딩 실패(건너뜀): {n_chexmask_missing:,} 개")
    if n_src_missing > 0:
        print(f"  - 원본 CXR 파일 없음(건너뜀): {n_src_missing:,} 개")
    print(f"  - 크롭 성공: {n_crop_ok:,} 개")
    print(f"  - 총 출력 manifest 항목: {len(out_manifest_entries):,} 개")
    print(f"  - 출력 루트: {out_root}")

    # 완료 후 전체 이미지 손상 스캔
    print("🔍 Scanning all output images for corruption...")
    n_ok = 0
    n_bad = 0
    bad_list = []
    for entry in tqdm(out_manifest_entries, desc="Validating images"):
        cropped_path = entry.get("cropped_cxr_jpg", "")
        if not cropped_path:
            continue
        if _is_valid_image_file(cropped_path):
            n_ok += 1
        else:
            n_bad += 1
            bad_list.append(cropped_path)
    print(f"📋 Scan result: {n_ok:,} images OK, {n_bad:,} images with missing/corrupt file(s)")
    if bad_list:
        for bp in bad_list[:50]:
            print(f"   - {bp}")
        if len(bad_list) > 50:
            print(f"   ... and {len(bad_list) - 50:,} more")


if __name__ == "__main__":
    main()


