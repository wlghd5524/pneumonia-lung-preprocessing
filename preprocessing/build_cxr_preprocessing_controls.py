#!/usr/bin/env python3
"""MedSAM3 전처리 대조(control) 영상 생성.

리뷰어 요청: 해부학적 영역 제한 / 폐 확대 / 배경 억제 / 인위적 마스크 경계의
기여를 분리하기 위해, 아래 대조 입력으로 분류기를 실제로 학습합니다.

  center    : 본 실험 크롭과 같은 크기의 사각형을 영상 한가운데에 둔 크롭
  margin0   : 폐 사각형(여백 0%) 크롭
  margin10  : 폐 사각형 + 위/아래 높이의 10%, 좌/우 너비의 10% 여백
  margin20  : 폐 사각형 + 위/아래 높이의 20%, 좌/우 너비의 20% 여백
  soft      : hard mask와 같은 폐 마스크를 가우시안으로 흐린 가중치(alpha)를 곱한 전체 영상

사각형과 soft mask 정의는 ``mask_analysis/analyze_input_geometry.py`` 와 같습니다.
크롭 계열의 뒷처리(검은 테두리 trim → 정사각형 검은 여백 → JPEG 기본 품질)와
soft mask 저장 방식(RGB, quality=95)은 각각 본 실험 크롭 / hard mask 영상과 같습니다.

마스크가 불량이어서 본 실험 크롭이 80% × 80% 중앙 크롭으로 대체된 영상은
크롭 대조 4종에서도 같은 80% 중앙 크롭을 씁니다.

출력:
  {out_base}/cxr_medsam3_control_{control}_{view}/
      manifest_{view}_medsam3_ok.json
      files/...               (크롭 계열, manifest 필드 cropped_cxr_jpg)
      masked_cxr/files/...    (soft,     manifest 필드 masked_cxr_jpg)

manifest 항목과 순서는 ``cxr_medsam3_lung_seg_cropped_{view}`` manifest와 같습니다.
한 장이라도 실패하면 manifest를 쓰지 않습니다.

예:
  python build_cxr_preprocessing_controls.py --view both
  python build_cxr_preprocessing_controls.py --view PA --limit 50 --out-base /tmp/ctrl_test
  python build_cxr_preprocessing_controls.py --view PA --verify-study-crop 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile
from scipy.ndimage import gaussian_filter

STUDY_DIR = Path(__file__).resolve().parents[1]
DATA_BASE_DIR = Path(os.environ.get("DATA_BASE_DIR", STUDY_DIR)).resolve()
sys.path.insert(0, str(STUDY_DIR / "preprocessing"))

from build_cxr_lung_crop_dataset import (  # noqa: E402
    _center_crop_bbox,
    check_mask_quality,
    compute_bbox_from_mask_info,
    filter_mask_small_components,
    pad_to_square,
    square_padding_ltrb,
    trim_zero_border_box,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

CROP_CONTROLS = ("center", "margin0", "margin10", "margin20")
ALL_CONTROLS = CROP_CONTROLS + ("soft",)
MARGIN_FRAC = {"margin0": 0.0, "margin10": 0.10, "margin20": 0.20}

INPUT_SIZE = 512
SOFT_SIGMA_512 = 0.02 * INPUT_SIZE
STUDY_PAD_TOP_FRAC = 0.07
STUDY_PAD_BOTTOM_FRAC = 0.10
STUDY_PAD_LR_FRAC = 0.05
MASK_KEEP_TOPK = 2
MASK_MIN_AREA = 20
QUALITY_MIN = 0.10
FALLBACK_CENTER_RATIO = 0.8
BBOX_KW = dict(
    bbox_mode="auto",
    bbox_min_area_ratio=0.50,
    bbox_min_width_ratio=0.70,
    bbox_min_height_ratio=0.55,
    single_comp_wide_threshold=0.55,
    tall_bbox_aspect_threshold=1.5,
    tall_bbox_min_width_frac=0.5,
)
SOFT_JPEG_QUALITY = 95

DEFAULT_SOURCE_ROOT = Path(os.environ.get("MIMIC_CXR_ROOT", STUDY_DIR / "data" / "mimic-cxr-jpg" / "2.1.0"))


def reference_manifest(view: str) -> Path:
    v = view.lower()
    return DATA_BASE_DIR / f"cxr_medsam3_lung_seg_cropped_{v}" / f"manifest_{v}_medsam3_ok.json"


def control_root(out_base: Path, control: str, view: str) -> Path:
    return out_base / f"cxr_medsam3_control_{control}_{view.lower()}"


def control_image_path(out_base: Path, control: str, view: str, rel_path: str) -> Path:
    root = control_root(out_base, control, view)
    rel = rel_path.lstrip("/")
    if control == "soft":
        return root / "masked_cxr" / rel
    return root / rel


def resolve_study_path(raw: str) -> Path:
    """이전 컴퓨터의 절대경로(/mnt/d/.../IEEE_ICCBE/...)를 현재 STUDY_DIR로 옮깁니다."""
    p = Path(str(raw))
    if p.is_file():
        return p
    text = str(raw).replace("\\", "/")
    token = "/IEEE_ICCBE/"
    if token in text:
        return STUDY_DIR / text.split(token, 1)[1]
    return p


def resolve_source_path(entry: dict, source_root: Path) -> Path:
    rel = str(entry.get("source_image_rel_path") or "").lstrip("/")
    if rel:
        cand = source_root / rel
        if cand.is_file():
            return cand
    return Path(str(entry.get("source_image_abs_path") or ""))


def clamp_bbox(bbox, w: int, h: int) -> Tuple[int, int, int, int]:
    left, upper, right, lower = [int(v) for v in bbox]
    left = max(0, min(w - 1, left))
    upper = max(0, min(h - 1, upper))
    right = max(left + 1, min(w, right))
    lower = max(upper + 1, min(h, lower))
    return left, upper, right, lower


def union_bbox(a, b, w: int, h: int) -> Tuple[int, int, int, int]:
    if b is None:
        return clamp_bbox(a, w, h)
    return clamp_bbox((min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])), w, h)


def pad_bbox(core, w: int, h: int, pad_top: int, pad_bottom: int, pad_lr: int):
    left, upper, right, lower = core
    return clamp_bbox((left - pad_lr, upper - pad_top, right + pad_lr, lower + pad_bottom), w, h)


def matched_center_bbox(w: int, h: int, crop_w: int, crop_h: int):
    crop_w = max(1, min(w, crop_w))
    crop_h = max(1, min(h, crop_h))
    left = max(0, (w - crop_w) // 2)
    upper = max(0, (h - crop_h) // 2)
    return clamp_bbox((left, upper, left + crop_w, upper + crop_h), w, h)


def compute_boxes(mask_filtered: Image.Image, w: int, h: int) -> Dict[str, object]:
    """본 실험 크롭 사각형과 대조 크롭 사각형 4종을 계산합니다."""
    is_good, reason = check_mask_quality(mask_filtered, (w, h), min_area_ratio=QUALITY_MIN)
    core = None
    if is_good:
        bbox0, info = compute_bbox_from_mask_info(mask_filtered, pad_top=0, pad_bottom=0, pad_lr=0, **BBOX_KW)
        if bbox0 is not None:
            core = union_bbox(bbox0, info.get("mask_bbox"), w, h)
        else:
            reason = "no_bbox"

    if core is None:
        fb = clamp_bbox(_center_crop_bbox((w, h), crop_ratio=FALLBACK_CENTER_RATIO, full_height=False), w, h)
        boxes = {"study": fb, "center": fb, "margin0": fb, "margin10": fb, "margin20": fb}
        return {"fallback": True, "reason": str(reason), "core": None, "boxes": boxes}

    study = pad_bbox(
        core, w, h,
        int(round(STUDY_PAD_TOP_FRAC * h)),
        int(round(STUDY_PAD_BOTTOM_FRAC * h)),
        int(round(STUDY_PAD_LR_FRAC * w)),
    )
    boxes = {"study": study}
    boxes["center"] = matched_center_bbox(w, h, study[2] - study[0], study[3] - study[1])
    for name, frac in MARGIN_FRAC.items():
        boxes[name] = pad_bbox(core, w, h, int(round(frac * h)), int(round(frac * h)), int(round(frac * w)))
    return {"fallback": False, "reason": "ok", "core": core, "boxes": boxes}


def render_crop(img: Image.Image, bbox) -> Tuple[Image.Image, Dict[str, object]]:
    """본 실험 크롭과 같은 뒷처리: crop → 검은 테두리 trim → 정사각형 검은 여백."""
    cropped = img.crop(tuple(int(v) for v in bbox))
    trimmed, trim_box = trim_zero_border_box(cropped, threshold=0)
    padding = square_padding_ltrb(trimmed.size)
    final = pad_to_square(trimmed, fill=0)
    geom = {
        "control_bbox": [int(v) for v in bbox],
        "trim_bbox": [int(v) for v in trim_box],
        "padding": [int(v) for v in padding],
        "final_crop_size": [int(final.size[0]), int(final.size[1])],
        "make_square": True,
    }
    return final, geom


def soft_alpha(mask_bool: np.ndarray, w: int, h: int) -> np.ndarray:
    """512 격자에서 폐 마스크를 흐린 뒤(σ = 512의 2%) 원본 크기로 되돌린 가중치."""
    m512 = Image.fromarray(mask_bool.astype(np.uint8) * 255).resize(
        (INPUT_SIZE, INPUT_SIZE), resample=Image.Resampling.NEAREST
    )
    a512 = gaussian_filter((np.asarray(m512) > 0).astype(np.float32), sigma=SOFT_SIGMA_512)
    a_full = Image.fromarray(a512.astype(np.float32)).resize((w, h), resample=Image.Resampling.BILINEAR)
    return np.clip(np.asarray(a_full, dtype=np.float32), 0.0, 1.0)


def render_soft(img: Image.Image, mask_bool: np.ndarray) -> Image.Image:
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    alpha = soft_alpha(mask_bool, w, h)
    out = np.clip(np.rint(rgb * alpha[..., None]), 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def atomic_save(img: Image.Image, out_path: Path, **save_kw) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.stem}.tmp{os.getpid()}{out_path.suffix}")
    img.save(tmp, format="JPEG", **save_kw)
    os.replace(tmp, out_path)


def load_inputs(entry: dict, source_root: Path):
    src_path = resolve_source_path(entry, source_root)
    npy_path = resolve_study_path(entry.get("lung_mask_npy") or "")
    if not src_path.is_file():
        raise FileNotFoundError(f"source image missing: {src_path}")
    if not npy_path.is_file():
        raise FileNotFoundError(f"lung mask missing: {npy_path}")
    img = Image.open(src_path)
    img.load()
    arr = np.load(npy_path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    mask_bool = arr.astype(bool)
    w, h = img.size
    mask_img = Image.fromarray(mask_bool.astype(np.uint8) * 255)
    if mask_img.size != (w, h):
        mask_img = mask_img.resize((w, h), Image.NEAREST)
        mask_bool = np.asarray(mask_img) > 0
    return img, mask_bool, mask_img


def process_entry(task) -> Dict[str, object]:
    idx, entry, view, controls, out_base, source_root, resume = task
    rel = str(entry.get("source_image_rel_path") or f"{entry.get('dicom_id')}.jpg")
    out_paths = {c: control_image_path(out_base, c, view, rel) for c in controls}
    try:
        img, mask_bool, mask_img = load_inputs(entry, source_root)
        w, h = img.size
        filtered = filter_mask_small_components(mask_img, keep_top_k=MASK_KEEP_TOPK, min_area=MASK_MIN_AREA)
        if filtered.size != (w, h):
            filtered = filtered.resize((w, h), Image.NEAREST)
        box_info = compute_boxes(filtered, w, h)

        per_control: Dict[str, Dict[str, object]] = {}
        for c in controls:
            out_path = out_paths[c]
            if c == "soft":
                if not (resume and out_path.is_file() and out_path.stat().st_size > 0):
                    atomic_save(render_soft(img, mask_bool), out_path, quality=SOFT_JPEG_QUALITY)
                per_control[c] = {"path": str(out_path), "soft_sigma_512": SOFT_SIGMA_512}
                continue
            bbox = box_info["boxes"][c]
            if resume and out_path.is_file() and out_path.stat().st_size > 0:
                cropped = img.crop(tuple(int(v) for v in bbox))
                _, trim_box = trim_zero_border_box(cropped, threshold=0)
                tw, th = trim_box[2] - trim_box[0], trim_box[3] - trim_box[1]
                side = max(tw, th)
                geom = {
                    "control_bbox": [int(v) for v in bbox],
                    "trim_bbox": [int(v) for v in trim_box],
                    "padding": [int(v) for v in square_padding_ltrb((tw, th))],
                    "final_crop_size": [int(side), int(side)],
                    "make_square": True,
                }
            else:
                final, geom = render_crop(img, bbox)
                atomic_save(final, out_path)
            geom["path"] = str(out_path)
            per_control[c] = geom

        return {
            "idx": idx,
            "ok": True,
            "original_size": [int(w), int(h)],
            "fallback": bool(box_info["fallback"]),
            "fallback_reason": box_info["reason"],
            "core_bbox": None if box_info["core"] is None else [int(v) for v in box_info["core"]],
            "study_crop_bbox": [int(v) for v in box_info["boxes"]["study"]],
            "controls": per_control,
        }
    except Exception as exc:  # noqa: BLE001
        return {"idx": idx, "ok": False, "dicom_id": entry.get("dicom_id"), "error": f"{type(exc).__name__}: {exc}"}


def build_manifest_entry(entry: dict, res: Dict[str, object], control: str) -> dict:
    out = dict(entry)
    c = res["controls"][control]
    if control == "soft":
        out.pop("cropped_cxr_jpg", None)
        out["masked_cxr_jpg"] = c["path"]
        out["soft_sigma_512"] = c["soft_sigma_512"]
    else:
        out.pop("masked_cxr_jpg", None)
        out["cropped_cxr_jpg"] = c["path"]
        for k in ("control_bbox", "trim_bbox", "padding", "final_crop_size", "make_square"):
            out[k] = c[k]
    out["control_mode"] = f"medsam3_{control}"
    out["original_size"] = res["original_size"]
    out["study_crop_bbox"] = res["study_crop_bbox"]
    out["core_bbox"] = res["core_bbox"]
    out["fallback_center_crop"] = res["fallback"]
    return out


def split_dicom_ids(view: str) -> Optional[set]:
    split_dir = STUDY_DIR / "splits"
    ids = set()
    for name in (f"{view}_outer_test.csv", f"{view}_fold_assignment.csv"):
        p = split_dir / name
        if not p.is_file():
            return None
        with p.open() as f:
            header = f.readline().strip().split(",")
            col = header.index("dicom_id")
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) > col and parts[col]:
                    ids.add(parts[col])
    return ids


def run_view(view: str, controls: List[str], out_base: Path, source_root: Path,
             workers: int, limit: int, resume: bool) -> bool:
    ref_path = reference_manifest(view)
    entries = json.loads(ref_path.read_text(encoding="utf-8"))
    if limit > 0:
        entries = entries[:limit]
    n = len(entries)
    n_pos = sum(int(e.get("pneumonia") or 0) for e in entries)
    print(f"\n[{view}] reference manifest: {ref_path} ({n:,} images, pneumonia={n_pos:,})")
    print(f"[{view}] controls: {', '.join(controls)}  →  {out_base}")

    results: List[Optional[Dict[str, object]]] = [None] * n
    failures = []
    t0 = time.time()
    tasks = ((i, e, view, controls, out_base, source_root, resume) for i, e in enumerate(entries))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_entry, t) for t in tasks]
        for k, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res["ok"]:
                results[res["idx"]] = res
            else:
                failures.append(res)
            if k % 500 == 0 or k == n:
                rate = k / max(1e-6, time.time() - t0)
                eta = (n - k) / max(1e-6, rate)
                print(f"[{view}] {k:,}/{n:,}  {rate:.1f} img/s  ETA {eta / 60:.1f} min  fail={len(failures)}", flush=True)

    if failures:
        fail_path = out_base / f"control_failures_{view.lower()}.json"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        fail_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{view}] ❌ {len(failures)} images failed; manifest not written. See {fail_path}")
        for f in failures[:10]:
            print(f"   - {f['dicom_id']}: {f['error']}")
        return False

    n_fallback = sum(1 for r in results if r["fallback"])
    ref_ids = [str(e.get("dicom_id")) for e in entries]
    split_ids = split_dicom_ids(view) if limit <= 0 else None
    if split_ids is not None and set(ref_ids) != split_ids:
        print(f"[{view}] ❌ reference manifest dicom_ids differ from splits/ ({len(set(ref_ids))} vs {len(split_ids)})")
        return False

    for c in controls:
        root = control_root(out_base, c, view)
        root.mkdir(parents=True, exist_ok=True)
        out_entries = [build_manifest_entry(e, r, c) for e, r in zip(entries, results)]
        assert [str(e.get("dicom_id")) for e in out_entries] == ref_ids
        man_path = root / f"manifest_{view.lower()}_medsam3_ok.json"
        tmp = man_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out_entries, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, man_path)
        print(f"[{view}] ✅ {c:8s} manifest: {man_path} ({len(out_entries):,})")

    print(f"[{view}] fallback (80% center crop) images: {n_fallback:,}")
    if split_ids is not None:
        print(f"[{view}] dicom_id set == splits/{view}_outer_test.csv + {view}_fold_assignment.csv ✅")
    print(f"[{view}] done in {(time.time() - t0) / 60:.1f} min")
    return True


def verify_study_crop(view: str, n: int, source_root: Path) -> None:
    """재계산한 본 실험 크롭이 저장된 cxr_medsam3_lung_seg_cropped_* 영상과 같은지 확인합니다."""
    entries = json.loads(reference_manifest(view).read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(entries), size=min(n, len(entries)), replace=False)
    size_match = 0
    diffs = []
    for i in pick:
        e = entries[int(i)]
        img, _mask_bool, mask_img = load_inputs(e, source_root)
        w, h = img.size
        filtered = filter_mask_small_components(mask_img, keep_top_k=MASK_KEEP_TOPK, min_area=MASK_MIN_AREA)
        info = compute_boxes(filtered, w, h)
        mine, _ = render_crop(img, info["boxes"]["study"])
        saved = Image.open(resolve_study_path(e["cropped_cxr_jpg"]))
        saved.load()
        if mine.size == saved.size:
            size_match += 1
            a = np.asarray(mine.convert("L"), dtype=np.float32)
            b = np.asarray(saved.convert("L"), dtype=np.float32)
            diffs.append(float(np.abs(a - b).mean()))
    print(f"[{view}] study-crop reproduction: size match {size_match}/{len(pick)}; "
          f"mean |Δpixel| (0–255) median={np.median(diffs) if diffs else float('nan'):.2f} "
          f"max={max(diffs) if diffs else float('nan'):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", default="both", choices=["AP", "PA", "both"])
    ap.add_argument("--controls", nargs="+", default=list(ALL_CONTROLS), choices=list(ALL_CONTROLS))
    ap.add_argument("--out-base", type=Path, default=STUDY_DIR)
    ap.add_argument("--source-image-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    ap.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 4))
    ap.add_argument("--limit", type=int, default=0, help="앞에서부터 N장만 처리 (점검용; 0=전체)")
    ap.add_argument("--no-resume", dest="resume", action="store_false", help="이미 있는 출력도 다시 만듭니다")
    ap.add_argument("--verify-study-crop", type=int, default=0,
                    help="N장을 골라 본 실험 크롭 재현 여부만 확인하고 종료")
    args = ap.parse_args()

    views = ["AP", "PA"] if args.view == "both" else [args.view]
    if args.verify_study_crop > 0:
        for v in views:
            verify_study_crop(v, args.verify_study_crop, args.source_image_root)
        return

    ok = True
    for v in views:
        ok &= run_view(v, list(args.controls), args.out_base, args.source_image_root,
                       args.workers, args.limit, args.resume)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
