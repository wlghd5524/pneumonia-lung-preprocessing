#!/usr/bin/env python3
"""Input-geometry analysis for raw / crop / mask and extra untrained controls."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
PREPROCESSING_DIR = PACKAGE_ROOT / "preprocessing"
DATA_BASE_DIR = Path(os.environ.get("DATA_BASE_DIR", PACKAGE_ROOT)).resolve()
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", PACKAGE_ROOT / "results_pneumonia")).resolve()
sys.path.insert(0, str(PREPROCESSING_DIR))

from build_cxr_lung_crop_dataset import (  # noqa: E402
    _center_crop_bbox,
    check_mask_quality,
    compute_bbox_from_mask_info,
    filter_mask_small_components,
)

INPUT_SIZE = 512
QUALITY_MIN = 0.10
BBOX_KW = dict(
    bbox_mode="auto",
    bbox_min_area_ratio=0.50,
    bbox_min_width_ratio=0.70,
    bbox_min_height_ratio=0.55,
    single_comp_wide_threshold=0.55,
    tall_bbox_aspect_threshold=1.5,
    tall_bbox_min_width_frac=0.5,
)
OUT_JSON = RESULTS_DIR / "input_geometry_summary.json"
OUT_MD = RESULTS_DIR / "Supplementary_Table_input_geometry.md"
OUT_METHODS = RESULTS_DIR / "Methods_crop_vs_mask_and_geometry.md"
FIG_DIR = RESULTS_DIR / "geometry_control_examples"

MANIFESTS = {
    ("MedSAM3", "AP"): DATA_BASE_DIR / "cxr_medsam3_lung_seg_ap" / "manifest_ap_medsam3_ok.json",
    ("MedSAM3", "PA"): DATA_BASE_DIR / "cxr_medsam3_lung_seg_pa" / "manifest_pa_medsam3_ok.json",
    ("CheXMask", "AP"): DATA_BASE_DIR / "cxr_chexmask_lung_seg_ap" / "manifest_ap_chexmask_ok.json",
    ("CheXMask", "PA"): DATA_BASE_DIR / "cxr_chexmask_lung_seg_pa" / "manifest_pa_chexmask_ok.json",
}


def remap(raw: str) -> Path:
    text = str(raw)
    text = text.replace("/mnt/e/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0", str(DATA_BASE_DIR.parent / "mimic-cxr-jpg" / "2.1.0"))
    return Path(text)


def clamp_bbox(bbox, w: int, h: int):
    left, upper, right, lower = [int(v) for v in bbox]
    left = max(0, min(w - 1, left))
    upper = max(0, min(h - 1, upper))
    right = max(left + 1, min(w, right))
    lower = max(upper + 1, min(h, lower))
    return left, upper, right, lower


def union_bbox(a, b, w, h):
    if b is None:
        return clamp_bbox(a, w, h)
    return clamp_bbox((min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])), w, h)


def pad_bbox(core, w: int, h: int, pad_top: int, pad_bottom: int, pad_lr: int):
    left, upper, right, lower = core
    return clamp_bbox((left - pad_lr, upper - pad_top, right + pad_lr, lower + pad_bottom), w, h)


def core_bbox(mask_img: Image.Image, w: int, h: int):
    """One connected-component pass; paddings are applied afterwards."""
    is_good, reason = check_mask_quality(mask_img, (w, h), min_area_ratio=QUALITY_MIN)
    if not is_good:
        bbox = clamp_bbox(_center_crop_bbox((w, h), crop_ratio=0.8, full_height=False), w, h)
        return bbox, True, reason
    bbox, info = compute_bbox_from_mask_info(
        mask_img, pad_top=0, pad_bottom=0, pad_lr=0, **BBOX_KW,
    )
    if bbox is None:
        bbox = clamp_bbox(_center_crop_bbox((w, h), crop_ratio=0.8, full_height=False), w, h)
        return bbox, True, "no_bbox"
    return union_bbox(bbox, info.get("mask_bbox"), w, h), False, str(info.get("reason") or "ok")


def to512(mask_bool: np.ndarray) -> np.ndarray:
    if mask_bool.size == 0:
        return np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
    img = Image.fromarray(mask_bool.astype(np.uint8) * 255)
    return (np.array(img.resize((INPUT_SIZE, INPUT_SIZE), resample=Image.Resampling.NEAREST)) > 0).astype(np.uint8)


def occ512(mask_bool: np.ndarray) -> float:
    return float(to512(mask_bool).mean())


def crop_mask(mask_bool: np.ndarray, bbox) -> np.ndarray:
    left, upper, right, lower = bbox
    return mask_bool[upper:lower, left:right]


def matched_center_bbox(w: int, h: int, crop_w: int, crop_h: int):
    crop_w = max(1, min(w, crop_w))
    crop_h = max(1, min(h, crop_h))
    left = max(0, (w - crop_w) // 2)
    upper = max(0, (h - crop_h) // 2)
    return clamp_bbox((left, upper, left + crop_w, upper + crop_h), w, h)


def analyze_one(rec: dict) -> dict:
    npy = remap(rec["lung_mask_npy"])
    view = rec["view_position"]
    source = rec["source"]
    if not npy.is_file():
        return {"source": source, "view": view, "ok": False}
    mask = np.asarray(np.load(npy)).astype(bool)
    h, w = mask.shape
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_img = filter_mask_small_components(mask_img, keep_top_k=2, min_area=20)
    mask = np.array(mask_img.convert("L")) > 0
    orig_area = float(w * h)
    orig_aspect = float(w / h) if h else float("nan")

    core, fallback, _reason = core_bbox(mask_img, w, h)
    bbox = pad_bbox(core, w, h, int(round(0.07 * h)), int(round(0.10 * h)), int(round(0.05 * w)))
    left, upper, right, lower = bbox
    crop_w, crop_h = right - left, lower - upper
    crop_area_frac = float(crop_w * crop_h) / orig_area
    crop_aspect = float(crop_w / crop_h) if crop_h else float("nan")

    raw512 = to512(mask)
    raw_occ = float(raw512.mean())
    crop_occ = occ512(crop_mask(mask, bbox))
    mag = float(crop_occ / raw_occ) if raw_occ > 0 else float("nan")
    mask_zero = 1.0 - raw_occ

    ctr = matched_center_bbox(w, h, crop_w, crop_h)
    center_area_frac = float((ctr[2] - ctr[0]) * (ctr[3] - ctr[1])) / orig_area
    center_occ = occ512(crop_mask(mask, ctr))

    margin = {}
    for pct in (0.0, 0.10, 0.20):
        bb = pad_bbox(core, w, h, int(round(pct * h)), int(round(pct * h)), int(round(pct * w)))
        cw, ch = bb[2] - bb[0], bb[3] - bb[1]
        margin[pct] = {
            "crop_area_frac": float(cw * ch) / orig_area,
            "lung_occ_512": occ512(crop_mask(mask, bb)),
        }

    # Soft mask on the 512 grid (same final input size). Sigma ≈ 2% of 512.
    soft512 = gaussian_filter(raw512.astype(np.float32), sigma=0.02 * INPUT_SIZE)
    soft_occ = float(soft512.mean())
    soft_zero_exact = float((soft512 == 0).mean())
    soft_bg = 1.0 - soft_occ

    return {
        "source": source,
        "view": view,
        "ok": True,
        "orig_aspect": orig_aspect,
        "crop_area_frac": crop_area_frac,
        "crop_aspect": crop_aspect,
        "raw_lung_occ_512": raw_occ,
        "crop_lung_occ_512": crop_occ,
        "lung_magnification": mag,
        "hard_mask_zero_frac_512": mask_zero,
        "fallback": fallback,
        "center_area_frac": center_area_frac,
        "center_lung_occ_512": center_occ,
        "margin0_area_frac": margin[0.0]["crop_area_frac"],
        "margin0_lung_occ_512": margin[0.0]["lung_occ_512"],
        "margin10_area_frac": margin[0.10]["crop_area_frac"],
        "margin10_lung_occ_512": margin[0.10]["lung_occ_512"],
        "margin20_area_frac": margin[0.20]["crop_area_frac"],
        "margin20_lung_occ_512": margin[0.20]["lung_occ_512"],
        "soft_lung_occ_512": soft_occ,
        "soft_exact_zero_frac_512": soft_zero_exact,
        "soft_background_frac_512": soft_bg,
    }


def miqr(vals) -> str:
    x = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if x.size == 0:
        return "n/a"
    med = float(np.median(x))
    lo = float(np.percentile(x, 25))
    hi = float(np.percentile(x, 75))
    return f"{med:.3f} [{lo:.3f}–{hi:.3f}]"


def collect(rows, source, view, key):
    return [r[key] for r in rows if r.get("ok") and r["source"] == source and r["view"] == view]


def main() -> None:
    records = []
    for (source, _view), path in MANIFESTS.items():
        items = json.loads(path.read_text())
        print(f"loaded {source} {_view} {len(items)}")
        for it in items:
            it = dict(it)
            it["source"] = source
            records.append(it)

    rows = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(analyze_one, rec) for rec in records]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 4000 == 0:
                print(f"  {i}/{len(records)}")

    ok = [r for r in rows if r.get("ok")]
    print(f"ok {len(ok)}/{len(rows)}")

    metrics = [
        ("Original aspect ratio (W/H)", "orig_aspect"),
        ("Study crop area / original area", "crop_area_frac"),
        ("Study crop aspect ratio (W/H)", "crop_aspect"),
        ("Raw lung occupancy in 512²", "raw_lung_occ_512"),
        ("Study crop lung occupancy in 512²", "crop_lung_occ_512"),
        ("Lung-area magnification (crop / raw)", "lung_magnification"),
        ("Hard-mask zero-pixel fraction in 512²", "hard_mask_zero_frac_512"),
        ("Matched center-crop area / original area", "center_area_frac"),
        ("Matched center-crop lung occupancy in 512²", "center_lung_occ_512"),
        ("0% margin crop area / original area", "margin0_area_frac"),
        ("0% margin lung occupancy in 512²", "margin0_lung_occ_512"),
        ("10% margin crop area / original area", "margin10_area_frac"),
        ("10% margin lung occupancy in 512²", "margin10_lung_occ_512"),
        ("20% margin crop area / original area", "margin20_area_frac"),
        ("20% margin lung occupancy in 512²", "margin20_lung_occ_512"),
        ("Soft-mask lung occupancy in 512²", "soft_lung_occ_512"),
        ("Soft-mask exact-zero fraction in 512²", "soft_exact_zero_frac_512"),
        ("Soft-mask background fraction (1 − mean α)", "soft_background_frac_512"),
    ]

    summary = {"n_ok": len(ok), "n_total": len(rows), "input_size": INPUT_SIZE, "groups": {}}
    lines = [
        "# Input geometry after 512×512 resize",
        "",
        "All classifier inputs are bilinearly resized to 512×512. "
        "The study crop uses the lung mask only to set a padded rectangular window "
        "(top 7%, bottom 10%, left/right 5%); pixels inside that rectangle are not suppressed. "
        "Hard masking sets non-lung pixels to 0 on the full radiograph. "
        "Extra controls (matched center crop, uniform 0/10/20% margin, Gaussian-blurred soft mask) "
        "were generated in memory for geometry comparison only; no classifier was retrained. "
        "Values are median [IQR].",
        "",
        "| Mask source | View | Metric | Median [IQR] | N |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for source in ("MedSAM3", "CheXMask"):
        for view in ("AP", "PA"):
            n = sum(1 for r in ok if r["source"] == source and r["view"] == view)
            block = {"n": n}
            for label, key in metrics:
                vals = [
                    value
                    for value in collect(ok, source, view, key)
                    if value is not None and np.isfinite(value)
                ]
                block[key] = {
                    "median": float(np.median(vals)) if vals else None,
                    "q1": float(np.percentile(vals, 25)) if vals else None,
                    "q3": float(np.percentile(vals, 75)) if vals else None,
                }
                lines.append(f"| {source} | {view} | {label} | {miqr(vals)} | {n:,} |")
            summary["groups"][f"{source}_{view}"] = block

    lines += [
        "",
        "Study crop = current MedSAM3/CheXMask crop used in the 70 trained configurations. "
        "Matched center crop uses the same rectangle size placed at the image center. "
        "Soft mask uses a Gaussian blur with radius 2% of the shorter image side.",
        "",
    ]
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {OUT_MD}")
    write_methods()
    make_examples()


def write_methods() -> None:
    OUT_METHODS.write_text(
        """# Crop vs mask, geometry controls, and causal wording

## Rectangular bounding-box control (already in the study)

The MedSAM3 crop and CheXMask crop conditions already implement rectangular lung bounding-box inputs without pixel-wise mask suppression. The lung mask is used only to locate a padded bounding box (top 7%, bottom 10%, left/right 5% of image height/width, with the documented imbalance and fallback rules). The radiograph is then cropped to that rectangle. No non-lung pixel is set to zero inside the cropped field of view.

Hard masking is the only condition that applies pixel-wise suppression: all pixels outside the lung mask are set to 0 on the full image (`out[~mask_bool] = 0`).

Suggested Methods sentence:

> For cropping, the mask was used only to determine the crop geometry; no pixel-wise mask suppression was applied within the rectangular cropped input. Hard masking, by contrast, set all non-lung pixels to zero before resizing to 512×512.

## Geometry analysis (no retraining)

Every trained model receives a 512×512 bilinear resize. We therefore measured, on the full 41,503-image cohort and separately for AP and PA, how raw, cropped, and masked inputs change field of view and effective lung scale: crop area / original area, pre-crop and crop aspect ratio, lung pixels / 512², lung-area magnification relative to raw, and the zero-pixel fraction of hard-masked 512×512 inputs. The same quantities were computed for three untrained controls: a segmentation-independent matched center crop of the same rectangle size, uniform 0%/10%/20% margin crops, and a Gaussian-blurred soft mask.

Suggested reviewer sentence:

> We added quantitative controls to characterize how the transformations changed the input geometry, although additional classifier retraining for these controls was outside the scope of the revision. Retraining a new model for each extra transform would not be comparable to simply feeding a new input into one of the existing 70 trained configurations.

## Causal wording

The experiments show that cropping keeps continuous image texture while narrowing the field of view, whereas hard masking introduces a zero-valued background and sharp mask boundaries. They do not isolate which of those factors produced the AUROC change.

Do not write that hard masking caused degradation because of artificial boundaries.

Preferred wording:

> The observed degradation may reflect a combination of anatomical exclusion, zero-valued background, boundary artifacts, and differences in retained contextual information. These mechanisms were not isolated by the present design.
""",
        encoding="utf-8",
    )
    print(f"saved {OUT_METHODS}")


def make_examples() -> None:
    """Save a small visual panel; full-cohort image dumps were not written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    picks = []
    for path, view in (
        (DATA_BASE_DIR / "cxr_medsam3_lung_seg_ap" / "manifest_ap_medsam3_ok.json", "AP"),
        (DATA_BASE_DIR / "cxr_medsam3_lung_seg_pa" / "manifest_pa_medsam3_ok.json", "PA"),
    ):
        for rec in json.loads(path.read_text()):
            if float(rec.get("lung_area_ratio") or 0) < 0.15:
                continue
            src = remap(rec["source_image_abs_path"])
            npy = remap(rec["lung_mask_npy"])
            if src.is_file() and npy.is_file():
                picks.append((view, rec, src, npy))
                break
    if not picks:
        print("no example source images found")
        return

    fig, axes = plt.subplots(len(picks), 6, figsize=(14.5, 4.6 * len(picks)))
    if len(picks) == 1:
        axes = np.array([axes])
    titles = ["Raw", "Hard mask", "Study crop", "Matched center", "Soft mask", "0% / 20% margin"]
    for row, (view, rec, src, npy) in enumerate(picks):
        rgb = np.array(Image.open(src).convert("RGB"))
        mask = np.asarray(np.load(npy)).astype(bool)
        if mask.shape != rgb.shape[:2]:
            mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)) > 0
        h, w = mask.shape
        mask_img = filter_mask_small_components(Image.fromarray(mask.astype(np.uint8) * 255), keep_top_k=2, min_area=20)
        mask = np.array(mask_img.convert("L")) > 0
        core, _, _ = core_bbox(mask_img, w, h)
        bbox = pad_bbox(core, w, h, int(round(0.07 * h)), int(round(0.10 * h)), int(round(0.05 * w)))
        hard = rgb.copy()
        hard[~mask] = 0
        crop = rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        ctr = matched_center_bbox(w, h, bbox[2] - bbox[0], bbox[3] - bbox[1])
        center = rgb[ctr[1]:ctr[3], ctr[0]:ctr[2]]
        alpha = gaussian_filter(mask.astype(np.float32), sigma=0.02 * min(w, h))
        alpha = np.clip(alpha, 0, 1)
        soft = (rgb.astype(np.float32) * alpha[..., None]).astype(np.uint8)
        bb0 = pad_bbox(core, w, h, 0, 0, 0)
        bb20 = pad_bbox(core, w, h, int(round(0.20 * h)), int(round(0.20 * h)), int(round(0.20 * w)))
        vis = rgb.copy()
        # draw 0% red and 20% cyan boxes
        vis[bb0[1]:bb0[1] + 4, bb0[0]:bb0[2]] = (220, 40, 40)
        vis[bb0[3] - 4:bb0[3], bb0[0]:bb0[2]] = (220, 40, 40)
        vis[bb20[1]:bb20[1] + 4, bb20[0]:bb20[2]] = (40, 160, 220)
        vis[bb20[3] - 4:bb20[3], bb20[0]:bb20[2]] = (40, 160, 220)
        panels = [rgb, hard, crop, center, soft, vis]
        for col, im in enumerate(panels):
            axes[row, col].imshow(im)
            axes[row, col].set_title(titles[col] if row == 0 else "", fontsize=10)
            axes[row, col].set_axis_off()
        axes[row, 0].set_ylabel(f"{view}  {rec['dicom_id'][:8]}", fontsize=9)
    fig.suptitle("Geometry controls (examples only; no classifier retraining)", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "Figure_geometry_controls_examples.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
