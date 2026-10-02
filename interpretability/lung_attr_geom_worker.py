"""CPU-only crop-geometry reconstruction for lung-attribution analysis.

Kept separate so process-pool workers do not import torch / pneumonia_train.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from build_cxr_lung_crop_dataset import crop_with_mask, load_npy_mask_as_pil

STUDY_CROP_KWARGS = dict(
    pad_top=0,
    pad_bottom=0,
    pad_lr=0,
    pad_frac=0.05,
    pad_top_frac=0.07,
    pad_bottom_frac=0.10,
    pad_lr_frac=0.05,
    save_mask=False,
    save_orig=False,
    make_square=True,
    mask_keep_topk=2,
    mask_min_area=20,
    bbox_mode="auto",
    bbox_min_area_ratio=0.50,
    bbox_min_width_ratio=0.70,
    bbox_min_height_ratio=0.55,
    single_comp_wide_threshold=0.55,
    tall_bbox_aspect_threshold=1.5,
    tall_bbox_min_width_frac=0.5,
    mask_quality_min_area_ratio=0.10,
    fallback_center_ratio=0.8,
    fallback_center_full_height=False,
    write_files=False,
    quiet=True,
)


def geom_job(payload: tuple):
    dicom, mode, src_path, mask_npy, crop_path = payload
    try:
        mask = load_npy_mask_as_pil(mask_npy)
        if mask is None:
            return dicom, mode, None, f"mask load failed: {mask_npy}"
        geom = crop_with_mask(
            src_path,
            "",
            crop_path,
            mask_img_override=mask,
            **STUDY_CROP_KWARGS,
        )
        if not geom:
            return dicom, mode, None, "crop_with_mask returned None"
        geom.pop("image", None)
        existing = Image.open(crop_path)
        saved = existing.size
        existing.close()
        wf, hf = [int(v) for v in geom["final_crop_size"]]
        if (wf, hf) != saved:
            return (
                dicom,
                mode,
                None,
                f"reconstructed crop size {(wf, hf)} != saved {saved}",
            )
        compact = {
            "original_size": [int(v) for v in geom["original_size"]],
            "crop_bbox": [int(v) for v in geom["crop_bbox"]],
            "trim_bbox": [int(v) for v in geom["trim_bbox"]],
            "padding": [int(v) for v in geom["padding"]],
            "final_crop_size": [int(v) for v in geom["final_crop_size"]],
        }
        return dicom, mode, compact, ""
    except Exception as exc:
        return dicom, mode, None, str(exc)
