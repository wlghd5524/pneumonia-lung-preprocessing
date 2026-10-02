#!/usr/bin/env python3
"""Build Mask source × View × Pneumonia failure/fallback table."""

from __future__ import annotations

import os

import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

STUDY_DIR = Path(__file__).resolve().parents[1]
DATA_BASE_DIR = Path(os.environ.get("DATA_BASE_DIR", STUDY_DIR)).resolve()
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", STUDY_DIR / "results_pneumonia")).resolve()
sys.path.insert(0, str(STUDY_DIR / "preprocessing"))

from build_cxr_lung_crop_dataset import (  # noqa: E402
    check_mask_quality,
    filter_mask_small_components,
)

QUALITY_MIN_AREA = 0.10
OUT_JSON = RESULTS_DIR / "paper_tables" / "Supplementary_Table_mask_failure_fallback.json"
OUT_MD = RESULTS_DIR / "paper_tables" / "Supplementary_Table_mask_failure_fallback.md"

MANIFESTS = {
    ("CheXMask", "AP"): DATA_BASE_DIR / "cxr_chexmask_lung_seg_ap" / "manifest_ap_chexmask_ok.json",
    ("CheXMask", "PA"): DATA_BASE_DIR / "cxr_chexmask_lung_seg_pa" / "manifest_pa_chexmask_ok.json",
}

# MedSAM3 was already recomputed on the same 41,503-image cohort.
MEDSAM3_FIXED = {
    ("MedSAM3", "AP", "Negative"): {"n": 11515, "fail": 0, "fallback": 43},
    ("MedSAM3", "AP", "Positive"): {"n": 11379, "fail": 2, "fallback": 67},
    ("MedSAM3", "PA", "Negative"): {"n": 12781, "fail": 0, "fallback": 2},
    ("MedSAM3", "PA", "Positive"): {"n": 5828, "fail": 0, "fallback": 3},
}


def remap_mask_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_file():
        return p
    text = raw.replace("/mnt/d/CXR-Sepsis-Prediction/IEEE_ICCBE", str(DATA_BASE_DIR))
    return Path(text)


def pipeline_fallback_used(npy_raw: str, area_ratio: float) -> bool:
    """Same crop-pipeline fallback rule as Panel B / the lung-crop builder."""
    npy = remap_mask_path(str(npy_raw or ""))
    area = float(area_ratio or 0.0)
    if not npy.is_file():
        return area < QUALITY_MIN_AREA
    arr = np.asarray(np.load(npy)).astype(bool)
    h, w = arr.shape
    mask_img = Image.fromarray((arr.astype(np.uint8) * 255))
    mask_img = filter_mask_small_components(mask_img, keep_top_k=2, min_area=20)
    is_good, _reason = check_mask_quality(mask_img, (w, h), min_area_ratio=QUALITY_MIN_AREA)
    return not is_good


def pipeline_fallback_pair(payload: dict) -> dict:
    return {
        "dicom_id": payload["dicom_id"],
        "fallback_medsam": pipeline_fallback_used(payload["medsam_npy"], payload.get("medsam_area") or 0.0),
        "fallback_chex": pipeline_fallback_used(payload["chex_npy"], payload.get("chex_area") or 0.0),
    }


def analyze_one(payload: dict) -> dict:
    area = float(payload.get("lung_area_ratio") or 0.0)
    n_det = payload.get("num_detections")
    empty = area == 0.0
    zero_det = (n_det is not None) and int(n_det) == 0
    rec = {
        "source": payload["source"],
        "view": payload["view_position"],
        "pneumonia": int(payload["pneumonia"]),
        "mask_failure": bool(empty or zero_det),
        "fallback_used": pipeline_fallback_used(
            str(payload.get("lung_mask_npy") or ""),
            area,
        ),
    }
    return rec


def _fmt(n: int, d: int) -> str:
    return f"{n} ({100.0 * n / d:.2f}%)" if d else "0 (n/a)"


def main() -> None:
    records = []
    for (source, _view), path in MANIFESTS.items():
        items = json.loads(path.read_text())
        print(f"loaded {source} {_view} {len(items)}")
        for item in items:
            item = dict(item)
            item["source"] = source
            records.append(item)

    rows = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(analyze_one, rec) for rec in records]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 4000 == 0:
                print(f"  processed {i}/{len(records)}")

    agg = defaultdict(lambda: {"n": 0, "fail": 0, "fallback": 0})
    agg.update({k: dict(v) for k, v in MEDSAM3_FIXED.items()})
    for r in rows:
        pna = "Positive" if r["pneumonia"] == 1 else "Negative"
        key = (r["source"], r["view"], pna)
        agg[key]["n"] += 1
        agg[key]["fail"] += int(r["mask_failure"])
        agg[key]["fallback"] += int(r["fallback_used"])

    order = [
        ("MedSAM3", "AP", "Negative"),
        ("MedSAM3", "AP", "Positive"),
        ("MedSAM3", "PA", "Negative"),
        ("MedSAM3", "PA", "Positive"),
        ("CheXMask", "AP", "Negative"),
        ("CheXMask", "AP", "Positive"),
        ("CheXMask", "PA", "Negative"),
        ("CheXMask", "PA", "Positive"),
    ]
    table_rows = []
    for key in order:
        g = agg[key]
        table_rows.append({
            "Mask source": key[0],
            "View": key[1],
            "Pneumonia": key[2],
            "N images": g["n"],
            "Mask failure, n": g["fail"],
            "Mask failure, %": round(100.0 * g["fail"] / g["n"], 2) if g["n"] else None,
            "Center-crop fallback, n": g["fallback"],
            "Center-crop fallback, %": round(100.0 * g["fallback"] / g["n"], 2) if g["n"] else None,
        })

    OUT_JSON.write_text(json.dumps({"rows": table_rows, "n_total_pairs": len(rows)}, indent=2), encoding="utf-8")

    lines = [
        "# Mask failure and center-crop fallback, by source, view, and pneumonia label",
        "",
        "The table is restricted to the 41,503-image analysis cohort (22,894 AP + 18,609 PA). "
        "Mask failure is an empty lung mask, or, for MedSAM3, zero detections after the 0.5 confidence filter and NMS. "
        "Center-crop fallback is an empty mask or a mask covering <10% of the image after keeping at most two 4-connected components and removing components <20 pixels; those images received an 80% × 80% center crop. "
        "The two MedSAM3 empty masks are a subset of the fallback counts.",
        "",
        "CheXMask mask failure is 0 **in this analysis cohort** because five AP radiographs with missing "
        "CheXMask lung RLE were excluded from the entire study before the cohort was fixed "
        "(2 pneumonia-negative, 3 pneumonia-positive). Those five images are not in *N* and are therefore "
        "not counted as CheXMask failures in the table. This is not evidence that CheXMask succeeded on "
        "every original radiograph. If the five missing-RLE images are counted as CheXMask failures before "
        "cohort restriction, CheXMask AP-negative failure is 2 / 11,517 and AP-positive failure is 3 / 11,382.",
        "",
        "| Mask source | View | Pneumonia | N images | Mask failure, n (%) | Center-crop fallback, n (%) |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in table_rows:
        lines.append(
            f"| {r['Mask source']} | {r['View']} | {r['Pneumonia']} | {r['N images']:,} | "
            f"{_fmt(r['Mask failure, n'], r['N images'])} | "
            f"{_fmt(r['Center-crop fallback, n'], r['N images'])} |"
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {OUT_MD}")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
