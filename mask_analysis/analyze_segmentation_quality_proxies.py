#!/usr/bin/env python3
"""Outer-test proxy analysis: mask agreement / fallback / RCA vs classification error."""

from __future__ import annotations

import os

import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

STUDY_DIR = Path(__file__).resolve().parents[1]
DATA_BASE_DIR = Path(os.environ.get("DATA_BASE_DIR", STUDY_DIR)).resolve()
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", STUDY_DIR / "results_pneumonia")).resolve()
RESULTS = RESULTS_DIR
SPLITS = STUDY_DIR / "splits"
OUT_MD = RESULTS / "paper_tables" / "Supplementary_Table_segmentation_quality_proxies.md"
OUT_JSON = RESULTS / "paper_tables" / "segmentation_quality_proxies.json"
OUT_CSV = RESULTS / "paper_tables" / "segmentation_quality_proxies.csv"

EVA_RUNS = {
    ("AP", "raw"): "20260909_184003_eva_x_base_AP_raw_cv_5fold_holdout15pct",
    ("AP", "medsam3_seg"): "20260910_151857_eva_x_base_AP_medsam3_seg_cv_5fold_holdout15pct",
    ("AP", "chexmask_seg"): "20260911_234000_eva_x_base_AP_chexmask_seg_cv_5fold_holdout15pct",
    ("AP", "medsam3_crop"): "20260911_103605_eva_x_base_AP_medsam3_crop_cv_5fold_holdout15pct",
    ("AP", "chexmask_crop"): "20260912_120123_eva_x_base_AP_chexmask_crop_cv_5fold_holdout15pct",
    ("PA", "raw"): "20260913_004127_eva_x_base_PA_raw_cv_5fold_holdout15pct",
    ("PA", "medsam3_seg"): "20260908_015426_eva_x_base_PA_medsam3_seg_cv_5fold_holdout15pct",
    ("PA", "chexmask_seg"): "20260909_084516_eva_x_base_PA_chexmask_seg_cv_5fold_holdout15pct",
    ("PA", "medsam3_crop"): "20260908_172509_eva_x_base_PA_medsam3_crop_cv_5fold_holdout15pct",
    ("PA", "chexmask_crop"): "20260913_113626_eva_x_base_PA_chexmask_crop_cv_5fold_holdout15pct",
}


def remap(raw: str) -> Path:
    text = str(raw).replace("/mnt/d/CXR-Sepsis-Prediction/IEEE_ICCBE", str(DATA_BASE_DIR))
    return Path(text)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    if a.shape != b.shape:
        b = np.array(Image.fromarray(b.astype(np.uint8) * 255).resize((a.shape[1], a.shape[0]), Image.Resampling.NEAREST)) > 0
    inter = int(np.logical_and(a, b).sum())
    s = int(a.sum()) + int(b.sum())
    return float(2 * inter / s) if s else float("nan")


def pair_dice(payload: dict) -> dict:
    mp = remap(payload["medsam_npy"])
    cp = remap(payload["chex_npy"])
    if not mp.is_file() or not cp.is_file():
        return {**payload, "pair_dice": float("nan")}
    d = dice(np.load(mp), np.load(cp))
    return {**payload, "pair_dice": d}


def load_manifest(path: Path) -> dict:
    return {r["dicom_id"]: r for r in json.loads(path.read_text())}


def load_outer(view: str) -> list[dict]:
    rows = list(csv.DictReader((SPLITS / f"{view}_outer_test.csv").open(encoding="utf-8")))
    for r in rows:
        r["view"] = view
        r["label"] = int(r["label"])
    return rows


def load_preds(run: str) -> dict:
    path = RESULTS / run / "outer_test" / "outer_test_predictions.csv"
    out = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dicom = Path(r["img_path"]).stem
            out[dicom] = float(r["prob_ensemble"])
    return out


def err_rate(rows: list[dict], key: str) -> tuple[int, int, float]:
    use = [r for r in rows if key in r and r[key] is not None]
    n = len(use)
    e = sum(int(((r[key] >= 0.5) != bool(r["label"]))) for r in use)
    return e, n, (e / n if n else float("nan"))


def fmt(e: int, n: int, p: float) -> str:
    return f"{e}/{n} ({100 * p:.1f}%)" if n else "n/a"


def main() -> None:
    medsam = {}
    medsam.update(load_manifest(DATA_BASE_DIR / "cxr_medsam3_lung_seg_ap" / "manifest_ap_medsam3_ok.json"))
    medsam.update(load_manifest(DATA_BASE_DIR / "cxr_medsam3_lung_seg_pa" / "manifest_pa_medsam3_ok.json"))
    chex = {}
    chex.update(load_manifest(DATA_BASE_DIR / "cxr_chexmask_lung_seg_ap" / "manifest_ap_chexmask_ok.json"))
    chex.update(load_manifest(DATA_BASE_DIR / "cxr_chexmask_lung_seg_pa" / "manifest_pa_chexmask_ok.json"))

    items = []
    for view in ("AP", "PA"):
        for r in load_outer(view):
            d = r["dicom_id"]
            if d not in medsam or d not in chex:
                continue
            items.append({
                "dicom_id": d,
                "view": view,
                "label": r["label"],
                "medsam_npy": medsam[d]["lung_mask_npy"],
                "chex_npy": chex[d]["lung_mask_npy"],
                "medsam_area": float(medsam[d].get("lung_area_ratio") or 0),
                "chex_area": float(chex[d].get("lung_area_ratio") or 0),
                "chex_rca": chex[d].get("chexmask_dice_mean"),
                "medsam_ndet": medsam[d].get("num_detections"),
                "fallback_medsam": float(medsam[d].get("lung_area_ratio") or 0) < 0.10,
                "fallback_chex": float(chex[d].get("lung_area_ratio") or 0) < 0.10,
            })

    print(f"outer-test images with both masks: {len(items)}")
    rows = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(pair_dice, it) for it in items]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 1000 == 0:
                print(f"  dice {i}/{len(items)}")

    preds = {k: load_preds(v) for k, v in EVA_RUNS.items()}
    for r in rows:
        for mode in ("raw", "medsam3_seg", "chexmask_seg"):
            r[f"prob_{mode}"] = preds[(r["view"], mode)].get(r["dicom_id"])

    def bin_dice(d):
        if not np.isfinite(d):
            return "missing"
        if d >= 0.90:
            return "Dice ≥ 0.90"
        if d >= 0.80:
            return "0.80–0.90"
        return "Dice < 0.80"

    tables = {}
    for view in ("AP", "PA", "Overall"):
        sub = rows if view == "Overall" else [r for r in rows if r["view"] == view]
        by = defaultdict(list)
        for r in sub:
            by[bin_dice(r["pair_dice"])].append(r)
        tables[view] = {}
        for name in ("Dice ≥ 0.90", "0.80–0.90", "Dice < 0.80"):
            grp = by[name]
            tables[view][name] = {
                "n": len(grp),
                "raw": err_rate(grp, "prob_raw"),
                "medsam3_seg": err_rate(grp, "prob_medsam3_seg"),
                "chexmask_seg": err_rate(grp, "prob_chexmask_seg"),
            }
        tables[view]["fallback_any"] = {}
        for status, grp in (
            ("yes", [r for r in sub if r["fallback_medsam"] or r["fallback_chex"]]),
            ("no", [r for r in sub if not r["fallback_medsam"] and not r["fallback_chex"]]),
        ):
            tables[view]["fallback_any"][status] = {
                "n": len(grp),
                "raw": err_rate(grp, "prob_raw"),
                "medsam3_seg": err_rate(grp, "prob_medsam3_seg"),
                "chexmask_seg": err_rate(grp, "prob_chexmask_seg"),
            }
        tables[view]["fallback_by_pneumonia"] = {}
        for pna_name, pred in (("Negative", 0), ("Positive", 1)):
            pna_rows = [r for r in sub if int(r["label"]) == pred]
            tables[view]["fallback_by_pneumonia"][pna_name] = {"n": len(pna_rows)}
            for status, grp in (
                ("yes", [r for r in pna_rows if r["fallback_medsam"] or r["fallback_chex"]]),
                ("no", [r for r in pna_rows if not r["fallback_medsam"] and not r["fallback_chex"]]),
            ):
                tables[view]["fallback_by_pneumonia"][pna_name][status] = {
                    "n": len(grp),
                    "raw": err_rate(grp, "prob_raw"),
                    "medsam3_seg": err_rate(grp, "prob_medsam3_seg"),
                    "chexmask_seg": err_rate(grp, "prob_chexmask_seg"),
                }
        tables[view]["fallback_frequency_by_source"] = {}
        for source, key in (("MedSAM3", "fallback_medsam"), ("CheXMask", "fallback_chex")):
            tables[view]["fallback_frequency_by_source"][source] = {}
            for pna_name, pred in (("Negative", 0), ("Positive", 1)):
                pna_rows = [r for r in sub if int(r["label"]) == pred]
                n = len(pna_rows)
                fb = sum(int(bool(r[key])) for r in pna_rows)
                tables[view]["fallback_frequency_by_source"][source][pna_name] = {
                    "n": n,
                    "fallback": fb,
                    "fallback_rate": (fb / n if n else float("nan")),
                }

    payload = {
        "n_outer_with_both_masks": len(rows),
        "mean_pair_dice": float(np.nanmean([r["pair_dice"] for r in rows])),
        "classifier": "EVA-X-Base ensemble, threshold 0.5",
        "note": "MedSAM3–CheXMask Dice is a consistency proxy, not expert accuracy.",
        "tables": tables,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Segmentation-quality proxies vs EVA-X-Base outer-test error",
        "",
        "Cross-source Dice is **not** expert ground-truth accuracy. "
        "Error is EVA-X-Base ensemble probability vs the pneumonia label at threshold 0.5. "
        f"N = {len(rows)} outer-test images with both stored masks.",
        "",
        "| View | MedSAM3–CheXMask Dice | N | Raw error | MedSAM3-mask model error | CheXMask-mask model error |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for view in ("AP", "PA", "Overall"):
        for name in ("Dice ≥ 0.90", "0.80–0.90", "Dice < 0.80"):
            t = tables[view][name]
            lines.append(
                f"| {view} | {name} | {t['n']} | {fmt(*t['raw'])} | {fmt(*t['medsam3_seg'])} | {fmt(*t['chexmask_seg'])} |"
            )
    lines += [
        "",
        "Outer-test fallback vs classification error, using the same crop-pipeline "
        "rule as the full-cohort frequency table (connected-component filter, then empty "
        "or lung area < 10%). An image is Fallback = Yes if either mask source met the rule:",
        "",
        "| View | Pneumonia | Fallback | N | Raw error | MedSAM3-mask model error | CheXMask-mask model error |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for view in ("AP", "PA", "Overall"):
        for pna_name in ("Negative", "Positive"):
            fb = tables[view]["fallback_by_pneumonia"][pna_name]
            for status, label in (("no", "No"), ("yes", "Yes")):
                t = fb[status]
                lines.append(
                    f"| {view} | {pna_name} | {label} | {t['n']} | {fmt(*t['raw'])} | "
                    f"{fmt(*t['medsam3_seg'])} | {fmt(*t['chexmask_seg'])} |"
                )
        fb = tables[view]["fallback_any"]
        for status, label in (("no", "No"), ("yes", "Yes")):
            t = fb[status]
            lines.append(
                f"| {view} | All | {label} | {t['n']} | {fmt(*t['raw'])} | "
                f"{fmt(*t['medsam3_seg'])} | {fmt(*t['chexmask_seg'])} |"
            )
    lines += [
        "",
        "Outer-test fallback frequency by mask source, view, and pneumonia class "
        "(same crop-pipeline rule as the full-cohort frequency table):",
        "",
        "| Mask source | View | Pneumonia | N images | Center-crop fallback, n (%) |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for source in ("MedSAM3", "CheXMask"):
        for view in ("AP", "PA"):
            for pna_name in ("Negative", "Positive"):
                t = tables[view]["fallback_frequency_by_source"][source][pna_name]
                n = t["n"]
                fb = t["fallback"]
                pct = f"{fb} ({100.0 * fb / n:.2f}%)" if n else "0 (n/a)"
                lines.append(f"| {source} | {view} | {pna_name} | {n} | {pct} |")
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    csv_rows = []
    for view in ("AP", "PA", "Overall"):
        for name in ("Dice ≥ 0.90", "0.80–0.90", "Dice < 0.80"):
            t = tables[view][name]
            csv_rows.append({
                "analysis": "cross_source_dice",
                "view": view,
                "pneumonia": "All",
                "stratum": name,
                "n": t["n"],
                "raw_errors": t["raw"][0],
                "raw_error_rate": t["raw"][2],
                "medsam3_mask_errors": t["medsam3_seg"][0],
                "medsam3_mask_error_rate": t["medsam3_seg"][2],
                "chexmask_mask_errors": t["chexmask_seg"][0],
                "chexmask_mask_error_rate": t["chexmask_seg"][2],
            })
        for pna_name in ("Negative", "Positive"):
            fb = tables[view]["fallback_by_pneumonia"][pna_name]
            for status, label in (("no", "No"), ("yes", "Yes")):
                t = fb[status]
                csv_rows.append({
                    "analysis": "center_crop_fallback_any_source",
                    "view": view,
                    "pneumonia": pna_name,
                    "stratum": label,
                    "n": t["n"],
                    "raw_errors": t["raw"][0],
                    "raw_error_rate": t["raw"][2],
                    "medsam3_mask_errors": t["medsam3_seg"][0],
                    "medsam3_mask_error_rate": t["medsam3_seg"][2],
                    "chexmask_mask_errors": t["chexmask_seg"][0],
                    "chexmask_mask_error_rate": t["chexmask_seg"][2],
                })
        for status, label in (("no", "No"), ("yes", "Yes")):
            t = tables[view]["fallback_any"][status]
            csv_rows.append({
                "analysis": "center_crop_fallback_any_source",
                "view": view,
                "pneumonia": "All",
                "stratum": label,
                "n": t["n"],
                "raw_errors": t["raw"][0],
                "raw_error_rate": t["raw"][2],
                "medsam3_mask_errors": t["medsam3_seg"][0],
                "medsam3_mask_error_rate": t["medsam3_seg"][2],
                "chexmask_mask_errors": t["chexmask_seg"][0],
                "chexmask_mask_error_rate": t["chexmask_seg"][2],
            })
        if view == "Overall":
            continue
        for source in ("MedSAM3", "CheXMask"):
            for pna_name in ("Negative", "Positive"):
                t = tables[view]["fallback_frequency_by_source"][source][pna_name]
                csv_rows.append({
                    "analysis": "outer_test_fallback_frequency_by_source",
                    "view": view,
                    "pneumonia": pna_name,
                    "stratum": source,
                    "n": t["n"],
                    "raw_errors": t["fallback"],
                    "raw_error_rate": t["fallback_rate"],
                    "medsam3_mask_errors": "",
                    "medsam3_mask_error_rate": "",
                    "chexmask_mask_errors": "",
                    "chexmask_mask_error_rate": "",
                })
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"saved {OUT_MD}")
    print("mean pair dice", float(np.nanmean([r["pair_dice"] for r in rows])))
    for view in ("AP", "PA", "Overall"):
        print("====", view)
        for name in ("Dice ≥ 0.90", "0.80–0.90", "Dice < 0.80"):
            t = tables[view][name]
            print(name, t["n"], fmt(*t["raw"]), fmt(*t["medsam3_seg"]), fmt(*t["chexmask_seg"]))
        fb = tables[view]["fallback_any"]
        print(
            "fallback no", fmt(*fb["no"]["raw"]),
            "yes", fmt(*fb["yes"]["raw"]),
        )


if __name__ == "__main__":
    main()
