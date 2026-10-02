#!/usr/bin/env python3
"""Figure 6 / Supplementary Grad-CAM on original CXR coordinates.

Case selection uses only the report-derived pneumonia label and file
existence. Each fold CAM is reprojected to the original radiograph, then
the five maps are averaged and min–max scaled.
"""
from __future__ import annotations

import os

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from build_cxr_lung_crop_dataset import crop_with_mask, load_npy_mask_as_pil
from generate_gradcam_pneumonia import (
    DEFAULT_RUNS,
    MODEL_DISPLAY,
    ensemble_gradcam,
    normalize_cam,
    overlay_cam,
    read_predictions,
    select_label_available_samples,
)

STUDY_DIR = Path(__file__).resolve().parents[1]
DATA_BASE_DIR = Path(os.environ.get("DATA_BASE_DIR", STUDY_DIR)).resolve()
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", STUDY_DIR / "results_pneumonia")).resolve()
RESULTS = RESULTS_DIR
OUT_DIR = RESULTS / "analysis_outputs" / "figure6_gradcam"

MODES = ("raw", "medsam3_crop", "chexmask_crop")
MODE_TITLE = {
    "raw": "Raw",
    "medsam3_crop": "MedSAM3 crop",
    "chexmask_crop": "CheXMask crop",
}
CROP_ROOT = {
    ("AP", "medsam3_crop"): DATA_BASE_DIR / "cxr_medsam3_lung_seg_cropped_ap",
    ("PA", "medsam3_crop"): DATA_BASE_DIR / "cxr_medsam3_lung_seg_cropped_pa",
    ("AP", "chexmask_crop"): DATA_BASE_DIR / "cxr_chexmask_lung_seg_cropped_ap",
    ("PA", "chexmask_crop"): DATA_BASE_DIR / "cxr_chexmask_lung_seg_cropped_pa",
}
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
MAE_OK = 12.0
DISPLAY_MAX_SIDE = 960


def remap_path(path: str) -> str:
    text = str(path)
    text = text.replace("/mnt/d/CXR-Sepsis-Prediction/IEEE_ICCBE", str(DATA_BASE_DIR))
    text = text.replace(
        "/mnt/e/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0",
        os.environ.get("MIMIC_CXR_ROOT", str(DATA_BASE_DIR / "data" / "mimic-cxr-jpg" / "2.1.0")),
    )
    return text


def load_crop_manifest(view: str, mode: str) -> dict[str, dict]:
    root = CROP_ROOT[(view, mode)]
    tag = "medsam3" if "medsam3" in mode else "chexmask"
    path = root / f"manifest_{view.lower()}_{tag}_ok.json"
    with path.open(encoding="utf-8") as f:
        entries = json.load(f)
    index = {}
    for entry in entries:
        dicom = str(entry.get("dicom_id") or "")
        if dicom:
            index[dicom] = entry
    return index


def resize_cam_hw(cam: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = np.asarray(cam, dtype=np.float32)
    if arr.shape == (height, width):
        return arr
    out = Image.fromarray(arr).resize((int(width), int(height)), resample=Image.BILINEAR)
    return np.asarray(out, dtype=np.float32)


def reconstruct_geometry(src_path: str, mask_npy: str, existing_crop: str) -> dict:
    mask = load_npy_mask_as_pil(mask_npy)
    if mask is None:
        raise RuntimeError(f"Cannot load mask {mask_npy}")
    geom = crop_with_mask(
        src_path,
        "",
        existing_crop,
        mask_img_override=mask,
        **STUDY_CROP_KWARGS,
    )
    if not geom:
        raise RuntimeError(f"Crop reconstruction failed for {src_path}")
    recon = geom.pop("image")
    existing = Image.open(existing_crop).convert("RGB")
    recon_rgb = recon.convert("RGB")
    if existing.size != recon_rgb.size:
        raise RuntimeError(
            f"Reconstructed crop size {recon_rgb.size} != saved {existing.size} ({existing_crop})"
        )
    a = np.asarray(existing, dtype=np.float32)
    b = np.asarray(recon_rgb, dtype=np.float32)
    mae = float(np.mean(np.abs(a - b)))
    geom["verify_mae"] = mae
    geom["verify_size"] = list(existing.size)
    if mae > MAE_OK:
        raise RuntimeError(f"Crop reconstruction MAE {mae:.2f} > {MAE_OK} for {existing_crop}")
    return geom


def reproject_cam(cam: np.ndarray, geom: dict, original_size: tuple[int, int]) -> np.ndarray:
    """Map one fold CAM from the 512×512 crop input back to original CXR pixels.

    Order: bilinear upsample to the square padded crop, remove padding, restore
    the trimmed border, insert into the crop bounding box. Pixels outside that
    box stay zero.
    """
    orig_w, orig_h = int(original_size[0]), int(original_size[1])
    wf, hf = [int(v) for v in geom["final_crop_size"]]
    cam_final = resize_cam_hw(cam, hf, wf)
    pl, pt, pr, pb = [int(v) for v in geom["padding"]]
    h_content = hf - pt - pb
    w_content = wf - pl - pr
    cam_trim = cam_final[pt : pt + h_content, pl : pl + w_content]
    tl, tu, tr, td = [int(v) for v in geom["trim_bbox"]]
    cl, cu, cr, cd = [int(v) for v in geom["crop_bbox"]]
    crop_w, crop_h = cr - cl, cd - cu
    canvas_crop = np.zeros((crop_h, crop_w), dtype=np.float32)
    th, tw = td - tu, tr - tl
    placed = resize_cam_hw(cam_trim, th, tw)
    canvas_crop[tu:td, tl:tr] = placed
    canvas = np.zeros((orig_h, orig_w), dtype=np.float32)
    canvas[cu:cd, cl:cr] = canvas_crop
    return canvas


def fit_display(img: Image.Image, cam: np.ndarray, max_side: int = DISPLAY_MAX_SIDE):
    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img2 = img.resize((nw, nh), resample=Image.BILINEAR)
    cam2 = resize_cam_hw(cam, nh, nw)
    return img2, cam2, scale


def overlay_original(
    orig: Image.Image,
    cam_orig: np.ndarray,
    geom: dict | None,
    alpha: float,
    draw_bbox: bool,
) -> np.ndarray:
    img_d, cam_d, scale = fit_display(orig, cam_orig)
    overlay = overlay_cam(img_d, cam_d, alpha=alpha)
    if draw_bbox and geom is not None:
        cl, cu, cr, cd = [int(v) for v in geom["crop_bbox"]]
        box = [
            int(round(cl * scale)),
            int(round(cu * scale)),
            int(round(cr * scale)) - 1,
            int(round(cd * scale)) - 1,
        ]
        pil = Image.fromarray((overlay * 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil)
        draw.rectangle(box, outline=(255, 220, 0), width=2)
        overlay = np.asarray(pil).astype(np.float32) / 255.0
    return overlay


def average_reprojected_folds(
    fold_cams: tuple[np.ndarray, ...],
    mode: str,
    geom: dict | None,
    original_size: tuple[int, int],
) -> np.ndarray:
    orig_w, orig_h = int(original_size[0]), int(original_size[1])
    fold_maps = []
    for cam in fold_cams:
        if mode == "raw":
            fold_maps.append(resize_cam_hw(cam, orig_h, orig_w))
        else:
            if geom is None:
                raise ValueError("Crop geometry is required for crop-mode reprojection")
            fold_maps.append(reproject_cam(cam, geom, original_size))
    mean_cam = np.mean(np.stack(fold_maps, axis=0), axis=0)
    return normalize_cam(mean_cam)


def draw_two_case_figure(
    cases: list[dict],
    path: Path,
    labels: list[str],
    use_qc: bool,
) -> None:
    """Stack two 2×3 case blocks: ResNet-152 / RAD-DINO × Raw / MedSAM3 / CheXMask."""
    fig = plt.figure(figsize=(10.8, 13.6))
    gs = fig.add_gridspec(
        6,
        3,
        height_ratios=[0.16, 1.0, 1.0, 0.16, 1.0, 1.0],
        left=0.08,
        right=0.995,
        top=0.985,
        bottom=0.02,
        wspace=0.035,
        hspace=0.045,
    )
    models = ["resnet152", "rad_dino"]
    overlay_key = "overlay_qc" if use_qc else "overlay"
    for block, case in enumerate(cases):
        header_row = 0 if block == 0 else 3
        row_pair = (1, 2) if block == 0 else (4, 5)
        header = fig.add_subplot(gs[header_row, :])
        header.axis("off")
        letter = "A" if block == 0 else "B"
        header.text(
            0.0,
            0.42,
            f"({letter})  {labels[block]}",
            transform=header.transAxes,
            fontsize=13,
            fontweight="bold",
            va="center",
        )
        for r, model in enumerate(models):
            for c, mode in enumerate(MODES):
                ax = fig.add_subplot(gs[row_pair[r], c])
                ax.imshow(case["panels"][model][mode][overlay_key])
                ax.axis("off")
                if r == 0:
                    ax.set_title(MODE_TITLE[mode], fontsize=11, pad=5)
                if c == 0:
                    ax.text(
                        -0.045,
                        0.5,
                        MODEL_DISPLAY[model],
                        transform=ax.transAxes,
                        fontsize=11,
                        va="center",
                        ha="right",
                        rotation=90,
                    )
    stem = path.with_suffix("")
    for ext in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(ext), dpi=360, bbox_inches="tight")
    plt.close(fig)


def case_table_row(figure: str, panel: str, view: str, dicom: str, key: str) -> str:
    parts = Path(key).parts
    subject = next((p for p in parts if p.startswith("p") and len(p) > 3), "")
    study = next((p for p in parts if p.startswith("s")), "")
    return (
        f"| {figure} | {panel} | {view} | `{dicom}` | {subject} | {study} |"
    )


def write_case_table(main_cases: dict, extra_cases: list[dict], path: Path) -> None:
    lines = [
        "# Supplementary Table. Grad-CAM case identifiers",
        "",
        "Cases are report-derived pneumonia-positive outer-test radiographs. "
        "Selection required that the same DICOM exist as a raw image, a MedSAM3 crop, "
        "and a CheXMask crop. No classifier probability, threshold, or true-positive "
        "status was used. Eligible DICOM identifiers were sorted lexicographically and "
        "permuted with NumPy Generator seed 42. Two images were taken from each view: "
        "the first for Figure 6 and the second for Supplementary Figure S2.",
        "",
        "| Figure | Panel | View | DICOM ID | subject_id | study_id |",
        "|---|---|---|---|---|---|",
    ]
    order = [("AP", "Case 1 — AP view", "Figure 6", "A"), ("PA", "Case 2 — PA view", "Figure 6", "B")]
    for view, label, fig, panel in order:
        case = main_cases[view]
        lines.append(case_table_row(fig, f"({panel}) {label}", view, case["dicom_id"], case["key"]))
    extra_by_view = {c["view"]: c for c in extra_cases}
    extra_order = [
        ("AP", "Additional case — AP view", "Supplementary Figure S2", "A"),
        ("PA", "Additional case — PA view", "Supplementary Figure S2", "B"),
    ]
    for view, label, fig, panel in extra_order:
        if view not in extra_by_view:
            continue
        case = extra_by_view[view]
        lines.append(case_table_row(fig, f"({panel}) {label}", view, case["dicom_id"], case["key"]))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, default=0.38)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--n-main", type=int, default=1)
    parser.add_argument("--n-extra", type=int, default=1)
    parser.add_argument("--skip-cam", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    import torch

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    models = ["resnet152", "rad_dino"]
    n_total = int(args.n_main) + int(args.n_extra)
    selection = {
        "seed": int(args.selection_seed),
        "rule": (
            "report-derived pneumonia label = 1; raw, MedSAM3 crop, and CheXMask crop "
            "files all exist; NumPy Generator permutation of lexicographic DICOM IDs; "
            "no probability, threshold, or true-positive status"
        ),
        "views": {},
    }
    pred_cache = {}
    for view in ("AP", "PA"):
        pred_maps = {
            mode: read_predictions(RESULTS / DEFAULT_RUNS["resnet152"][view][mode])
            for mode in MODES
        }
        pred_cache[view] = pred_maps
        samples, diags = select_label_available_samples(
            view, pred_maps, MODES, n=n_total, seed=int(args.selection_seed)
        )
        selection["views"][view] = {
            "n_pool": diags[0]["n_pool"] if diags else 0,
            "main": diags[: args.n_main],
            "extra": diags[args.n_main :],
            "samples": [{"key": s.key, "dicom_id": Path(s.key).stem} for s in samples],
        }

    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2), flush=True)

    manifest_idx = {
        (view, mode): load_crop_manifest(view, mode)
        for view in ("AP", "PA")
        for mode in ("medsam3_crop", "chexmask_crop")
    }

    if args.skip_cam:
        print("skip-cam: selection only", flush=True)
        return

    figure6_payload = {}
    extra_cases = []

    for view in ("AP", "PA"):
        samples, _ = select_label_available_samples(
            view, pred_cache[view], MODES, n=n_total, seed=int(args.selection_seed)
        )
        for case_i, sample in enumerate(samples):
            dicom = Path(sample.key).stem
            orig_path = remap_path(sample.rows_by_mode["raw"].img_path)
            orig = Image.open(orig_path).convert("RGB")
            orig_size = orig.size
            geoms = {}
            verify_rows = []
            for mode in ("medsam3_crop", "chexmask_crop"):
                entry = manifest_idx[(view, mode)][dicom]
                mask_npy = remap_path(entry["lung_mask_npy"])
                crop_path = remap_path(sample.rows_by_mode[mode].img_path)
                geom = reconstruct_geometry(orig_path, mask_npy, crop_path)
                geoms[mode] = geom
                verify_rows.append(
                    {
                        "view": view,
                        "dicom_id": dicom,
                        "mode": mode,
                        "mae": geom["verify_mae"],
                        "final_crop_size": geom["final_crop_size"],
                        "crop_bbox": geom["crop_bbox"],
                    }
                )
                print(
                    f"[{view}] {dicom} {mode} MAE={geom['verify_mae']:.3f} "
                    f"bbox={geom['crop_bbox']} pad={geom['padding']}",
                    flush=True,
                )

            panels = {m: {} for m in models}
            for model in models:
                for mode in MODES:
                    run_dir = RESULTS / DEFAULT_RUNS[model][view][mode]
                    img_path = remap_path(sample.rows_by_mode[mode].img_path)
                    print(f"  Grad-CAM {model}/{view}/{mode}/{dicom}", flush=True)
                    result = ensemble_gradcam(
                        run_dir=run_dir,
                        img_path=img_path,
                        device=device,
                        target_class=1,
                        data_mode=mode,
                    )
                    geom = None if mode == "raw" else geoms[mode]
                    cam_orig = average_reprojected_folds(
                        result.fold_cams, mode, geom, orig_size
                    )
                    overlay = overlay_original(
                        orig, cam_orig, geom, args.alpha, draw_bbox=False
                    )
                    overlay_qc = overlay_original(
                        orig, cam_orig, geom, args.alpha, draw_bbox=mode != "raw"
                    )
                    panels[model][mode] = {
                        "overlay": overlay,
                        "overlay_qc": overlay_qc,
                        "prob": result.prob,
                        "fold_probs": list(result.fold_probs),
                    }

            case_dir = out_dir / view.lower() / f"case_{case_i+1:02d}_{dicom[:16]}"
            case_dir.mkdir(parents=True, exist_ok=True)
            with (case_dir / "geometry.json").open("w", encoding="utf-8") as f:
                json.dump({"verify": verify_rows, "geoms": geoms}, f, indent=2)
            payload = {
                "view": view,
                "dicom_id": dicom,
                "key": sample.key,
                "role": "main" if case_i < args.n_main else "extra",
                "panels": panels,
            }
            if case_i < args.n_main:
                figure6_payload[view] = payload
            else:
                extra_cases.append(payload)

    main_list = [figure6_payload["AP"], figure6_payload["PA"]]
    draw_two_case_figure(
        main_list,
        out_dir / "Figure_6_gradcam",
        ["Case 1 — AP view", "Case 2 — PA view"],
        use_qc=False,
    )
    draw_two_case_figure(
        main_list,
        out_dir / "Supplementary_Figure_S3_crop_extent_QC",
        ["Case 1 — AP view (crop window)", "Case 2 — PA view (crop window)"],
        use_qc=True,
    )

    extra_by_view = {c["view"]: c for c in extra_cases}
    if extra_by_view.get("AP") and extra_by_view.get("PA"):
        draw_two_case_figure(
            [extra_by_view["AP"], extra_by_view["PA"]],
            out_dir / "Supplementary_Figure_S2_gradcam",
            ["Additional case — AP view", "Additional case — PA view"],
            use_qc=False,
        )

    write_case_table(
        figure6_payload,
        extra_cases,
        out_dir / "Supplementary_Table_gradcam_case_ids.md",
    )

    fig6_caption = (
        "Figure 6. Grad-CAM of five-fold ensemble predictions for ResNet-152 and RAD-DINO "
        "on the same original radiograph under raw, MedSAM3-crop, and CheXMask-crop inputs. "
        "(A) Case 1, AP view. (B) Case 2, PA view. Cases are report-derived pneumonia-positive "
        "outer-test radiographs selected independently of model predictions: eligibility required "
        "only a positive label and the existence of all three inputs, after which two images per "
        "view were drawn with a fixed seed permutation of DICOM identifiers (seed 42). "
        "Each fold-specific attribution map was independently reprojected to the original "
        "radiograph coordinates and then averaged. All cropped attribution maps were reprojected "
        "to the original radiograph coordinate system; attributions outside the original crop "
        "extent were set to zero after reprojection. Each averaged Grad-CAM was independently "
        "min–max normalized to [0,1]; therefore, color intensity represents relative attribution "
        "within each map rather than an absolute attribution magnitude across models. "
        "The explained score is the binary decision logit z1−z0. Heatmaps are qualitative and "
        "are not interpreted as lesion-level localisation accuracy. DICOM identifiers are listed "
        "in the supplementary case table.\n"
    )
    s2_caption = (
        "Supplementary Figure S2. Additional Grad-CAM cases selected independently of model "
        "predictions with the same rule as Figure 6 (report-derived pneumonia-positive label and "
        "presence of raw, MedSAM3-crop, and CheXMask-crop inputs; seed-42 permutation; second "
        "image in each view). Layout matches Figure 6. All cropped attribution maps were "
        "reprojected to the original radiograph coordinate system; attributions outside the "
        "original crop extent were set to zero after reprojection. Each averaged map was "
        "independently min–max normalized to [0,1]. DICOM identifiers are listed in the "
        "supplementary case table.\n"
    )
    s3_caption = (
        "Supplementary Figure S3. Quality-control overlay of Figure 6 with the crop bounding "
        "box drawn in yellow. These boxes are omitted from the main figure so that raw and crop "
        "columns share the same original radiograph without an extra visual mark on the crop "
        "conditions. The box confirms that crop attributions were inserted only inside the "
        "mask-derived crop window after inversion of square padding and border trim.\n"
    )
    (out_dir / "Figure_6_caption.md").write_text(fig6_caption, encoding="utf-8")
    (out_dir / "Supplementary_Figure_S2_caption.md").write_text(s2_caption, encoding="utf-8")
    (out_dir / "Supplementary_Figure_S3_caption.md").write_text(s3_caption, encoding="utf-8")
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
