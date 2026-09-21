#!/usr/bin/env python
"""
Generate Grad-CAM comparison heatmaps for pneumonia experiments.

Default models: EVA-X-Small and ResNet50
Default preprocessing per view: raw vs MedSAM3 crop vs CheXmask crop

Uses trained 5-fold checkpoints and averages Grad-CAM maps across folds.

The heatmap target is the same binary logit used in BCE training, z1 - z0,
not the raw class-1 logit z1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAINING_DIR = PROJECT_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from pneumonia_train import (
    CXRSpatialClassifier,
    _binary_logits_from_outputs,
    _load_state_dict_compat,
    binary_positive_probability,
)


RESULTS_ROOT = Path(os.environ.get("RESULTS_DIR", PROJECT_ROOT / "results_pneumonia")).resolve()


DEFAULT_RUNS: Dict[str, Dict[str, Dict[str, str]]] = {
    "eva_x_small": {
        "AP": {
            "raw": "20260609_084149_eva_x_small_AP_raw_cv_5fold_holdout15pct",
            "medsam3_crop": "20260609_105650_eva_x_small_AP_medsam3_crop_cv_5fold_holdout15pct",
            "chexmask_crop": "20260609_130953_eva_x_small_AP_chexmask_crop_cv_5fold_holdout15pct",
        },
        "PA": {
            "raw": "20260613_020126_eva_x_small_PA_raw_cv_5fold_holdout15pct",
            "medsam3_crop": "20260613_222246_eva_x_small_PA_medsam3_crop_cv_5fold_holdout15pct",
            "chexmask_crop": "20260614_183837_eva_x_small_PA_chexmask_crop_cv_5fold_holdout15pct",
        },
    },
    "resnet50": {
        "AP": {
            "raw": "20260606_115100_resnet50_AP_raw_cv_5fold_holdout15pct",
            "medsam3_crop": "20260608_011949_resnet50_AP_medsam3_crop_cv_5fold_holdout15pct",
            "chexmask_crop": "20260608_195738_resnet50_AP_chexmask_crop_cv_5fold_holdout15pct",
        },
        "PA": {
            "raw": "20260612_170535_resnet50_PA_raw_cv_5fold_holdout15pct",
            "medsam3_crop": "20260613_134008_resnet50_PA_medsam3_crop_cv_5fold_holdout15pct",
            "chexmask_crop": "20260614_100050_resnet50_PA_chexmask_crop_cv_5fold_holdout15pct",
        },
    },
    "resnet152": {
        "AP": {
            "raw": "20260917_104528_resnet152_AP_raw_cv_5fold_holdout15pct",
            "medsam3_crop": "20260917_175129_resnet152_AP_medsam3_crop_cv_5fold_holdout15pct",
            "chexmask_crop": "20260918_010544_resnet152_AP_chexmask_crop_cv_5fold_holdout15pct",
        },
        "PA": {
            "raw": "20260918_042254_resnet152_PA_raw_cv_5fold_holdout15pct",
            "medsam3_crop": "20260918_112744_resnet152_PA_medsam3_crop_cv_5fold_holdout15pct",
            "chexmask_crop": "20260918_180651_resnet152_PA_chexmask_crop_cv_5fold_holdout15pct",
        },
    },
    "rad_dino": {
        "AP": {
            "raw": "20260915_124445_rad_dino_AP_raw_cv_5fold_holdout15pct",
            "medsam3_crop": "20260915_195640_rad_dino_AP_medsam3_crop_cv_5fold_holdout15pct",
            "chexmask_crop": "20260916_030439_rad_dino_AP_chexmask_crop_cv_5fold_holdout15pct",
        },
        "PA": {
            "raw": "20260916_064318_rad_dino_PA_raw_cv_5fold_holdout15pct",
            "medsam3_crop": "20260916_151509_rad_dino_PA_medsam3_crop_cv_5fold_holdout15pct",
            "chexmask_crop": "20260916_211411_rad_dino_PA_chexmask_crop_cv_5fold_holdout15pct",
        },
    },
}

MODEL_DISPLAY = {
    "eva_x_small": "EVA-X-Small",
    "resnet50": "ResNet50",
    "resnet152": "ResNet152",
    "rad_dino": "RAD-DINO",
}

FILE_PREFIX = {
    "eva_x_small": "evax",
    "resnet50": "resnet50",
    "resnet152": "resnet152",
    "rad_dino": "raddino",
}

DISPLAY_NAMES = {
    "raw": "Raw",
    "medsam3_crop": "MedSAM3 crop",
    "chexmask_crop": "CheXmask crop",
}

MODE_SEG_TAG = {
    "raw": "medsam3",
    "medsam3_crop": "medsam3",
    "chexmask_crop": "chexmask",
}


@dataclass(frozen=True)
class GradcamResult:
    img: Image.Image
    cam: np.ndarray
    cam_raw: np.ndarray
    prob: float
    fold_probs: Tuple[float, ...]
    lar: Dict[str, Optional[float]]
    fold_cams: Tuple[np.ndarray, ...] = ()


@dataclass(frozen=True)
class PredictionRow:
    img_path: str
    label: int
    prob_ensemble: float
    fold_probs: Tuple[float, ...]


@dataclass(frozen=True)
class SelectedSample:
    view: str
    key: str
    label: int
    rows_by_mode: Dict[str, PredictionRow]


def manifest_path(view: str, mode: str) -> Path:
    seg = MODE_SEG_TAG.get(mode, "medsam3")
    view_tag = view.lower()
    if mode.endswith("_crop"):
        root_name = f"cxr_{'chexmask' if seg == 'chexmask' else 'medsam3'}_lung_seg_cropped_{view_tag}"
    else:
        root_name = f"cxr_{'chexmask' if seg == 'chexmask' else 'medsam3'}_lung_seg_{view_tag}"
    return SCRIPT_DIR / root_name / f"manifest_{view_tag}_{seg}_ok.json"


def build_manifest_index(manifest_file: Path) -> Dict[str, dict]:
    with manifest_file.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    index: Dict[str, dict] = {}
    for entry in entries:
        for field in ("source_image_abs_path", "cropped_cxr_jpg", "masked_cxr_jpg"):
            path = entry.get(field)
            if path:
                index[image_key(str(path))] = entry
        dicom_id = entry.get("dicom_id")
        if dicom_id:
            index[str(dicom_id)] = entry
    return index


def load_raw_lung_mask_bool(entry: dict, input_size: int) -> np.ndarray:
    """Lung mask on full raw CXR, resized to model input."""
    mask_npy = entry.get("lung_mask_npy")
    if not mask_npy or not os.path.isfile(mask_npy):
        return np.ones((input_size, input_size), dtype=bool)
    mask = np.load(mask_npy)
    mask_bool = mask.astype(bool) if mask.dtype == bool else (mask > 0)
    mask_img = Image.fromarray(mask_bool.astype(np.uint8) * 255).resize(
        (input_size, input_size), resample=Image.NEAREST
    )
    return np.asarray(mask_img) > 0


def load_crop_content_mask_bool(img_path: str, input_size: int, threshold: int = 10) -> np.ndarray:
    """
    For cropped inputs: use non-black pixels of the actual crop file as the valid region.
    Full-image lung_mask_npy does not align after crop/trim unless we store crop bbox.
    """
    gray = Image.open(img_path).convert("L").resize((input_size, input_size), resample=Image.NEAREST)
    return np.asarray(gray) > int(threshold)


def attribution_mask_for_mode(
    entry: Optional[dict],
    img_path: str,
    mode: str,
    input_size: int,
) -> Tuple[np.ndarray, str]:
    if mode == "raw":
        if entry is None:
            return np.ones((input_size, input_size), dtype=bool), "lung_mask_on_raw"
        return load_raw_lung_mask_bool(entry, input_size), "lung_mask_on_raw"
    return load_crop_content_mask_bool(img_path, input_size), "content_mask_on_crop"


def resize_cam(cam: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    cam_f = np.asarray(cam, dtype=np.float32)
    if cam_f.shape == size:
        return cam_f
    out = Image.fromarray(cam_f).resize(size, resample=Image.BILINEAR)
    return np.asarray(out, dtype=np.float32)


def compute_attribution_ratio(cam_raw: np.ndarray, region_mask: np.ndarray) -> Dict[str, Optional[float]]:
    """Fraction of CAM mass / top-10% pixels inside region_mask."""
    cam = resize_cam(cam_raw, region_mask.shape)
    mask = np.asarray(region_mask, dtype=bool)
    cam_pos = np.clip(cam, 0.0, None)
    total = float(cam_pos.sum())
    if total <= 1e-12:
        return {"lar_mass": None, "lar_top10pct": None}

    lar_mass = float((cam_pos * mask).sum() / total)
    thresh = float(np.percentile(cam_pos, 90))
    top = cam_pos >= thresh
    top_count = int(top.sum())
    lar_top10 = float((top & mask).sum() / top_count) if top_count > 0 else None
    return {"lar_mass": lar_mass, "lar_top10pct": lar_top10}


def image_key(path: str) -> str:
    """Return a stable key shared by raw and cropped MIMIC-CXR paths."""
    parts = Path(path).parts
    if len(parts) < 4:
        return path
    return "/".join(parts[-4:])


def read_predictions(run_dir: Path) -> Dict[str, PredictionRow]:
    pred_path = run_dir / "outer_test" / "outer_test_predictions.csv"
    rows: Dict[str, PredictionRow] = {}
    with pred_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold_probs = []
            for i in range(1, 6):
                key = f"prob_fold{i}"
                if row.get(key) not in (None, ""):
                    fold_probs.append(float(row[key]))
            item = PredictionRow(
                img_path=row["img_path"],
                label=int(float(row["label"])),
                prob_ensemble=float(row["prob_ensemble"]),
                fold_probs=tuple(fold_probs),
            )
            rows[image_key(item.img_path)] = item
    return rows


def read_config(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return dict(summary.get("config", {}))


def existing_modes_for_key(
    key: str,
    pred_maps: Dict[str, Dict[str, PredictionRow]],
    modes: Sequence[str],
) -> bool:
    return all(key in pred_maps[mode] and os.path.isfile(pred_maps[mode][key].img_path) for mode in modes)


def select_sample(
    view: str,
    pred_maps: Dict[str, Dict[str, PredictionRow]],
    modes: Sequence[str],
    requested_key: Optional[str] = None,
) -> SelectedSample:
    common = set.intersection(*(set(pred_maps[mode].keys()) for mode in modes))
    if requested_key:
        key = requested_key.strip()
        if key not in common:
            raise ValueError(f"{view} requested sample key is not present in all modes: {key}")
        if not existing_modes_for_key(key, pred_maps, modes):
            raise ValueError(f"{view} requested sample image file is missing for at least one mode: {key}")
        rows_by_mode = {mode: pred_maps[mode][key] for mode in modes}
        label = next(iter(rows_by_mode.values())).label
        return SelectedSample(view=view, key=key, label=label, rows_by_mode=rows_by_mode)

    candidates: List[Tuple[str, str]] = []
    for key in common:
        if not existing_modes_for_key(key, pred_maps, modes):
            continue
        labels = {pred_maps[mode][key].label for mode in modes}
        if labels != {1}:
            continue
        candidates.append((Path(key).stem, key))

    if not candidates:
        raise RuntimeError(f"{view}: no label=1 sample with existing images across {', '.join(modes)}")

    candidates.sort()
    rng = np.random.default_rng(42)
    key = candidates[int(rng.permutation(len(candidates))[0])][1]
    rows_by_mode = {mode: pred_maps[mode][key] for mode in modes}
    return SelectedSample(view=view, key=key, label=1, rows_by_mode=rows_by_mode)


def read_outer_threshold(run_dir: Path, policy: str = "youden") -> Optional[float]:
    path = run_dir / "outer_test" / "cv5_outer_test_results.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    op = data.get("key_operating_points", {}).get(policy)
    if not op:
        return None
    return float(op["threshold"])


def select_label_available_samples(
    view: str,
    pred_maps: Dict[str, Dict[str, PredictionRow]],
    modes: Sequence[str],
    n: int,
    seed: int = 42,
) -> Tuple[List[SelectedSample], List[dict]]:
    """Select outer-test pneumonia images without using model scores.

    A case qualifies only when the report-derived label is 1 and the same
    source image exists for every requested preprocessing. Ranking uses a
    fixed seed permutation of lexicographic DICOM IDs.
    """
    common = set.intersection(*(set(pred_maps[mode].keys()) for mode in modes))
    candidates: List[Tuple[str, str]] = []
    for key in common:
        if not existing_modes_for_key(key, pred_maps, modes):
            continue
        labels = {pred_maps[mode][key].label for mode in modes}
        if labels != {1}:
            continue
        dicom_id = Path(key).stem
        candidates.append((dicom_id, key))
    if not candidates:
        raise RuntimeError(
            f"{view}: no label=1 outer-test image exists in all of {list(modes)}"
        )
    candidates.sort()
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(candidates))
    picked = [candidates[int(i)] for i in order[: max(1, int(n))]]
    samples: List[SelectedSample] = []
    diagnostics: List[dict] = []
    for rank, (dicom_id, key) in enumerate(picked):
        samples.append(
            SelectedSample(
                view=view,
                key=key,
                label=1,
                rows_by_mode={mode: pred_maps[mode][key] for mode in modes},
            )
        )
        diagnostics.append(
            {
                "rank": rank,
                "dicom_id": dicom_id,
                "key": key,
                "n_pool": len(candidates),
                "seed": int(seed),
            }
        )
    print(
        f"[{view}] prediction-independent pool={len(candidates)}; "
        f"selected {len(samples)} with seed={seed}",
        flush=True,
    )
    return samples, diagnostics


def case_dir_name(index: int, key: str) -> str:
    stem = Path(key).name.replace(".jpg", "").replace(".png", "")
    return f"case_{index:02d}_{stem[:16]}"


def load_image_tensor(
    img_path: str,
    input_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tuple[Image.Image, torch.Tensor]:
    img = Image.open(img_path).convert("RGB")
    if img.size != (input_size, input_size):
        img = img.resize((input_size, input_size), resample=Image.BILINEAR)
    tensor = transforms.functional.to_tensor(img)
    tensor = transforms.functional.normalize(tensor, mean, std)
    return img, tensor.unsqueeze(0)


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    cam = cam - float(cam.min())
    denom = float(cam.max())
    if denom > 1e-8:
        cam = cam / denom
    return np.clip(cam, 0.0, 1.0)


def overlay_cam(img: Image.Image, cam: np.ndarray, alpha: float = 0.28) -> np.ndarray:
    """Blend jet colormap onto the image. Alpha scales with CAM intensity so weak
    regions stay close to the original X-ray."""
    base = np.asarray(img).astype(np.float32) / 255.0
    cam_norm = normalize_cam(cam)
    cam_resized = np.asarray(
        Image.fromarray((cam_norm * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    ).astype(np.float32) / 255.0
    colored = plt.get_cmap("jet")(cam_resized)[..., :3]
    blend = float(alpha) * cam_resized[..., None]
    return np.clip((1.0 - blend) * base + blend * colored, 0.0, 1.0)


def gradcam_one_model(
    model: CXRSpatialClassifier,
    tensor: torch.Tensor,
    device: torch.device,
    target_class: int = 1,
) -> Tuple[np.ndarray, float]:
    """Compute Grad-CAM for the pneumonia-positive decision.

    ``target_class`` is kept for call-site compatibility, but the backward
    target is never ``logits[:, target_class]``. For this 2-logit head the
    explained score is ``z1 - z0``, the same binary logit BCE trains on.
    """
    if target_class != 1:
        raise ValueError("Binary pneumonia Grad-CAM supports target_class=1 only.")
    model.zero_grad(set_to_none=True)
    x = tensor.to(device)
    logits, extras = model(x, return_extras=True)
    if logits.dim() != 2 or logits.size(1) != 2:
        raise ValueError(
            f"Expected 2-output classifier logits shaped [batch, 2], got {tuple(logits.shape)}."
        )
    feat_map = extras["feat_map"]
    feat_map.retain_grad()
    binary_logit = _binary_logits_from_outputs(logits)
    score = binary_logit.sum()
    score.backward()
    grads = feat_map.grad
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feat_map).sum(dim=1)).squeeze(0)
    prob = float(binary_positive_probability(logits.detach()).float().cpu().item())
    return cam.detach().float().cpu().numpy(), prob


def checkpoint_paths(run_dir: Path, max_folds: int = 5) -> List[Path]:
    paths = []
    for fold in range(1, max_folds + 1):
        path = run_dir / f"fold_{fold}" / "best_model.pth"
        if path.is_file():
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No best_model.pth files found in {run_dir}")
    return paths


def ensemble_gradcam(
    run_dir: Path,
    img_path: str,
    device: torch.device,
    target_class: int,
    manifest_entry: Optional[dict] = None,
    data_mode: str = "raw",
) -> GradcamResult:
    cfg = read_config(run_dir)
    input_size = int(cfg.get("input_size", 512))
    mean = cfg.get("norm_mean") or [0.485, 0.456, 0.406]
    std = cfg.get("norm_std") or [0.229, 0.224, 0.225]
    img, tensor = load_image_tensor(img_path, input_size=input_size, mean=mean, std=std)

    cams_raw: List[np.ndarray] = []
    probs: List[float] = []
    ckpts = checkpoint_paths(run_dir, max_folds=int(cfg.get("max_folds") or cfg.get("n_folds") or 5))
    for ckpt in ckpts:
        model = CXRSpatialClassifier(
            model_name=cfg.get("model_name", "eva_x_small"),
            input_size=input_size,
            dropout=float(cfg.get("dropout", 0.15)),
            eva_x_ckpt_dir=cfg.get("eva_x_ckpt_dir"),
        ).to(device)
        state = torch.load(ckpt, map_location=device)
        _load_state_dict_compat(model, state)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(True)
        cam, prob = gradcam_one_model(model, tensor, device=device, target_class=target_class)
        cams_raw.append(cam.astype(np.float32))
        probs.append(prob)
        del model, state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_cam_raw = np.mean(np.stack(cams_raw, axis=0), axis=0)
    mean_cam = normalize_cam(mean_cam_raw)
    lar: Dict[str, Optional[float]] = {"lar_mass": None, "lar_top10pct": None, "lar_mask_type": None}
    if manifest_entry is not None or data_mode != "raw":
        region_mask, mask_type = attribution_mask_for_mode(
            manifest_entry, img_path, data_mode, input_size
        )
        lar = compute_attribution_ratio(mean_cam_raw, region_mask)
        lar["lar_mask_type"] = mask_type

    return GradcamResult(
        img=img,
        cam=mean_cam,
        cam_raw=mean_cam_raw,
        prob=float(np.mean(probs)),
        fold_probs=tuple(probs),
        lar=lar,
        fold_cams=tuple(cams_raw),
    )


def save_view_grid(
    model_name: str,
    view: str,
    sample: SelectedSample,
    mode_outputs: Dict[str, GradcamResult],
    out_dir: Path,
    modes: Sequence[str],
    alpha: float,
) -> Path:
    prefix = FILE_PREFIX.get(model_name, model_name)
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 5.7))
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        result = mode_outputs[mode]
        overlay = overlay_cam(result.img, result.cam, alpha=alpha)
        row = sample.rows_by_mode[mode]
        ax.imshow(overlay)
        ax.axis("off")
        lar_mass = result.lar.get("lar_mass")
        lar_txt = f"LAR={lar_mass:.2f}" if lar_mass is not None else "LAR=?"
        ax.set_title(
            f"{DISPLAY_NAMES.get(mode, mode)}\n"
            f"label={row.label}, csv p={row.prob_ensemble:.3f}, CAM p={result.prob:.3f}\n{lar_txt}",
            fontsize=10,
        )
    model_label = MODEL_DISPLAY.get(model_name, model_name)
    fig.suptitle(f"{model_label} Grad-CAM ({view}) - {sample.key}", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / f"{prefix}_gradcam_{view.lower()}_raw_vs_medsam3_vs_chexmask.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_combined_model_grid(
    view: str,
    sample_key: str,
    model_outputs: Dict[str, Dict[str, GradcamResult]],
    samples_by_model: Dict[str, SelectedSample],
    out_dir: Path,
    modes: Sequence[str],
    alpha: float,
) -> Path:
    model_names = list(model_outputs.keys())
    fig, axes = plt.subplots(len(model_names), len(modes), figsize=(5.2 * len(modes), 5.2 * len(model_names)))
    if len(model_names) == 1:
        axes = np.array([axes])
    for row_idx, model_name in enumerate(model_names):
        sample = samples_by_model[model_name]
        for col_idx, mode in enumerate(modes):
            ax = axes[row_idx, col_idx]
            result = model_outputs[model_name][mode]
            overlay = overlay_cam(result.img, result.cam, alpha=alpha)
            row = sample.rows_by_mode[mode]
            ax.imshow(overlay)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(DISPLAY_NAMES.get(mode, mode), fontsize=11)
            if col_idx == 0:
                ax.text(
                    -0.08,
                    0.5,
                    MODEL_DISPLAY.get(model_name, model_name),
                    transform=ax.transAxes,
                    fontsize=11,
                    va="center",
                    ha="right",
                    rotation=90,
                )
            lar_mass = result.lar.get("lar_mass")
            lar_txt = f"p={result.prob:.3f}, LAR={lar_mass:.2f}" if lar_mass is not None else f"p={result.prob:.3f}"
            ax.set_xlabel(lar_txt, fontsize=9, labelpad=2)
    fig.suptitle(f"Grad-CAM ({view}) - {sample_key}", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / f"combined_gradcam_{view.lower()}_raw_vs_medsam3_vs_chexmask.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_overlay_image(result: GradcamResult, path: Path, alpha: float) -> Path:
    overlay = overlay_cam(result.img, result.cam, alpha=alpha)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(path)
    return path


def save_individual_outputs(
    model_name: str,
    view: str,
    sample: SelectedSample,
    mode_outputs: Dict[str, GradcamResult],
    out_dir: Path,
    modes: Sequence[str],
    alpha: float,
    overlay_only: bool = False,
    name_prefix: Optional[str] = None,
) -> List[Path]:
    prefix = name_prefix or FILE_PREFIX.get(model_name, model_name)
    paths = []
    for mode in modes:
        result = mode_outputs[mode]
        stem = f"{prefix}_gradcam_{view.lower()}_{mode}"
        overlay_path = out_dir / f"{stem}_overlay.png"
        save_overlay_image(result, overlay_path, alpha=alpha)
        paths.append(overlay_path)
        if not overlay_only:
            heatmap_path = out_dir / f"{stem}_heatmap.png"
            Image.fromarray((normalize_cam(result.cam) * 255).astype(np.uint8)).resize(
                result.img.size, Image.BILINEAR
            ).save(heatmap_path)
            paths.append(heatmap_path)
    return paths


def resolve_sample_for_model(
    view: str,
    model_name: str,
    pred_maps: Dict[str, Dict[str, PredictionRow]],
    modes: Sequence[str],
    forced_key: Optional[str] = None,
) -> SelectedSample:
    if forced_key:
        key = forced_key.strip()
        if not existing_modes_for_key(key, pred_maps, modes):
            raise ValueError(f"{view}/{model_name}: sample missing for key {key}")
        rows_by_mode = {mode: pred_maps[mode][key] for mode in modes}
        label = next(iter(rows_by_mode.values())).label
        return SelectedSample(view=view, key=key, label=label, rows_by_mode=rows_by_mode)
    return select_sample(view, pred_maps, modes)


def write_metadata(
    out_dir: Path,
    selections: Dict[str, Dict[str, SelectedSample]],
    generated: Dict[str, Dict[str, object]],
    lar_results: Dict[str, Dict[str, object]],
    alpha: float,
    models: Sequence[str],
) -> Path:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "alpha": float(alpha),
        "gradcam_target": "z1-z0",
        "gradcam_target_note": (
            "Heatmaps explain BCE binary logit logits[:,1]-logits[:,0], "
            "not the raw class-1 logit logits[:,1]."
        ),
        "models": list(models),
        "runs": {model: DEFAULT_RUNS[model] for model in models},
        "samples": {
            view: {
                model: {
                    "key": sample.key,
                    "label": sample.label,
                    "modes": {
                        mode: {
                            "img_path": row.img_path,
                            "prob_ensemble_csv": row.prob_ensemble,
                            "fold_probs_csv": row.fold_probs,
                            "lung_attribution": lar_results.get(f"{view}/{model}/{mode}", {}),
                        }
                        for mode, row in sample.rows_by_mode.items()
                    },
                }
                for model, sample in per_view.items()
            }
            for view, per_view in selections.items()
        },
        "generated": generated,
    }
    path = out_dir / "gradcam_metadata.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def parse_sample_overrides(values: Optional[Iterable[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for value in values or []:
        if ":" not in value:
            raise ValueError("--sample-key must look like AP:p10/p.../s.../image.jpg")
        view, key = value.split(":", 1)
        overrides[view.strip().upper()] = key.strip()
    return overrides


def run_tp_overlay_batch(
    args: argparse.Namespace,
    device: torch.device,
    models: Sequence[str],
    modes: Sequence[str],
    out_dir: Path,
    manifest_cache: Dict[Tuple[str, str], Dict[str, dict]],
) -> Dict[str, object]:
    tp_root = out_dir / "tp_overlays"
    tp_root.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {
        "views": {},
        "gradcam_target": "z1-z0",
        "models": list(models),
        "modes": list(modes),
        "selection_seed": int(getattr(args, "selection_seed", 42)),
        "selection_criteria": [
            "pneumonia-positive (label=1) outer-test sample",
            "same source image present and readable in raw, medsam3_crop, chexmask_crop",
            "no model probability or threshold used for ranking",
            "deterministic permutation of lexicographic DICOM IDs with a fixed seed",
        ],
    }

    for view in args.views:
        pred_maps_by_model: Dict[str, Dict[str, Dict[str, PredictionRow]]] = {}
        for model_name in models:
            pred_maps_by_model[model_name] = {}
            for mode in modes:
                run_dir = args.results_root / DEFAULT_RUNS[model_name][view][mode]
                pred_maps_by_model[model_name][mode] = read_predictions(run_dir)

        ref_maps = pred_maps_by_model[models[0]]
        tp_samples, tp_diagnostics = select_label_available_samples(
            view,
            ref_maps,
            modes,
            n=int(args.tp_cases),
            seed=int(getattr(args, "selection_seed", 42)),
        )
        print(f"[{view}] selected {len(tp_samples)} label-available case(s)", flush=True)

        view_summary: List[dict] = []
        view_model_outputs_all: Dict[str, Dict[str, Dict[str, GradcamResult]]] = {}

        for case_idx, sample in enumerate(tp_samples, start=1):
            case_dir = tp_root / view.lower() / case_dir_name(case_idx, sample.key)
            case_dir.mkdir(parents=True, exist_ok=True)
            diag = tp_diagnostics[case_idx - 1]
            case_entry = {
                "key": sample.key,
                "label": sample.label,
                "dir": str(case_dir),
                "raw_prob_ensemble": sample.rows_by_mode["raw"].prob_ensemble,
                "selection": diag,
                "models": {},
            }
            print(
                f"[{view}] case {case_idx}: {sample.key} "
                f"(dicom={diag['dicom_id']}, pool={diag['n_pool']})",
                flush=True,
            )

            view_model_outputs: Dict[str, Dict[str, GradcamResult]] = {}
            for model_name in models:
                run_names = DEFAULT_RUNS[model_name][view]
                pred_maps = pred_maps_by_model[model_name]
                model_sample = SelectedSample(
                    view=view,
                    key=sample.key,
                    label=sample.label,
                    rows_by_mode={mode: pred_maps[mode][sample.key] for mode in modes},
                )
                mode_outputs: Dict[str, GradcamResult] = {}
                model_entry: Dict[str, object] = {"overlays": {}, "grid": None}
                for mode in modes:
                    run_dir = args.results_root / run_names[mode]
                    row = model_sample.rows_by_mode[mode]
                    manifest_entry = manifest_cache.get((view, mode), {}).get(sample.key)
                    print(f"  [{model_name}/{mode}] Grad-CAM overlay", flush=True)
                    mode_outputs[mode] = ensemble_gradcam(
                        run_dir=run_dir,
                        img_path=row.img_path,
                        device=device,
                        target_class=1,
                        manifest_entry=manifest_entry,
                        data_mode=mode,
                    )
                overlay_paths = save_individual_outputs(
                    model_name,
                    view,
                    model_sample,
                    mode_outputs,
                    case_dir,
                    modes,
                    alpha=float(args.alpha),
                    overlay_only=True,
                )
                model_entry["overlays"] = {modes[i]: str(overlay_paths[i]) for i in range(len(modes))}
                if not args.no_grids:
                    grid_path = save_view_grid(
                        model_name, view, model_sample, mode_outputs, case_dir, modes, alpha=float(args.alpha)
                    )
                    model_entry["grid"] = str(grid_path)
                case_entry["models"][model_name] = model_entry
                view_model_outputs[model_name] = mode_outputs

            view_model_outputs_all[sample.key] = view_model_outputs
            if len(models) > 1 and not args.no_grids:
                combined_path = save_combined_model_grid(
                    view,
                    sample.key,
                    view_model_outputs,
                    {m: sample for m in models},
                    case_dir,
                    modes,
                    alpha=float(args.alpha),
                )
                case_entry["combined_grid"] = str(combined_path)
            view_summary.append(case_entry)

        summary["views"][view] = {
            "n_pool": tp_diagnostics[0]["n_pool"] if tp_diagnostics else 0,
            "seed": int(getattr(args, "selection_seed", 42)),
            "cases": view_summary,
        }

    summary_path = tp_root / "tp_overlay_index.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] tp_overlay_index={summary_path}", flush=True)
    return {"tp_overlay_root": str(tp_root), "tp_overlay_index": str(summary_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Max overlay opacity at hottest CAM pixels (weak regions blend less)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["eva_x_small", "resnet50"],
        choices=list(DEFAULT_RUNS.keys()),
        help="Backbones to visualize",
    )
    parser.add_argument(
        "--sample-key",
        action="append",
        default=None,
        help="Optional view-specific sample key, e.g. AP:p10/p123/s456/image.jpg. Can be used twice.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip PNG export; only compute LAR and update metadata JSON",
    )
    parser.add_argument(
        "--overlay-only",
        action="store_true",
        help="Save overlay PNGs only (skip heatmap PNGs)",
    )
    parser.add_argument(
        "--tp-cases",
        type=int,
        default=0,
        help="If >0, export overlays for N report-derived pneumonia-positive outer-test "
             "cases per view. Selection uses label=1 and file existence only; "
             "model scores, thresholds, and true-positive status are not used.",
    )
    parser.add_argument(
        "--tp-threshold-policy",
        default="youden",
        choices=["youden", "spec80"],
        help="Unused for case selection; kept only for CLI compatibility.",
    )
    parser.add_argument(
        "--tp-margin",
        type=float,
        default=0.10,
        help="Unused for case selection; kept only for CLI compatibility.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=["AP", "PA"],
        choices=["AP", "PA"],
        help="Views to render",
    )
    parser.add_argument(
        "--no-grids",
        action="store_true",
        help="Skip multi-panel grid PNGs (individual overlays only)",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=42,
        help="Fixed seed for prediction-independent DICOM permutation",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    modes = ("raw", "medsam3_crop", "chexmask_crop")
    models = list(args.models)
    out_dir = args.output_dir
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = args.results_root / f"gradcam_ap_pa_raw_medsam3_chexmask_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_cache: Dict[Tuple[str, str], Dict[str, dict]] = {}
    for view in args.views:
        for mode in modes:
            mp = manifest_path(view, mode)
            if mp.is_file():
                manifest_cache[(view, mode)] = build_manifest_index(mp)

    if int(args.tp_cases) > 0:
        tp_meta = run_tp_overlay_batch(args, device, models, modes, out_dir, manifest_cache)
        print(f"[done] output_dir={out_dir}", flush=True)
        print(f"[done] tp_overlays={tp_meta['tp_overlay_root']}", flush=True)
        return

    overrides = parse_sample_overrides(args.sample_key)
    selections: Dict[str, Dict[str, SelectedSample]] = {}
    generated: Dict[str, Dict[str, object]] = {}
    lar_results: Dict[str, Dict[str, object]] = {}

    for view in args.views:
        selections[view] = {}
        view_model_outputs: Dict[str, Dict[str, GradcamResult]] = {}
        sample_key_for_view: Optional[str] = overrides.get(view)

        for model_name in models:
            run_names = DEFAULT_RUNS[model_name][view]
            pred_maps = {
                mode: read_predictions(args.results_root / run_names[mode])
                for mode in modes
            }
            sample = resolve_sample_for_model(
                view,
                model_name,
                pred_maps,
                modes,
                forced_key=sample_key_for_view,
            )
            selections[view][model_name] = sample
            if sample_key_for_view is None:
                sample_key_for_view = sample.key
            print(f"[{view}/{model_name}] selected label={sample.label} key={sample.key}", flush=True)

            mode_outputs: Dict[str, GradcamResult] = {}
            for mode in modes:
                run_dir = args.results_root / run_names[mode]
                row = sample.rows_by_mode[mode]
                manifest_entry = manifest_cache.get((view, mode), {}).get(sample.key)
                print(f"[{view}/{model_name}/{mode}] Grad-CAM: {row.img_path}", flush=True)
                mode_outputs[mode] = ensemble_gradcam(
                    run_dir=run_dir,
                    img_path=row.img_path,
                    device=device,
                    target_class=1,
                    manifest_entry=manifest_entry,
                    data_mode=mode,
                )
                lar_key = f"{view}/{model_name}/{mode}"
                lar_results[lar_key] = {
                    "lar_mass": mode_outputs[mode].lar.get("lar_mass"),
                    "lar_top10pct": mode_outputs[mode].lar.get("lar_top10pct"),
                    "lar_mask_type": mode_outputs[mode].lar.get("lar_mask_type"),
                    "cam_prob": mode_outputs[mode].prob,
                }
                lar_mass = mode_outputs[mode].lar.get("lar_mass")
                mask_type = mode_outputs[mode].lar.get("lar_mask_type")
                if lar_mass is not None:
                    print(
                        f"  LAR mass={lar_mass:.3f}, top10%={mode_outputs[mode].lar.get('lar_top10pct')}, "
                        f"mask={mask_type}",
                        flush=True,
                    )

            view_model_outputs[model_name] = mode_outputs
            if not args.skip_plots:
                grid_path = save_view_grid(
                    model_name, view, sample, mode_outputs, out_dir, modes, alpha=float(args.alpha)
                )
                individual_paths = save_individual_outputs(
                    model_name, view, sample, mode_outputs, out_dir, modes,
                    alpha=float(args.alpha), overlay_only=bool(args.overlay_only),
                )
                generated[f"{view}/{model_name}"] = {
                    "grid": str(grid_path),
                    "individual": [str(p) for p in individual_paths],
                    "cam_prob_mean_by_mode": {mode: mode_outputs[mode].prob for mode in modes},
                    "cam_fold_probs_by_mode": {mode: list(mode_outputs[mode].fold_probs) for mode in modes},
                    "lung_attribution_by_mode": {mode: mode_outputs[mode].lar for mode in modes},
                }
            else:
                generated[f"{view}/{model_name}"] = {
                    "lung_attribution_by_mode": {mode: mode_outputs[mode].lar for mode in modes},
                }

        if len(models) > 1 and not args.skip_plots:
            combined_path = save_combined_model_grid(
                view,
                sample_key_for_view or selections[view][models[0]].key,
                view_model_outputs,
                selections[view],
                out_dir,
                modes,
                alpha=float(args.alpha),
            )
            generated[f"{view}/combined"] = {"grid": str(combined_path)}

    lar_path = out_dir / "lung_attribution_ratios.json"
    with lar_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "description": {
                    "lar_mass": "CAM mass inside region mask / total CAM mass",
                    "lar_top10pct": "top-10% CAM pixels inside region mask",
                    "lar_mask_type": {
                        "lung_mask_on_raw": "MedSAM3/CheXmask lung_mask_npy on full raw CXR (valid for raw only)",
                        "content_mask_on_crop": "non-black pixels of the cropped JPG (crop lung mask is not stored aligned)",
                    },
                    "note": "Do not compare lar_mass across raw vs crop directly; mask definitions differ.",
                },
                "results": lar_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    metadata_path = write_metadata(out_dir, selections, generated, lar_results, float(args.alpha), models)
    print(f"[done] output_dir={out_dir}", flush=True)
    print(f"[done] metadata={metadata_path}", flush=True)
    print(f"[done] lung_attribution={lar_path}", flush=True)


if __name__ == "__main__":
    main()
