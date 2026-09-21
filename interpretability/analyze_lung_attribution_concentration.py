#!/usr/bin/env python3
"""Lung-field attribution concentration on the full outer-test set.

Primary metric: fraction of non-negative fold-mean Grad-CAM mass inside a
reference lung mask (MedSAM3 and CheXMask, computed independently).
Secondary metric: fraction of the highest 10% positive-CAM pixels inside
the same mask.

Scope matches Figure 6: AP and PA separately; ResNet-152 and RAD-DINO;
raw / MedSAM3 crop / CheXMask crop only (hard-mask excluded). Each fold
CAM is reprojected to original CXR coordinates, then averaged. Metrics
are computed on ReLU(mean CAM) before visualization min-max scaling.

Paired crop-minus-raw differences use a subject-level cluster bootstrap
(2,000 replicates, seed 42), matching the outer-test CI convention.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
TRAINING_DIR = PACKAGE_ROOT / "training"
for _source_dir in (SCRIPT_DIR, TRAINING_DIR):
    if str(_source_dir) not in sys.path:
        sys.path.insert(0, str(_source_dir))

from generate_figure6_gradcam import (
    load_crop_manifest,
    remap_path,
    reproject_cam,
    resize_cam_hw,
)
from lung_attr_geom_worker import geom_job
from generate_gradcam_pneumonia import (
    DEFAULT_RUNS,
    checkpoint_paths,
    read_config,
    read_predictions,
)
from pneumonia_train import (
    CXRSpatialClassifier,
    _binary_logits_from_outputs,
    _load_state_dict_compat,
)

RESULTS = Path(os.environ.get("RESULTS_DIR", PACKAGE_ROOT / "results_pneumonia")).resolve()
SPLITS = Path(os.environ.get("SPLIT_DIR", PACKAGE_ROOT / "splits")).resolve()
OUT_DIR = RESULTS / "lung_attribution_concentration"

MODELS = ("resnet152", "rad_dino")
VIEWS = ("AP", "PA")
MODES = ("raw", "medsam3_crop", "chexmask_crop")
REFERENCE_MASKS = ("medsam3", "chexmask")

MODEL_LABEL = {
    "resnet152": "ResNet-152",
    "rad_dino": "RAD-DINO",
}
MODE_LABEL = {
    "raw": "Raw",
    "medsam3_crop": "MedSAM3 crop",
    "chexmask_crop": "CheXMask crop",
}
REF_LABEL = {
    "medsam3": "MedSAM3",
    "chexmask": "CheXMask",
}

N_BOOT = 2000
BOOT_SEED = 42
METRIC_MAX_SIDE = 768
ROW_FIELDS = (
    "dicom_id",
    "subject_id",
    "view",
    "model",
    "mode",
    "reference_mask",
    "lung_mass",
    "top10_inside",
    "label",
    "n_folds",
    "cam_sum",
)


def lung_attribution_mass(cam: np.ndarray, lung_mask: np.ndarray) -> float:
    cam = np.maximum(np.asarray(cam, dtype=np.float64), 0.0)
    mask = np.asarray(lung_mask, dtype=bool)
    total = float(cam.sum())
    if total <= 0.0:
        return float("nan")
    return float(cam[mask].sum() / total)


def top_attribution_inside_lung(
    cam: np.ndarray,
    lung_mask: np.ndarray,
    top_fraction: float = 0.10,
) -> float:
    cam = np.maximum(np.asarray(cam, dtype=np.float64), 0.0)
    mask = np.asarray(lung_mask, dtype=bool)
    positive = cam > 0.0
    values = cam[positive]
    if values.size == 0:
        return float("nan")
    threshold = float(np.quantile(values, 1.0 - float(top_fraction)))
    top = (cam >= threshold) & positive
    n_top = int(top.sum())
    if n_top == 0:
        return float("nan")
    return float((top & mask).sum() / n_top)


def metric_hw(orig_wh: tuple[int, int], max_side: int = METRIC_MAX_SIDE) -> tuple[int, int, float]:
    orig_w, orig_h = int(orig_wh[0]), int(orig_wh[1])
    scale = min(1.0, float(max_side) / float(max(orig_w, orig_h)))
    out_w = max(1, int(round(orig_w * scale)))
    out_h = max(1, int(round(orig_h * scale)))
    return out_w, out_h, float(scale)


def scale_geom(geom: dict, scale: float) -> dict:
    if scale == 1.0:
        return geom

    def sc(vals: list) -> list[int]:
        return [int(round(float(v) * scale)) for v in vals]

    out = {
        "original_size": sc(geom["original_size"]),
        "crop_bbox": sc(geom["crop_bbox"]),
        "trim_bbox": sc(geom["trim_bbox"]),
        "padding": sc(geom["padding"]),
        "final_crop_size": sc(geom["final_crop_size"]),
    }
    wf, hf = out["final_crop_size"]
    pl, pt, pr, pb = out["padding"]
    if hf - pt - pb < 1:
        out["padding"][3] = max(0, hf - pt - 1)
    if wf - pl - pr < 1:
        out["padding"][2] = max(0, wf - pl - 1)
    cl, cu, cr, cd = out["crop_bbox"]
    if cr <= cl:
        out["crop_bbox"][2] = cl + 1
    if cd <= cu:
        out["crop_bbox"][3] = cu + 1
    tl, tu, tr, td = out["trim_bbox"]
    if tr <= tl:
        out["trim_bbox"][2] = tl + 1
    if td <= tu:
        out["trim_bbox"][3] = tu + 1
    return out


def load_lung_mask_bool(npy_path: str, orig_wh: tuple[int, int]) -> np.ndarray:
    arr = np.load(npy_path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    mask = np.asarray(arr, dtype=bool)
    orig_w, orig_h = int(orig_wh[0]), int(orig_wh[1])
    if mask.shape != (orig_h, orig_w):
        pil = Image.fromarray((mask.astype(np.uint8) * 255))
        pil = pil.resize((orig_w, orig_h), resample=Image.NEAREST)
        mask = np.asarray(pil) > 0
    return mask


def jpeg_size(path: str) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def shard_path(out_dir: Path, model: str, view: str, mode: str) -> Path:
    return out_dir / "shards" / f"{model}_{view}_{mode}.csv"


def load_split(view: str) -> list[dict]:
    path = SPLITS / f"{view}_outer_test.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_catalog(limit: int = 0) -> list[dict]:
    catalog = []
    manifests = {
        (view, mode): load_crop_manifest(view, mode)
        for view in VIEWS
        for mode in ("medsam3_crop", "chexmask_crop")
    }
    for view in VIEWS:
        pred_by_dicom = {}
        for mode in MODES:
            index = {}
            for key, pred in read_predictions(
                RESULTS / DEFAULT_RUNS["resnet152"][view][mode]
            ).items():
                stem = Path(pred.img_path).stem
                index[stem] = (key, pred)
            pred_by_dicom[mode] = index
        for row in load_split(view):
            dicom = str(row["dicom_id"]).strip()
            ms_entry = manifests[(view, "medsam3_crop")].get(dicom)
            cx_entry = manifests[(view, "chexmask_crop")].get(dicom)
            if ms_entry is None or cx_entry is None:
                continue
            raw_key = None
            paths = {}
            ok = True
            for mode in MODES:
                hit = pred_by_dicom[mode].get(dicom)
                if hit is None:
                    ok = False
                    break
                key, found = hit
                if not os.path.isfile(remap_path(found.img_path)):
                    ok = False
                    break
                if mode == "raw":
                    raw_key = key
                paths[mode] = remap_path(found.img_path)
            if not ok:
                continue
            ms_npy = remap_path(ms_entry["lung_mask_npy"])
            cx_npy = remap_path(cx_entry["lung_mask_npy"])
            if not os.path.isfile(ms_npy) or not os.path.isfile(cx_npy):
                continue
            catalog.append(
                {
                    "dicom_id": dicom,
                    "subject_id": str(row["subject_id"]).strip(),
                    "view": view,
                    "label": int(float(row["label"])),
                    "raw_key": raw_key,
                    "img_path": paths,
                    "orig_path": paths["raw"],
                    "mask_npy": {"medsam3": ms_npy, "chexmask": cx_npy},
                }
            )
    catalog.sort(key=lambda r: (r["view"], r["dicom_id"]))
    if limit > 0:
        by_view: dict[str, list[dict]] = {v: [] for v in VIEWS}
        for rec in catalog:
            by_view[rec["view"]].append(rec)
        trimmed = []
        per_view = max(1, limit // max(1, len(VIEWS)))
        for view in VIEWS:
            trimmed.extend(by_view[view][:per_view])
        catalog = trimmed
    return catalog


def geometry_cache_path(out_dir: Path) -> Path:
    return out_dir / "crop_geometry_cache.json"


def build_geometry_cache(
    catalog: list[dict],
    out_dir: Path,
    workers: int,
) -> dict:
    path = geometry_cache_path(out_dir)
    cache: dict = {"by_dicom": {}}
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            cache = json.load(f)
    by_dicom = cache.setdefault("by_dicom", {})
    jobs = []
    for rec in catalog:
        dicom = rec["dicom_id"]
        entry = by_dicom.get(dicom, {})
        for mode in ("medsam3_crop", "chexmask_crop"):
            if mode in entry and entry[mode]:
                continue
            jobs.append(
                (
                    dicom,
                    mode,
                    rec["orig_path"],
                    rec["mask_npy"]["medsam3" if mode == "medsam3_crop" else "chexmask"],
                    rec["img_path"][mode],
                )
            )
        if "original_size" not in entry:
            try:
                entry["original_size"] = list(jpeg_size(rec["orig_path"]))
                by_dicom[dicom] = entry
            except Exception:
                pass
    if jobs:
        print(f"[geometry] reconstructing {len(jobs)} crop maps", flush=True)
        workers = max(1, int(workers))
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(geom_job, job) for job in jobs]
            for fut in as_completed(futures):
                dicom, mode, geom, err = fut.result()
                entry = by_dicom.setdefault(dicom, {})
                if geom is None:
                    entry.setdefault("errors", {})[mode] = err
                else:
                    entry[mode] = geom
                    if "original_size" not in entry:
                        entry["original_size"] = geom["original_size"]
                done += 1
                if done % 200 == 0 or done == len(jobs):
                    print(f"[geometry] {done}/{len(jobs)}", flush=True)
                if done % 500 == 0:
                    path.write_text(json.dumps(cache), encoding="utf-8")
        path.write_text(json.dumps(cache), encoding="utf-8")
        print(f"[geometry] wrote {path}", flush=True)
    n_ok = sum(
        1
        for rec in catalog
        if rec["dicom_id"] in by_dicom
        and by_dicom[rec["dicom_id"]].get("medsam3_crop")
        and by_dicom[rec["dicom_id"]].get("chexmask_crop")
    )
    print(f"[geometry] complete pairs: {n_ok}/{len(catalog)}", flush=True)
    return cache


def load_fold_models(run_dir: Path, device: torch.device) -> tuple[list, dict]:
    cfg = read_config(run_dir)
    input_size = int(cfg.get("input_size", 512))
    mean = cfg.get("norm_mean") or [0.485, 0.456, 0.406]
    std = cfg.get("norm_std") or [0.229, 0.224, 0.225]
    ckpts = checkpoint_paths(run_dir, max_folds=int(cfg.get("max_folds") or 5))
    models = []
    for ckpt in ckpts:
        model = CXRSpatialClassifier(
            model_name=cfg.get("model_name", "resnet152"),
            input_size=input_size,
            dropout=float(cfg.get("dropout", 0.15)),
            eva_x_ckpt_dir=cfg.get("eva_x_ckpt_dir"),
        ).to(device)
        state = torch.load(ckpt, map_location=device)
        _load_state_dict_compat(model, state)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(True)
        models.append(model)
        del state
    meta = {"input_size": input_size, "mean": mean, "std": std, "n_folds": len(models)}
    return models, meta


def load_tensor(path: str, input_size: int, mean, std) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (input_size, input_size):
        img = img.resize((input_size, input_size), resample=Image.BILINEAR)
    tensor = transforms.functional.to_tensor(img)
    tensor = transforms.functional.normalize(tensor, mean, std)
    return tensor


def gradcam_batch(model, tensor_b: torch.Tensor, device: torch.device) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    x = tensor_b.to(device, non_blocking=True)
    logits, extras = model(x, return_extras=True)
    feat_map = extras["feat_map"]
    feat_map.retain_grad()
    score = _binary_logits_from_outputs(logits).sum()
    score.backward()
    grads = feat_map.grad
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feat_map).sum(dim=1))
    return cam.detach().float().cpu().numpy()


def average_reprojected_positive(
    fold_cams: list[np.ndarray],
    mode: str,
    geom: dict | None,
    original_size: tuple[int, int],
) -> np.ndarray:
    mean_cam = np.mean(np.stack(fold_cams, axis=0), axis=0)
    out_w, out_h, scale = metric_hw(original_size)
    if mode == "raw":
        mapped = resize_cam_hw(mean_cam, out_h, out_w)
    else:
        if geom is None:
            raise ValueError("crop geometry required")
        mapped = reproject_cam(mean_cam, scale_geom(geom, scale), (out_w, out_h))
    return np.maximum(mapped.astype(np.float64, copy=False), 0.0)


def build_metric_mask_cache(recs: list[dict], geom_cache: dict) -> dict[str, dict[str, np.ndarray]]:
    print(f"[masks] preparing metric-size lung masks for {len(recs)} images", flush=True)
    by_dicom = geom_cache.get("by_dicom", {})
    cache: dict[str, dict[str, np.ndarray]] = {}
    t0 = time.time()
    for i, rec in enumerate(recs, start=1):
        entry = by_dicom.get(rec["dicom_id"], {})
        orig = entry.get("original_size") or jpeg_size(rec["orig_path"])
        orig_wh = (int(orig[0]), int(orig[1]))
        out_w, out_h, _ = metric_hw(orig_wh)
        cache[rec["dicom_id"]] = {
            "medsam3": load_lung_mask_bool(rec["mask_npy"]["medsam3"], (out_w, out_h)),
            "chexmask": load_lung_mask_bool(rec["mask_npy"]["chexmask"], (out_w, out_h)),
            "orig_wh": orig_wh,
        }
        if i % 500 == 0 or i == len(recs):
            print(f"[masks] {i}/{len(recs)} ({(time.time()-t0)/60.0:.1f} min)", flush=True)
    return cache


def write_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_shard_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def shard_complete(path: Path, n_images: int) -> bool:
    rows = read_shard_rows(path)
    return len(rows) == n_images * len(REFERENCE_MASKS)


def process_run(
    catalog: list[dict],
    geom_cache: dict,
    model_name: str,
    view: str,
    mode: str,
    device: torch.device,
    batch_size: int,
    out_dir: Path,
    mask_cache: dict[str, dict[str, np.ndarray]],
    num_workers: int,
) -> Path:
    path = shard_path(out_dir, model_name, view, mode)
    recs = [r for r in catalog if r["view"] == view]
    existing = read_shard_rows(path)
    done_ids = {row["dicom_id"] for row in existing}
    pending = [r for r in recs if r["dicom_id"] not in done_ids]
    if not pending:
        print(f"[skip] {model_name}/{view}/{mode} already complete ({path})", flush=True)
        return path
    run_dir = RESULTS / DEFAULT_RUNS[model_name][view][mode]
    print(f"[cam] load {run_dir.name}", flush=True)
    t0 = time.time()
    models, meta = load_fold_models(run_dir, device)
    print(
        f"[cam] {model_name}/{view}/{mode}: {len(pending)} remaining / {len(recs)} "
        f"images, {meta['n_folds']} folds, batch={batch_size}",
        flush=True,
    )
    rows: list[dict] = list(existing)
    by_dicom = geom_cache.get("by_dicom", {})
    seen = 0
    workers = max(1, int(num_workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            tensors = list(
                pool.map(
                    lambda rec: load_tensor(
                        rec["img_path"][mode],
                        meta["input_size"],
                        meta["mean"],
                        meta["std"],
                    ),
                    batch,
                )
            )
            x = torch.stack(tensors, dim=0)
            fold_maps = [gradcam_batch(model, x, device) for model in models]
            for i, rec in enumerate(batch):
                entry = by_dicom.get(rec["dicom_id"], {})
                cached = mask_cache[rec["dicom_id"]]
                orig_wh = cached["orig_wh"]
                geom = None if mode == "raw" else entry.get(mode)
                fold_cams = [fm[i] for fm in fold_maps]
                cam_pos = average_reprojected_positive(fold_cams, mode, geom, orig_wh)
                cam_sum = float(cam_pos.sum())
                for ref in REFERENCE_MASKS:
                    rows.append(
                        {
                            "dicom_id": rec["dicom_id"],
                            "subject_id": rec["subject_id"],
                            "view": view,
                            "model": model_name,
                            "mode": mode,
                            "reference_mask": ref,
                            "lung_mass": lung_attribution_mass(cam_pos, cached[ref]),
                            "top10_inside": top_attribution_inside_lung(cam_pos, cached[ref]),
                            "label": rec["label"],
                            "n_folds": meta["n_folds"],
                            "cam_sum": cam_sum,
                        }
                    )
            seen += len(batch)
            if seen % max(batch_size * 4, 64) == 0 or seen == len(pending):
                write_shard(path, rows)
                elapsed = time.time() - t0
                rate = seen / max(elapsed, 1e-6)
                remain = (len(pending) - seen) / max(rate, 1e-6)
                print(
                    f"[cam] {model_name}/{view}/{mode} {seen}/{len(pending)} "
                    f"{rate:.1f} img/s eta {remain/60.0:.1f} min saved={len(rows)} "
                    f"({elapsed/60.0:.1f} min)",
                    flush=True,
                )
    write_shard(path, rows)
    for model in models:
        del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[cam] wrote {path} n={len(rows)}", flush=True)
    return path


def fmt_median_iqr(values: np.ndarray) -> str:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return "NA"
    q1, med, q3 = np.percentile(v, [25.0, 50.0, 75.0])
    return f"{med:.2f} [{q1:.2f}–{q3:.2f}]"


def cluster_bootstrap_median(
    values: np.ndarray,
    subjects: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    subjects = np.asarray(subjects)
    finite = np.isfinite(values)
    values = values[finite]
    subjects = subjects[finite]
    if values.size == 0:
        return float("nan"), float("nan")
    uniq, inv = np.unique(subjects, return_inverse=True)
    n_subjects = int(uniq.size)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=n_subjects)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        drawn = rng.integers(0, n_subjects, size=n_subjects)
        lengths = counts[drawn]
        total = int(lengths.sum())
        starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        flat = np.repeat(offsets[drawn] - starts, lengths) + np.arange(total)
        picked = order[flat]
        meds[b] = float(np.median(values[picked]))
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def load_all_shards(out_dir: Path, catalog: list[dict]) -> list[dict]:
    rows = []
    expected = {}
    for rec in catalog:
        expected[(rec["view"], rec["dicom_id"])] = rec
    for model in MODELS:
        for view in VIEWS:
            for mode in MODES:
                path = shard_path(out_dir, model, view, mode)
                if not path.is_file():
                    continue
                with path.open(newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        rows.append(row)
    return rows


def paired_delta_table(rows: list[dict]) -> tuple[list[dict], str]:
    by_key: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row["dicom_id"],
            row["view"],
            row["model"],
            row["mode"],
            row["reference_mask"],
        )
        by_key[key] = row

    summary_rows = []
    md_lines = [
        "# Table. Lung-field attribution concentration on the outer-test set",
        "",
        "Metrics are computed on the five-fold mean Grad-CAM after each fold map is reprojected to original radiograph coordinates. Positive attribution is retained (`ReLU`); visualization min-max scaling is not applied. Hard-mask inputs are excluded. Results are reported independently for MedSAM3- and CheXMask-derived lung regions. Paired differences are crop minus raw on the same image. 95% CIs use a 2,000-replicate subject-level cluster bootstrap (seed 42).",
        "",
        "| View | Backbone | Input | Reference mask | N | Lung attribution mass, median [IQR] | Δ mass vs raw [95% CI] | Top-10% inside lung, median [IQR] | Δ top-10% vs raw [95% CI] |",
        "|---|---|---|---|---:|---|---|---|---|",
    ]
    for view in VIEWS:
        for model in MODELS:
            for ref in REFERENCE_MASKS:
                raw_mass = []
                raw_top = []
                raw_subj = []
                raw_ids = []
                for row in rows:
                    if (
                        row["view"] == view
                        and row["model"] == model
                        and row["mode"] == "raw"
                        and row["reference_mask"] == ref
                    ):
                        raw_mass.append(float(row["lung_mass"]))
                        raw_top.append(float(row["top10_inside"]))
                        raw_subj.append(row["subject_id"])
                        raw_ids.append(row["dicom_id"])
                raw_index = {did: i for i, did in enumerate(raw_ids)}
                for mode in MODES:
                    mass_vals = []
                    top_vals = []
                    d_mass = []
                    d_top = []
                    d_subj = []
                    for row in rows:
                        if not (
                            row["view"] == view
                            and row["model"] == model
                            and row["mode"] == mode
                            and row["reference_mask"] == ref
                        ):
                            continue
                        mass_vals.append(float(row["lung_mass"]))
                        top_vals.append(float(row["top10_inside"]))
                        i = raw_index.get(row["dicom_id"])
                        if i is None:
                            continue
                        d_mass.append(float(row["lung_mass"]) - raw_mass[i])
                        d_top.append(float(row["top10_inside"]) - raw_top[i])
                        d_subj.append(row["subject_id"])
                    mass_arr = np.asarray(mass_vals, dtype=np.float64)
                    top_arr = np.asarray(top_vals, dtype=np.float64)
                    d_mass_arr = np.asarray(d_mass, dtype=np.float64)
                    d_top_arr = np.asarray(d_top, dtype=np.float64)
                    if mode == "raw":
                        delta_mass_txt = "Ref."
                        delta_top_txt = "Ref."
                        d_mass_med = 0.0
                        d_top_med = 0.0
                        d_mass_ci = [0.0, 0.0]
                        d_top_ci = [0.0, 0.0]
                    else:
                        if d_mass_arr.size == 0 or not np.isfinite(np.nanmedian(d_mass_arr)):
                            d_mass_med = float("nan")
                            d_top_med = float("nan")
                            d_mass_ci = [float("nan"), float("nan")]
                            d_top_ci = [float("nan"), float("nan")]
                            delta_mass_txt = "NA"
                            delta_top_txt = "NA"
                        else:
                            d_mass_med = float(np.nanmedian(d_mass_arr))
                            d_top_med = float(np.nanmedian(d_top_arr))
                            lo_m, hi_m = cluster_bootstrap_median(d_mass_arr, np.asarray(d_subj))
                            lo_t, hi_t = cluster_bootstrap_median(d_top_arr, np.asarray(d_subj))
                            d_mass_ci = [lo_m, hi_m]
                            d_top_ci = [lo_t, hi_t]
                            delta_mass_txt = f"{d_mass_med:+.2f} ({lo_m:+.2f} to {hi_m:+.2f})"
                            delta_top_txt = f"{d_top_med:+.2f} ({lo_t:+.2f} to {hi_t:+.2f})"
                    if not np.any(np.isfinite(mass_arr)):
                        continue
                    rec = {
                        "view": view,
                        "backbone": MODEL_LABEL[model],
                        "model": model,
                        "input": MODE_LABEL[mode],
                        "mode": mode,
                        "reference_mask": REF_LABEL[ref],
                        "reference_mask_id": ref,
                        "n": int(np.isfinite(mass_arr).sum()) if mass_arr.size else 0,
                        "lung_mass_median_iqr": fmt_median_iqr(mass_arr),
                        "delta_mass_vs_raw": delta_mass_txt,
                        "top10_median_iqr": fmt_median_iqr(top_arr),
                        "delta_top10_vs_raw": delta_top_txt,
                        "lung_mass_median": float(np.nanmedian(mass_arr)) if mass_arr.size else float("nan"),
                        "delta_mass_median": d_mass_med,
                        "delta_mass_ci_low": d_mass_ci[0],
                        "delta_mass_ci_high": d_mass_ci[1],
                        "top10_median": float(np.nanmedian(top_arr)) if top_arr.size else float("nan"),
                        "delta_top10_median": d_top_med,
                        "delta_top10_ci_low": d_top_ci[0],
                        "delta_top10_ci_high": d_top_ci[1],
                    }
                    summary_rows.append(rec)
                    md_lines.append(
                        f"| {view} | {rec['backbone']} | {rec['input']} | {rec['reference_mask']} | "
                        f"{rec['n']} | {rec['lung_mass_median_iqr']} | {rec['delta_mass_vs_raw']} | "
                        f"{rec['top10_median_iqr']} | {rec['delta_top10_vs_raw']} |"
                    )
    md_lines.append("")
    return summary_rows, "\n".join(md_lines)


def write_combined_csv(out_dir: Path, rows: list[dict]) -> Path:
    path = out_dir / "per_image_lung_attribution.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in ROW_FIELDS})
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--geom-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="Debug: images per view budget")
    p.add_argument("--geometry-only", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--views", nargs="+", default=list(VIEWS))
    p.add_argument("--modes", nargs="+", default=list(MODES))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)

    print("[catalog] building outer-test index", flush=True)
    catalog = build_catalog(limit=int(args.limit))
    n_ap = sum(r["view"] == "AP" for r in catalog)
    n_pa = sum(r["view"] == "PA" for r in catalog)
    print(f"[catalog] {len(catalog)} images (AP {n_ap}, PA {n_pa})", flush=True)
    (out_dir / "catalog_counts.json").write_text(
        json.dumps(
            {
                "n": len(catalog),
                "AP": n_ap,
                "PA": n_pa,
                "n_subjects": len({r["subject_id"] for r in catalog}),
                "limit": int(args.limit),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    geom_cache = build_geometry_cache(catalog, out_dir, workers=int(args.geom_workers))
    by_dicom = geom_cache.get("by_dicom", {})
    n_before = len(catalog)
    catalog = [
        rec
        for rec in catalog
        if by_dicom.get(rec["dicom_id"], {}).get("medsam3_crop")
        and by_dicom.get(rec["dicom_id"], {}).get("chexmask_crop")
        and by_dicom.get(rec["dicom_id"], {}).get("original_size")
    ]
    print(
        f"[catalog] geometry-complete {len(catalog)}/{n_before} "
        f"(AP {sum(r['view']=='AP' for r in catalog)}, "
        f"PA {sum(r['view']=='PA' for r in catalog)})",
        flush=True,
    )
    if args.geometry_only:
        return

    if not args.summarize_only:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        for view in args.views:
            view_recs = [r for r in catalog if r["view"] == view]
            mask_cache = build_metric_mask_cache(view_recs, geom_cache)
            for model in args.models:
                for mode in args.modes:
                    process_run(
                        catalog,
                        geom_cache,
                        model,
                        view,
                        mode,
                        device,
                        int(args.batch_size),
                        out_dir,
                        mask_cache,
                        int(args.num_workers),
                    )
            del mask_cache

    rows = load_all_shards(out_dir, catalog)
    combined = write_combined_csv(out_dir, rows)
    summary_rows, md = paired_delta_table(rows)
    md_path = out_dir / "Supplementary_Table_lung_attribution_concentration.md"
    csv_path = out_dir / "Supplementary_Table_lung_attribution_concentration.csv"
    json_path = out_dir / "Supplementary_Table_lung_attribution_concentration.json"
    md_path.write_text(md + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["view"])
        writer.writeheader()
        writer.writerows(summary_rows)
    json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    results_md = RESULTS / "Supplementary_Table_lung_attribution_concentration.md"
    if int(args.limit) == 0 and out_dir.resolve() == OUT_DIR.resolve():
        results_md.write_text(md + "\n", encoding="utf-8")
    print(f"[done] per-image {combined} rows={len(rows)}", flush=True)
    print(f"[done] table {md_path}", flush=True)
    print(md, flush=True)


if __name__ == "__main__":
    main()
