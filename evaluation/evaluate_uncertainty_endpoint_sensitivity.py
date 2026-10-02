#!/usr/bin/env python3
"""Evaluation-endpoint sensitivity for CheXpert Pneumonia=-1 labels.

The 70 frozen 5-fold ensembles are not retrained. Extra radiographs from
already-held-out patients are scored once, then recoded as 0 (U-Zero) or
1 (U-One). This is not training-label sensitivity.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from paired_cluster_bootstrap_delta import (
    draw,
    fast_ap,
    fast_auc,
    percentile_ci,
    subject_blocks,
    subject_of,
)
from pneumonia_train import (
    CXRSpatialClassifier,
    SingleCXRDataset,
    _load_state_dict_compat,
    binary_positive_probability,
    collate_skip_none,
    default_eva_x_ckpt_path,
)

STUDY_DIR = Path(__file__).resolve().parents[1]
DATA_BASE_DIR = Path(os.environ.get("DATA_BASE_DIR", STUDY_DIR)).resolve()
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", STUDY_DIR / "results_pneumonia")).resolve()
RESULTS_ROOT = RESULTS_DIR
OUT_DIR = RESULTS_ROOT / "analysis_outputs" / "uncertainty_endpoint_sensitivity"
CANONICAL_RUNS = RESULTS_ROOT / "aggregates" / "_canonical_70runs.txt"
LABELS_JSON = DATA_BASE_DIR / "pneumonia_labels_uncertain_outer_test.json"
N_BOOT = 2000
BOOT_SEED = 42
MODES = ["raw", "medsam3_seg", "medsam3_crop", "chexmask_seg", "chexmask_crop"]
POLICIES = ("u_ignore", "u_zero", "u_one")


def image_key(path: str) -> str:
    parts = Path(path).parts
    if len(parts) < 4:
        return path
    return "/".join(parts[-4:])


def extra_roots(view: str) -> Dict[str, Path]:
    tag = view.lower()
    return {
        "raw": DATA_BASE_DIR / f"cxr_medsam3_lung_seg_uncertain_{tag}",
        "medsam3_seg": DATA_BASE_DIR / f"cxr_medsam3_lung_seg_uncertain_{tag}",
        "medsam3_crop": DATA_BASE_DIR / f"cxr_medsam3_lung_seg_cropped_uncertain_{tag}",
        "chexmask_seg": DATA_BASE_DIR / f"cxr_chexmask_lung_seg_uncertain_{tag}",
        "chexmask_crop": DATA_BASE_DIR / f"cxr_chexmask_lung_seg_cropped_uncertain_{tag}",
    }


def extra_manifest(view: str, mode: str) -> Path:
    tag = view.lower()
    roots = extra_roots(view)
    if mode in {"raw", "medsam3_seg", "medsam3_crop"}:
        seg = "medsam3"
    else:
        seg = "chexmask"
    if mode.endswith("_crop"):
        return roots[mode] / f"manifest_{tag}_{seg}_ok.json"
    return roots[mode] / f"manifest_{tag}_{seg}_ok.json"


def resolve_existing(path_str: str, data_root: Path, field: str) -> Optional[str]:
    p = Path(str(path_str))
    if p.is_file():
        return str(p)
    normalized = str(path_str).replace("\\", "/")
    markers = ["masked_cxr"] if field == "masked_cxr_jpg" else ["cropped_cxr", "files"]
    if field == "source_image_abs_path":
        markers = ["files"]
    for marker in markers:
        token = f"/{marker}/"
        if token not in normalized:
            continue
        tail = normalized.split(token, 1)[1]
        candidate = data_root / marker / tail if marker != "files" else data_root / marker / tail
        if candidate.is_file():
            return str(candidate)
        candidate2 = data_root / "files" / tail
        if candidate2.is_file():
            return str(candidate2)
    return None


def load_extra_items(view: str, mode: str, labels: List[dict]) -> Dict[str, dict]:
    by_dicom = {str(r["dicom_id"]): r for r in labels if str(r.get("view_position", "")).upper() == view}
    items: Dict[str, dict] = {}
    if mode == "raw":
        for dicom_id, rec in by_dicom.items():
            path = rec.get("image_abs_path")
            if path and Path(path).is_file():
                items[dicom_id] = {
                    "dicom_id": dicom_id,
                    "subject_id": int(rec["subject_id"]),
                    "path": str(path),
                    "key": image_key(path),
                }
        return items

    man_path = extra_manifest(view, mode)
    if not man_path.is_file():
        return items
    with man_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    field = {
        "medsam3_seg": "masked_cxr_jpg",
        "chexmask_seg": "masked_cxr_jpg",
        "medsam3_crop": "cropped_cxr_jpg",
        "chexmask_crop": "cropped_cxr_jpg",
    }[mode]
    data_root = extra_roots(view)[mode]
    for e in entries:
        dicom_id = str(e.get("dicom_id") or "")
        if dicom_id not in by_dicom:
            continue
        resolved = resolve_existing(e.get(field) or "", data_root, field)
        if not resolved:
            continue
        items[dicom_id] = {
            "dicom_id": dicom_id,
            "subject_id": int(e.get("subject_id") or by_dicom[dicom_id]["subject_id"]),
            "path": resolved,
            "key": image_key(resolved),
        }
    return items


def load_original_predictions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "outer_test" / "outer_test_predictions.csv"
    df = pd.read_csv(path)
    df["dicom_id"] = df["img_path"].map(lambda p: Path(str(p)).stem)
    df["key"] = df["img_path"].map(image_key)
    df["subject_id"] = df["key"].map(subject_of)
    df["source"] = "original_outer_test"
    return df


def run_config(run_dir: Path) -> dict:
    with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
        return dict(json.load(f).get("config", {}))


def checkpoint_paths(run_dir: Path, n_folds: int = 5) -> List[Path]:
    paths = [run_dir / f"fold_{i}" / "best_model.pth" for i in range(1, n_folds + 1)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints in {run_dir}: {missing}")
    return paths


@torch.no_grad()
def infer_paths(
    run_dir: Path,
    paths: Sequence[str],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    cfg = run_config(run_dir)
    model_name = str(cfg["model_name"])
    input_size = int(cfg.get("input_size", 512))
    dropout = float(cfg.get("dropout", 0.2))
    eva_dir = cfg.get("eva_x_ckpt_dir")
    if eva_dir in (None, "", "null"):
        try:
            eva_dir = default_eva_x_ckpt_path(model_name)
        except Exception:
            eva_dir = None
    mean = cfg.get("norm_mean") or [0.485, 0.456, 0.406]
    std = cfg.get("norm_std") or [0.229, 0.224, 0.225]
    use_bf16 = bool(cfg.get("use_bf16", True)) and not bool(cfg.get("use_fp32", False))

    items = [(p, 0, p) for p in paths]
    dataset = SingleCXRDataset(
        items,
        list(range(len(items))),
        input_size=input_size,
        train=False,
        norm_mean=mean,
        norm_std=std,
        split_name="uncertain_extra",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        collate_fn=collate_skip_none,
    )
    ckpts = checkpoint_paths(run_dir, int(cfg.get("max_folds") or cfg.get("n_folds") or 5))
    model = CXRSpatialClassifier(
        model_name=model_name,
        input_size=input_size,
        dropout=dropout,
        eva_x_ckpt_dir=eva_dir,
    ).to(device)
    model.eval()

    fold_probs: List[List[float]] = []
    ref_paths: Optional[List[str]] = None
    for ckpt in ckpts:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        _load_state_dict_compat(model, state)
        del state
        probs: List[float] = []
        out_paths: List[str] = []
        for batch in loader:
            if batch is None:
                continue
            imgs, _labels, batch_paths = batch
            imgs = imgs.to(device, non_blocking=True)
            if device.type == "cuda" and use_bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(imgs)
            else:
                outputs = model(imgs)
            batch_probs = binary_positive_probability(outputs.float()).cpu().numpy().tolist()
            probs.extend(batch_probs)
            out_paths.extend(list(batch_paths))
        fold_probs.append(probs)
        if ref_paths is None:
            ref_paths = out_paths
        elif ref_paths != out_paths:
            raise RuntimeError(f"Path order changed across folds in {run_dir}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    arrs = [np.asarray(p, dtype=np.float64) for p in fold_probs]
    ens = np.mean(np.stack(arrs, axis=0), axis=0)
    if ref_paths != list(paths):
        # collate may drop unreadable images; align by path
        index = {p: i for i, p in enumerate(ref_paths or [])}
        aligned = np.full(len(paths), np.nan, dtype=np.float64)
        aligned_folds = [np.full(len(paths), np.nan, dtype=np.float64) for _ in arrs]
        for i, p in enumerate(paths):
            j = index.get(p)
            if j is None:
                continue
            aligned[i] = ens[j]
            for k, fold_arr in enumerate(arrs):
                aligned_folds[k][i] = fold_arr[j]
        return aligned, aligned_folds
    return ens, arrs


def infer_all_runs(device: torch.device, labels: List[dict], out_dir: Path) -> None:
    pred_dir = out_dir / "extra_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    runs = [line.strip() for line in CANONICAL_RUNS.read_text(encoding="utf-8").splitlines() if line.strip()]
    cache: Dict[Tuple[str, str], Dict[str, dict]] = {}
    for run_name in tqdm(runs, desc="Runs"):
        run_dir = RESULTS_ROOT / run_name
        out_csv = pred_dir / f"{run_name}.csv"
        if out_csv.is_file():
            continue
        cfg = run_config(run_dir)
        view = str(cfg["view"]).upper()
        mode = str(cfg["data_mode"])
        key = (view, mode)
        if key not in cache:
            cache[key] = load_extra_items(view, mode, labels)
        extra = cache[key]
        if not extra:
            raise RuntimeError(f"No extra {view}/{mode} images. Preprocess first.")
        ordered = sorted(extra.values(), key=lambda r: r["dicom_id"])
        paths = [r["path"] for r in ordered]
        ens, folds = infer_paths(run_dir, paths, device=device)
        rows = []
        for i, rec in enumerate(ordered):
            row = {
                "dicom_id": rec["dicom_id"],
                "subject_id": rec["subject_id"],
                "img_path": rec["path"],
                "key": rec["key"],
                "label_uncertain": -1,
                "prob_ensemble": float(ens[i]) if np.isfinite(ens[i]) else "",
            }
            for k, fold_arr in enumerate(folds, start=1):
                val = fold_arr[i]
                row[f"prob_fold{k}"] = float(val) if np.isfinite(val) else ""
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)


def stacked_arrays(original: pd.DataFrame, extra: pd.DataFrame, policy: str):
    orig_y = original["label"].to_numpy(dtype=np.int8)
    orig_p = original["prob_ensemble"].to_numpy(dtype=np.float64)
    orig_keys = original["key"].astype(str).tolist()
    if policy == "u_ignore" or extra.empty:
        return orig_keys, orig_y, orig_p
    extra = extra.loc[pd.to_numeric(extra["prob_ensemble"], errors="coerce").notna()].copy()
    extra_y = np.zeros(len(extra), dtype=np.int8) if policy == "u_zero" else np.ones(len(extra), dtype=np.int8)
    extra_p = extra["prob_ensemble"].to_numpy(dtype=np.float64)
    extra_keys = extra["key"].astype(str).tolist()
    keys = orig_keys + extra_keys
    y = np.concatenate([orig_y, extra_y])
    p = np.concatenate([orig_p, extra_p])
    return keys, y, p


def metric_with_ci(keys: List[str], y: np.ndarray, p: np.ndarray, seed: int) -> dict:
    auc = fast_auc(y, p)
    ap = fast_ap(y, p)
    order, counts, offsets, n_subjects = subject_blocks(keys)
    rng = np.random.default_rng(seed)
    auc_s = np.empty(N_BOOT)
    ap_s = np.empty(N_BOOT)
    filled = 0
    for _ in range(N_BOOT):
        picked = draw(order, counts, offsets, n_subjects, rng)
        y_b = y[picked]
        if y_b.min() == y_b.max():
            continue
        auc_s[filled] = fast_auc(y_b, p[picked])
        ap_s[filled] = fast_ap(y_b, p[picked])
        filled += 1
    auc_s = auc_s[:filled]
    ap_s = ap_s[:filled]
    auc_lo, auc_hi = percentile_ci(auc_s)
    ap_lo, ap_hi = percentile_ci(ap_s)
    return {
        "n_images": int(len(y)),
        "n_pos": int(y.sum()),
        "n_subjects": int(n_subjects),
        "auroc": float(auc),
        "auroc_lo": auc_lo,
        "auroc_hi": auc_hi,
        "auprc": float(ap),
        "auprc_lo": ap_lo,
        "auprc_hi": ap_hi,
        "prevalence": float(y.mean()) if len(y) else float("nan"),
    }


def delta_with_ci(keys: List[str], y: np.ndarray, p_raw: np.ndarray, p_pp: np.ndarray, seed: int) -> dict:
    d_auc = fast_auc(y, p_pp) - fast_auc(y, p_raw)
    d_ap = fast_ap(y, p_pp) - fast_ap(y, p_raw)
    order, counts, offsets, n_subjects = subject_blocks(keys)
    rng = np.random.default_rng(seed)
    auc_s = np.empty(N_BOOT)
    ap_s = np.empty(N_BOOT)
    filled = 0
    for _ in range(N_BOOT):
        picked = draw(order, counts, offsets, n_subjects, rng)
        y_b = y[picked]
        if y_b.min() == y_b.max():
            continue
        auc_s[filled] = fast_auc(y_b, p_pp[picked]) - fast_auc(y_b, p_raw[picked])
        ap_s[filled] = fast_ap(y_b, p_pp[picked]) - fast_ap(y_b, p_raw[picked])
        filled += 1
    auc_s = auc_s[:filled]
    ap_s = ap_s[:filled]
    auc_lo, auc_hi = percentile_ci(auc_s)
    ap_lo, ap_hi = percentile_ci(ap_s)
    return {
        "delta_auroc": float(d_auc),
        "delta_auroc_lo": auc_lo,
        "delta_auroc_hi": auc_hi,
        "delta_auprc": float(d_ap),
        "delta_auprc_lo": ap_lo,
        "delta_auprc_hi": ap_hi,
        "n_images_paired": int(len(y)),
        "n_subjects_paired": int(n_subjects),
    }


def score_all(out_dir: Path) -> None:
    runs = [line.strip() for line in CANONICAL_RUNS.read_text(encoding="utf-8").splitlines() if line.strip()]
    pred_dir = out_dir / "extra_predictions"
    run_meta = []
    extras = {}
    originals = {}
    for run_name in runs:
        run_dir = RESULTS_ROOT / run_name
        cfg = run_config(run_dir)
        view = str(cfg["view"]).upper()
        mode = str(cfg["data_mode"])
        model = str(cfg.get("arch") or cfg["model_name"])
        extra_csv = pred_dir / f"{run_name}.csv"
        extra = pd.read_csv(extra_csv) if extra_csv.is_file() else pd.DataFrame()
        orig = load_original_predictions(run_dir)
        run_meta.append({"run": run_name, "view": view, "mode": mode, "model": model})
        extras[run_name] = extra
        originals[run_name] = orig

    metric_rows = []
    for meta in run_meta:
        orig = originals[meta["run"]]
        extra = extras[meta["run"]]
        for policy in POLICIES:
            keys, y, p = stacked_arrays(orig, extra, policy)
            stats = metric_with_ci(keys, y, p, seed=BOOT_SEED)
            metric_rows.append({**meta, "policy": policy, **stats, "n_extra": int(len(extra))})
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "endpoint_metrics.csv", index=False)

    delta_rows = []
    lookup = {}
    for meta in run_meta:
        lookup[(meta["view"], meta["model"], meta["mode"])] = meta["run"]
    models = sorted({m["model"] for m in run_meta})
    views = sorted({m["view"] for m in run_meta})
    for view in views:
        for model in models:
            raw_run = lookup.get((view, model, "raw"))
            if raw_run is None:
                continue
            for mode in MODES:
                if mode == "raw":
                    continue
                pp_run = lookup.get((view, model, mode))
                if pp_run is None:
                    continue
                for policy in POLICIES:
                    raw_keys, raw_y, raw_p = stacked_arrays(originals[raw_run], extras[raw_run], policy)
                    pp_keys, pp_y, pp_p = stacked_arrays(originals[pp_run], extras[pp_run], policy)
                    raw_map = {k: (yy, pp) for k, yy, pp in zip(raw_keys, raw_y, raw_p)}
                    keys, y, p_raw, p_pp = [], [], [], []
                    for k, yy, ppv in zip(pp_keys, pp_y, pp_p):
                        if k not in raw_map:
                            continue
                        y_raw, p0 = raw_map[k]
                        if int(y_raw) != int(yy):
                            continue
                        keys.append(k)
                        y.append(int(yy))
                        p_raw.append(p0)
                        p_pp.append(ppv)
                    if len(set(y)) < 2:
                        continue
                    stats = delta_with_ci(
                        keys,
                        np.asarray(y, dtype=np.int8),
                        np.asarray(p_raw, dtype=np.float64),
                        np.asarray(p_pp, dtype=np.float64),
                        seed=BOOT_SEED,
                    )
                    delta_rows.append({
                        "view": view,
                        "model": model,
                        "mode": mode,
                        "policy": policy,
                        **stats,
                    })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "endpoint_delta_vs_raw.csv", index=False)
    write_markdown(out_dir, metrics_df, delta_df)


def fmt_ci(point: float, lo: float, hi: float) -> str:
    return f"{point:.4f} [{lo:.4f}, {hi:.4f}]"


def write_markdown(out_dir: Path, metrics: pd.DataFrame, deltas: pd.DataFrame) -> None:
    lines = [
        "# Evaluation endpoint sensitivity to CheXpert uncertain labels",
        "",
        "Trained 5-fold ensembles were **not** retrained. Radiographs with CheXpert `Pneumonia = -1` were scored only if the patient was already in the outer-test split. Two recoding policies were applied at evaluation: **U-Zero** treats uncertain labels as 0, **U-One** treats them as 1. **U-Ignore** is the original 0/1 outer-test endpoint.",
        "",
        "## Discrimination",
        "",
        "| View | Backbone | Input | Policy | N | Prevalence | AUROC (95% CI) | AUPRC (95% CI) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    order = {"u_ignore": 0, "u_zero": 1, "u_one": 2}
    metrics = metrics.sort_values(
        ["view", "model", "mode", "policy"],
        key=lambda s: s.map(order) if s.name == "policy" else s,
    )
    for _, r in metrics.iterrows():
        lines.append(
            f"| {r['view']} | {r['model']} | {r['mode']} | {r['policy']} | "
            f"{int(r['n_images'])} | {r['prevalence']:.3f} | "
            f"{fmt_ci(r['auroc'], r['auroc_lo'], r['auroc_hi'])} | "
            f"{fmt_ci(r['auprc'], r['auprc_lo'], r['auprc_hi'])} |"
        )
    lines += [
        "",
        "## Preprocessing minus raw",
        "",
        "| View | Backbone | Input | Policy | ΔAUROC (95% CI) | ΔAUPRC (95% CI) | N paired |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    deltas = deltas.sort_values(
        ["view", "model", "mode", "policy"],
        key=lambda s: s.map(order) if s.name == "policy" else s,
    )
    for _, r in deltas.iterrows():
        lines.append(
            f"| {r['view']} | {r['model']} | {r['mode']} | {r['policy']} | "
            f"{fmt_ci(r['delta_auroc'], r['delta_auroc_lo'], r['delta_auroc_hi'])} | "
            f"{fmt_ci(r['delta_auprc'], r['delta_auprc_lo'], r['delta_auprc_hi'])} | "
            f"{int(r['n_images_paired'])} |"
        )
    (out_dir / "Supplementary_Table_uncertainty_endpoint_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation endpoint sensitivity for Pneumonia=-1")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--labels-json", type=str, default=str(LABELS_JSON))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()
    if not args.infer and not args.score:
        args.infer = True
        args.score = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.labels_json, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if args.infer:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        infer_all_runs(device, labels, out_dir)
    if args.score:
        score_all(out_dir)
    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    os.chdir(STUDY_DIR)
    main()
