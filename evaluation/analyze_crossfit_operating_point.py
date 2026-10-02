#!/usr/bin/env python3
"""Supplementary Table S5: post hoc cross-fitted Youden operating points.

For each of the 70 view × preprocessing × backbone runs, the outer-test
five-fold ensemble probability is already stored. This script does not
retrain. It splits outer-test patients into five complementary folds
(StratifiedGroupKFold, seed 42), estimates a Youden threshold on the
other four folds, applies it to the held-out fold, and pools the
confusion matrix. Subject-level cluster bootstrap CIs re-estimate the
five thresholds inside each replicate.
"""
from __future__ import annotations

import os

import argparse
import csv
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedGroupKFold

STUDY_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", STUDY_DIR / "results_pneumonia")).resolve()
RESULTS = RESULTS_DIR
CANON = RESULTS / "aggregates" / "_canonical_70runs.txt"
PRED_CSV = RESULTS / "aggregates" / "_outer_test_predictions_with_subject.csv"
OUT_MD = RESULTS / "paper_tables" / "Supplementary_Table_S5_crossfit_operating_point.md"
OUT_CSV = RESULTS / "paper_tables" / "Supplementary_Table_S5_crossfit_operating_point.csv"
OUT_JSON = RESULTS / "paper_tables" / "Supplementary_Table_S5_crossfit_operating_point.json"

RUN_RE = re.compile(
    r"_(AP|PA)_(raw|medsam3_seg|medsam3_crop|chexmask_seg|chexmask_crop)_cv_5fold"
)
ARCH_RE = re.compile(
    r"^\d{8}_\d{6}_(.+)_(?:AP|PA)_(?:raw|medsam3_seg|medsam3_crop|chexmask_seg|chexmask_crop)_cv_5fold"
)

ARCH_LABEL = {
    "resnet152": "ResNet-152",
    "densenet": "DenseNet-201",
    "convnextv2_base": "ConvNeXt V2-Base",
    "swint_base": "Swin V2-Base",
    "dinov3_base": "DINOv3-Base",
    "eva_x_base": "EVA-X-Base",
    "rad_dino": "RAD-DINO",
}
MODE_LABEL = {
    "raw": "Raw",
    "medsam3_seg": "MedSAM3 mask",
    "medsam3_crop": "MedSAM3 crop",
    "chexmask_seg": "CheXMask mask",
    "chexmask_crop": "CheXMask crop",
}
ARCH_ORDER = {k: i for i, k in enumerate(ARCH_LABEL)}
MODE_ORDER = {k: i for i, k in enumerate(MODE_LABEL)}
N_FOLDS = 5
N_BOOT = 2000
SEED = 42
N_BINS_ECE = 10


def parse_run(run: str) -> tuple[str, str, str]:
    m = RUN_RE.search(run)
    a = ARCH_RE.match(run)
    if not m or not a:
        raise ValueError(f"Cannot parse run name: {run}")
    return a.group(1), m.group(1), m.group(2)


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0 or p.size == 0 or int(y.min()) == int(y.max()):
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    if len(thr) == 0:
        return 0.5
    return float(thr[int(np.argmax(tpr - fpr))])


def calc_brier(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(np.float64)
    p = np.clip(p.astype(np.float64), 1e-7, 1.0 - 1e-7)
    return float(np.mean((p - y) ** 2))


def calc_ece(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS_ECE) -> float:
    y = y.astype(np.float64)
    p = np.clip(p.astype(np.float64), 1e-7, 1.0 - 1e-7)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        ece += (mask.sum() / max(1, n)) * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def metrics_from_counts(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    n = tn + fp + fn + tp
    acc = (tp + tn) / max(1, n)
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    denom = math.sqrt(
        float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    )
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    return {
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "mcc": float(mcc),
    }


def confusion_counts(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    y = y.astype(np.int8, copy=False)
    pred = pred.astype(np.int8, copy=False)
    tp = int(np.count_nonzero((pred == 1) & (y == 1)))
    tn = int(np.count_nonzero((pred == 0) & (y == 0)))
    fp = int(np.count_nonzero((pred == 1) & (y == 0)))
    fn = int(np.count_nonzero((pred == 0) & (y == 1)))
    return tn, fp, fn, tp


def assign_subject_folds(subjects: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, int]:
    """One fold id (0..4) per subject, stratified by any-positive image label."""
    order = np.argsort(subjects, kind="stable")
    subj_sorted = subjects[order]
    lab_sorted = labels[order]
    uniq, start_idx = np.unique(subj_sorted, return_index=True)
    y_subj = np.array(
        [
            int(lab_sorted[s:e].max())
            for s, e in zip(start_idx, list(start_idx[1:]) + [len(subj_sorted)])
        ],
        dtype=np.int8,
    )
    x = np.arange(len(uniq))
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_of: dict[str, int] = {}
    for fold_i, (_, test_idx) in enumerate(splitter.split(x, y_subj, groups=uniq)):
        for i in test_idx:
            fold_of[str(uniq[i])] = int(fold_i)
    if len(fold_of) != len(uniq):
        raise RuntimeError("Fold assignment missed subjects")
    return fold_of


def crossfit_point(y: np.ndarray, p: np.ndarray, folds: np.ndarray) -> dict:
    pred = np.empty_like(y, dtype=np.int8)
    thresholds = []
    for k in range(N_FOLDS):
        tune = folds != k
        ev = folds == k
        thr = youden_threshold(y[tune], p[tune])
        thresholds.append(thr)
        pred[ev] = (p[ev] >= thr).astype(np.int8)
    tn, fp, fn, tp = confusion_counts(y, pred)
    out = metrics_from_counts(tn, fp, fn, tp)
    out.update(
        {
            "thresholds": thresholds,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "pred": pred,
        }
    )
    return out


def bootstrap_metrics(
    y: np.ndarray,
    p: np.ndarray,
    folds: np.ndarray,
    subject_code: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, list[float]]:
    n_subjects = int(subject_code.max()) + 1
    order = np.argsort(subject_code, kind="stable")
    counts = np.bincount(subject_code, minlength=n_subjects)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    rng = np.random.default_rng(seed)

    keys = ("accuracy", "sensitivity", "specificity", "precision", "f1", "mcc")
    samples = {k: np.empty(n_boot, dtype=np.float64) for k in keys}
    filled = 0
    n_degenerate = 0

    for _ in range(n_boot):
        drawn = rng.integers(0, n_subjects, size=n_subjects)
        lengths = counts[drawn]
        total = int(lengths.sum())
        starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        flat = np.repeat(offsets[drawn] - starts, lengths) + np.arange(total)
        picked = order[flat]
        yb = y[picked]
        pb = p[picked]
        # Keep the original patient-to-fold map; do not re-split in bootstrap.
        fb = folds[picked]
        if int(yb.min()) == int(yb.max()):
            n_degenerate += 1
            continue
        pred = np.empty(total, dtype=np.int8)
        ok = True
        for k in range(N_FOLDS):
            ev = fb == k
            if not np.any(ev):
                ok = False
                break
            tune = ~ev
            thr = youden_threshold(yb[tune], pb[tune])
            pred[ev] = (pb[ev] >= thr).astype(np.int8)
        if not ok:
            n_degenerate += 1
            continue
        tn, fp, fn, tp = confusion_counts(yb, pred)
        met = metrics_from_counts(tn, fp, fn, tp)
        for k in keys:
            samples[k][filled] = met[k]
        filled += 1

    if filled == 0:
        raise RuntimeError("Every bootstrap replicate was degenerate")
    ci = {
        k: np.percentile(samples[k][:filled], [2.5, 97.5]).tolist() for k in keys
    }
    ci["n_boot_effective"] = int(filled)
    ci["n_degenerate"] = int(n_degenerate)
    return ci


def process_job(job: dict) -> dict:
    y = job["y"]
    p = job["p"]
    folds = job["folds"]
    subject_code = job["subject_code"]
    point = crossfit_point(y, p, folds)
    ci = bootstrap_metrics(y, p, folds, subject_code, job["n_boot"], job["seed"])
    row = {
        "run": job["run"],
        "arch": job["arch"],
        "view": job["view"],
        "mode": job["mode"],
        "backbone": ARCH_LABEL[job["arch"]],
        "preprocessing": MODE_LABEL[job["mode"]],
        "n_images": int(y.size),
        "n_subjects": int(subject_code.max()) + 1,
        "n_pos": int(y.sum()),
        "thr_1": point["thresholds"][0],
        "thr_2": point["thresholds"][1],
        "thr_3": point["thresholds"][2],
        "thr_4": point["thresholds"][3],
        "thr_5": point["thresholds"][4],
        "tn": point["tn"],
        "fp": point["fp"],
        "fn": point["fn"],
        "tp": point["tp"],
        "brier": calc_brier(y, p),
        "ece": calc_ece(y, p),
        "n_boot_effective": ci["n_boot_effective"],
        "n_degenerate": ci["n_degenerate"],
    }
    for key in ("accuracy", "sensitivity", "specificity", "precision", "f1", "mcc"):
        row[key] = point[key]
        row[f"{key}_lo"] = ci[key][0]
        row[f"{key}_hi"] = ci[key][1]
    return row


def fmt_thr(x: float) -> str:
    return f"{x:.4f}"


def fmt_ci(point: float, lo: float, hi: float) -> str:
    return f"{point:.4f} [{lo:.4f}, {hi:.4f}]"


def write_md(rows: list[dict]) -> None:
    lines = [
        "# Supplementary Table S5. Post hoc patient-level cross-fitted operating-point performance of the five-model ensembles",
        "",
        "Five Youden thresholds were estimated for each configuration using complementary outer-cohort thresholding folds. The table reports the five fold-specific thresholds, pooled confusion-matrix counts, threshold-dependent performance measures, and subject-level bootstrap 95% confidence intervals.",
        "",
        "The five-fold ensemble probability on the outer holdout was used (no retraining). Outer-test patients were split once with `StratifiedGroupKFold` (5 folds, shuffle, seed 42), stratified by whether the patient had any pneumonia-positive image. That patient-to-fold map is shared by every backbone and preprocessing within a view, and it is held fixed in the bootstrap: patients are resampled with replacement but never reassigned to a new fold. For fold *k*, the Youden threshold (sensitivity + specificity − 1) was estimated on the complementary patients and applied only to fold *k*. Image-level counts were then pooled. 95% CIs are subject-level cluster bootstrap (2000 replicates, seed 42). Inside each replicate the same frozen fold assignment is used; Youden thresholds are re-estimated on the complementary patients of that replicate. Brier and ECE are threshold-free on the ensemble probabilities (ECE uses 10 equal-width bins, matching training).",
        "",
        "| View | Preprocessing | Backbone | Thr 1 | Thr 2 | Thr 3 | Thr 4 | Thr 5 | TN | FP | FN | TP | Accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | Precision (95% CI) | F1 (95% CI) | MCC (95% CI) | Brier | ECE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            "| {view} | {prep} | {bb} | {t1} | {t2} | {t3} | {t4} | {t5} | {tn} | {fp} | {fn} | {tp} | {acc} | {sens} | {spec} | {prec} | {f1} | {mcc} | {brier} | {ece} |".format(
                view=r["view"],
                prep=r["preprocessing"],
                bb=r["backbone"],
                t1=fmt_thr(r["thr_1"]),
                t2=fmt_thr(r["thr_2"]),
                t3=fmt_thr(r["thr_3"]),
                t4=fmt_thr(r["thr_4"]),
                t5=fmt_thr(r["thr_5"]),
                tn=r["tn"],
                fp=r["fp"],
                fn=r["fn"],
                tp=r["tp"],
                acc=fmt_ci(r["accuracy"], r["accuracy_lo"], r["accuracy_hi"]),
                sens=fmt_ci(r["sensitivity"], r["sensitivity_lo"], r["sensitivity_hi"]),
                spec=fmt_ci(r["specificity"], r["specificity_lo"], r["specificity_hi"]),
                prec=fmt_ci(r["precision"], r["precision_lo"], r["precision_hi"]),
                f1=fmt_ci(r["f1"], r["f1_lo"], r["f1_hi"]),
                mcc=fmt_ci(r["mcc"], r["mcc_lo"], r["mcc_hi"]),
                brier=f"{r['brier']:.4f}",
                ece=f"{r['ece']:.4f}",
            )
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    canon = [ln.strip() for ln in CANON.read_text().splitlines() if ln.strip()]
    if len(canon) != 70:
        raise SystemExit(f"Expected 70 canonical runs, got {len(canon)}")

    print(f"Loading {PRED_CSV} ...", flush=True)
    df = pd.read_csv(
        PRED_CSV,
        dtype={"run": str, "subject_id": str, "label": np.int8, "prob_ensemble": np.float64},
        usecols=["run", "subject_id", "label", "prob_ensemble"],
    )
    present = set(df["run"].unique())
    missing = [r for r in canon if r not in present]
    extra = sorted(present - set(canon))
    if missing:
        raise SystemExit(f"Prediction CSV missing {len(missing)} canonical runs, e.g. {missing[:3]}")
    df = df[df["run"].isin(canon)].copy()

    fold_maps: dict[str, dict[str, int]] = {}
    fold_counts: dict[str, dict[str, int]] = {}
    for view in ("AP", "PA"):
        ref_run = next(r for r in canon if f"_{view}_raw_" in r)
        sub = df[df["run"] == ref_run]
        fmap = assign_subject_folds(
            sub["subject_id"].to_numpy(),
            sub["label"].to_numpy(),
            args.seed,
        )
        fold_maps[view] = fmap
        counts = {str(k): 0 for k in range(N_FOLDS)}
        for f in fmap.values():
            counts[str(f)] += 1
        fold_counts[view] = counts
        print(f"{view} thresholding folds: {counts} subjects (n={len(fmap)})", flush=True)

    jobs = []
    for run in canon:
        arch, view, mode = parse_run(run)
        sub = df[df["run"] == run]
        subjects = sub["subject_id"].astype(str).to_numpy()
        fmap = fold_maps[view]
        unknown = sorted(set(subjects) - set(fmap))
        if unknown:
            raise RuntimeError(f"{run}: {len(unknown)} subjects not in {view} fold map")
        uniq, code = np.unique(subjects, return_inverse=True)
        jobs.append(
            {
                "run": run,
                "arch": arch,
                "view": view,
                "mode": mode,
                "y": sub["label"].to_numpy(dtype=np.int8, copy=True),
                "p": sub["prob_ensemble"].to_numpy(dtype=np.float64, copy=True),
                "folds": np.array([fmap[s] for s in subjects], dtype=np.int8),
                "subject_code": code.astype(np.int32),
                "n_boot": int(args.n_boot),
                "seed": int(args.seed),
            }
        )

    rows: list[dict] = []
    print(f"Running {len(jobs)} configs × {args.n_boot} bootstrap replicates on {args.workers} workers ...", flush=True)
    with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = {ex.submit(process_job, job): job["run"] for job in jobs}
        done = 0
        for fut in as_completed(futs):
            run = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:
                raise RuntimeError(f"Failed on {run}: {exc}") from exc
            done += 1
            if done % 5 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} done", flush=True)

    rows.sort(
        key=lambda r: (
            0 if r["view"] == "AP" else 1,
            MODE_ORDER[r["mode"]],
            ARCH_ORDER[r["arch"]],
        )
    )

    fieldnames = [
        "view",
        "preprocessing",
        "backbone",
        "arch",
        "mode",
        "run",
        "n_images",
        "n_subjects",
        "n_pos",
        "thr_1",
        "thr_2",
        "thr_3",
        "thr_4",
        "thr_5",
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        "accuracy_lo",
        "accuracy_hi",
        "sensitivity",
        "sensitivity_lo",
        "sensitivity_hi",
        "specificity",
        "specificity_lo",
        "specificity_hi",
        "precision",
        "precision_lo",
        "precision_hi",
        "f1",
        "f1_lo",
        "f1_hi",
        "mcc",
        "mcc_lo",
        "mcc_hi",
        "brier",
        "ece",
        "n_boot_effective",
        "n_degenerate",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    write_md(rows)
    OUT_JSON.write_text(
        json.dumps(
            {
                "n_boot": args.n_boot,
                "seed": args.seed,
                "n_folds": N_FOLDS,
                "fold_subject_counts": fold_counts,
                "n_runs": len(rows),
                "extra_prediction_runs_ignored": extra,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_MD}", flush=True)
    print(f"Wrote {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
