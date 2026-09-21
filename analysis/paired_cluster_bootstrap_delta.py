"""Paired subject-level cluster bootstrap for preprocessing effects (dAUROC / dAUPRC).

The study's primary question is whether lung segmentation or cropping changes
performance, so the quantity that needs an interval is the *difference* between
two preprocessing variants, not each variant on its own.

Two runs that share an outer-test split are evaluated on exactly the same
radiographs, so their errors are strongly correlated. Drawing one subject
resample and scoring both runs on it (a paired bootstrap) cancels the shared
sampling noise and yields a far tighter, correctly specified interval for the
difference than comparing two independent marginal intervals would.

Outputs
  1. per (view, backbone, mode_a -> mode_b) contrast
  2. mean contrast across backbones, using one shared subject resample per
     replicate so the across-backbone average keeps the same pairing
"""

import argparse
import csv
import glob
import itertools
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = Path(os.environ.get("RESULTS_DIR", PACKAGE_ROOT / "results_pneumonia")).resolve()
MIN_RUN_DATE = "20260717"

MODES = ["raw", "medsam3_seg", "medsam3_crop", "chexmask_seg", "chexmask_crop"]
BASELINE_MODE = "raw"


def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUROC, tie-aware; matches sklearn.roc_auc_score."""
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fast_ap(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision, tie-aware; matches sklearn.average_precision_score."""
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    desc = np.argsort(-p, kind="mergesort")
    y_s = y[desc]
    p_s = p[desc]
    last = np.where(np.diff(p_s))[0]
    idx = np.concatenate([last, [y.size - 1]])
    tp = np.cumsum(y_s)[idx]
    fp = (idx + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def image_key(path: str) -> str:
    """Stable id shared by the raw, masked and cropped copies of one radiograph."""
    return "/".join(Path(path).parts[-4:])


def subject_of(key: str) -> str:
    return key.split("/")[1][1:]


def load_runs_file(path: Path) -> list[str]:
    runs = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not runs:
        raise RuntimeError(f"{path}: run list is empty")
    if len(set(runs)) != len(runs):
        raise RuntimeError(f"{path}: duplicate run entries found")
    return runs


def load_run_entry(results_root: Path, run_dir: Path) -> tuple[tuple[str, str, str], dict]:
    summary_path = run_dir / "summary.json"
    pred_path = run_dir / "outer_test" / "outer_test_predictions.csv"
    if not summary_path.is_file() or not pred_path.is_file():
        raise RuntimeError(f"Missing summary or outer-test predictions: {run_dir}")
    with summary_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f).get("config", {})
    table = {}
    with pred_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            table[image_key(row["img_path"])] = (
                int(float(row["label"])),
                float(row["prob_ensemble"]),
            )
    key = (cfg["view"], cfg["data_mode"], cfg["model_name"])
    return key, {
        "run": os.path.relpath(run_dir, results_root),
        "table": table,
    }


def load_runs(results_root: Path, min_run_date: str, runs_file: Path | None = None) -> dict:
    runs = {}
    if runs_file is not None:
        for run in load_runs_file(runs_file):
            key, entry = load_run_entry(results_root, results_root / run)
            if key in runs:
                raise RuntimeError(f"Duplicate experiment key in runs file: {key}")
            runs[key] = entry
        return runs

    paths = sorted(
        glob.glob(str(results_root / "*" / "summary.json"))
        + glob.glob(str(results_root / "*" / "*" / "summary.json"))
    )
    for path in paths:
        rel = os.path.relpath(path, results_root)
        if rel.split(os.sep)[0].split("_")[0] < min_run_date:
            continue
        key, entry = load_run_entry(results_root, Path(path).parent)
        runs[key] = entry
    return runs


def subject_blocks(keys):
    """Flat image index array grouped by subject, with per-subject offsets."""
    subjects = np.array([subject_of(k) for k in keys])
    _, codes = np.unique(subjects, return_inverse=True)
    n_subjects = int(codes.max()) + 1
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=n_subjects)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return order, counts, offsets, n_subjects


def draw(order, counts, offsets, n_subjects, rng):
    drawn = rng.integers(0, n_subjects, size=n_subjects)
    lengths = counts[drawn]
    total = int(lengths.sum())
    starts_in_out = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    flat = np.repeat(offsets[drawn] - starts_in_out, lengths) + np.arange(total)
    return order[flat]


def percentile_ci(samples: np.ndarray):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_p_value(samples: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: delta = 0."""
    n = samples.size
    p_le = float((samples <= 0).sum()) / n
    p_ge = float((samples >= 0).sum()) / n
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def paired_contrast(job) -> dict:
    (view, model, mode_a, mode_b, run_a, run_b, keys, y, pa, pb, n_boot, seed, paired) = job
    keys = list(keys)
    y = np.asarray(y, dtype=np.int8)
    pa = np.asarray(pa)
    pb = np.asarray(pb)

    point = {
        "auc_a": fast_auc(y, pa),
        "auc_b": fast_auc(y, pb),
        "auprc_a": fast_ap(y, pa),
        "auprc_b": fast_ap(y, pb),
    }

    order, counts, offsets, n_subjects = subject_blocks(keys)
    rng = np.random.default_rng(seed)
    d_auc = np.empty(n_boot)
    d_ap = np.empty(n_boot)
    filled = 0
    for _ in range(n_boot):
        picked = draw(order, counts, offsets, n_subjects, rng)
        y_b = y[picked]
        if y_b.min() == y_b.max():
            continue
        d_auc[filled] = fast_auc(y_b, pb[picked]) - fast_auc(y_b, pa[picked])
        d_ap[filled] = fast_ap(y_b, pb[picked]) - fast_ap(y_b, pa[picked])
        filled += 1
    d_auc = d_auc[:filled]
    d_ap = d_ap[:filled]

    # Same contrast with independent resamples, reported only to document how much
    # the pairing tightens the interval.
    rng_u = np.random.default_rng(seed + 5_000_000)
    u_auc = np.empty(n_boot)
    u_ap = np.empty(n_boot)
    u_filled = 0
    for _ in range(n_boot):
        p1 = draw(order, counts, offsets, n_subjects, rng_u)
        p2 = draw(order, counts, offsets, n_subjects, rng_u)
        y1, y2 = y[p1], y[p2]
        if y1.min() == y1.max() or y2.min() == y2.max():
            continue
        u_auc[u_filled] = fast_auc(y2, pb[p2]) - fast_auc(y1, pa[p1])
        u_ap[u_filled] = fast_ap(y2, pb[p2]) - fast_ap(y1, pa[p1])
        u_filled += 1
    u_auc_lo, u_auc_hi = percentile_ci(u_auc[:u_filled])
    u_ap_lo, u_ap_hi = percentile_ci(u_ap[:u_filled])

    auc_lo, auc_hi = percentile_ci(d_auc)
    ap_lo, ap_hi = percentile_ci(d_ap)
    return {
        "view": view,
        "model": model,
        "mode_a": mode_a,
        "mode_b": mode_b,
        "contrast": f"{mode_b}_minus_{mode_a}",
        "vs_raw": mode_a == BASELINE_MODE,
        "pairing": "paired" if paired else "unpaired_split_mismatch",
        "n_images_paired": len(keys),
        "n_subjects_paired": n_subjects,
        "auc_a": point["auc_a"],
        "auc_b": point["auc_b"],
        "delta_auc": point["auc_b"] - point["auc_a"],
        "delta_auc_lo": auc_lo,
        "delta_auc_hi": auc_hi,
        "delta_auc_ci_width": auc_hi - auc_lo,
        "delta_auc_excludes_zero": bool(auc_lo > 0 or auc_hi < 0),
        "delta_auc_p": bootstrap_p_value(d_auc),
        "auprc_a": point["auprc_a"],
        "auprc_b": point["auprc_b"],
        "delta_auprc": point["auprc_b"] - point["auprc_a"],
        "delta_auprc_lo": ap_lo,
        "delta_auprc_hi": ap_hi,
        "delta_auprc_ci_width": ap_hi - ap_lo,
        "delta_auprc_excludes_zero": bool(ap_lo > 0 or ap_hi < 0),
        "delta_auprc_p": bootstrap_p_value(d_ap),
        "delta_auc_ci_width_unpaired": u_auc_hi - u_auc_lo,
        "delta_auprc_ci_width_unpaired": u_ap_hi - u_ap_lo,
        "pairing_width_reduction_auc": 1.0 - (auc_hi - auc_lo) / (u_auc_hi - u_auc_lo),
        "prob_corr_a_b": float(np.corrcoef(pa, pb)[0, 1]),
        "n_boot_effective": filled,
        "run_a": run_a,
        "run_b": run_b,
    }


def pooled_contrast(job) -> dict:
    """Mean delta across backbones under one shared subject resample per replicate."""
    (view, mode_a, mode_b, models, keys, y, pa_list, pb_list, n_boot, seed) = job
    keys = list(keys)
    y = np.asarray(y, dtype=np.int8)
    pa_list = [np.asarray(x) for x in pa_list]
    pb_list = [np.asarray(x) for x in pb_list]

    point = float(np.mean([
        (fast_auc(y, pb) - fast_auc(y, pa)) for pa, pb in zip(pa_list, pb_list)
    ]))
    point_ap = float(np.mean([
        (fast_ap(y, pb) - fast_ap(y, pa)) for pa, pb in zip(pa_list, pb_list)
    ]))

    order, counts, offsets, n_subjects = subject_blocks(keys)
    rng = np.random.default_rng(seed)
    d_auc = np.empty(n_boot)
    d_ap = np.empty(n_boot)
    filled = 0
    for _ in range(n_boot):
        picked = draw(order, counts, offsets, n_subjects, rng)
        y_b = y[picked]
        if y_b.min() == y_b.max():
            continue
        d_auc[filled] = float(np.mean([
            fast_auc(y_b, pb[picked]) - fast_auc(y_b, pa[picked])
            for pa, pb in zip(pa_list, pb_list)
        ]))
        d_ap[filled] = float(np.mean([
            fast_ap(y_b, pb[picked]) - fast_ap(y_b, pa[picked])
            for pa, pb in zip(pa_list, pb_list)
        ]))
        filled += 1
    d_auc = d_auc[:filled]
    d_ap = d_ap[:filled]

    auc_lo, auc_hi = percentile_ci(d_auc)
    ap_lo, ap_hi = percentile_ci(d_ap)
    return {
        "view": view,
        "mode_a": mode_a,
        "mode_b": mode_b,
        "contrast": f"{mode_b}_minus_{mode_a}",
        "n_models": len(models),
        "models": "|".join(models),
        "n_images_paired": len(keys),
        "n_subjects_paired": n_subjects,
        "mean_delta_auc": point,
        "mean_delta_auc_lo": auc_lo,
        "mean_delta_auc_hi": auc_hi,
        "mean_delta_auc_excludes_zero": bool(auc_lo > 0 or auc_hi < 0),
        "mean_delta_auc_p": bootstrap_p_value(d_auc),
        "mean_delta_auprc": point_ap,
        "mean_delta_auprc_lo": ap_lo,
        "mean_delta_auprc_hi": ap_hi,
        "mean_delta_auprc_excludes_zero": bool(ap_lo > 0 or ap_hi < 0),
        "mean_delta_auprc_p": bootstrap_p_value(d_ap),
        "n_boot_effective": filled,
    }


def build_pair_jobs(runs, models, n_boot, seed):
    jobs = []
    counter = 0
    for view in ("AP", "PA"):
        for model in models:
            for mode_a, mode_b in itertools.combinations(MODES, 2):
                a = runs.get((view, mode_a, model))
                b = runs.get((view, mode_b, model))
                if a is None or b is None:
                    continue
                shared = sorted(set(a["table"]) & set(b["table"]))
                paired = len(shared) == len(a["table"]) == len(b["table"])
                if not paired:
                    # Splits differ, so no honest pairing exists for this contrast.
                    jobs.append((
                        view, model, mode_a, mode_b, a["run"], b["run"],
                        shared,
                        [a["table"][k][0] for k in shared],
                        [a["table"][k][1] for k in shared],
                        [b["table"][k][1] for k in shared],
                        n_boot, seed + counter, False,
                    ))
                else:
                    jobs.append((
                        view, model, mode_a, mode_b, a["run"], b["run"],
                        shared,
                        [a["table"][k][0] for k in shared],
                        [a["table"][k][1] for k in shared],
                        [b["table"][k][1] for k in shared],
                        n_boot, seed + counter, True,
                    ))
                counter += 1
    return jobs


def build_pooled_jobs(runs, models, n_boot, seed):
    jobs = []
    counter = 0
    for view in ("AP", "PA"):
        for mode_a, mode_b in itertools.combinations(MODES, 2):
            usable, key_sets = [], []
            for model in models:
                a = runs.get((view, mode_a, model))
                b = runs.get((view, mode_b, model))
                if a is None or b is None:
                    continue
                if set(a["table"]) != set(b["table"]):
                    continue
                usable.append(model)
                key_sets.append(set(a["table"]))
            if not usable or len(usable) != len(models):
                continue
            shared = sorted(set.intersection(*key_sets))
            first = runs[(view, mode_a, usable[0])]["table"]
            jobs.append((
                view, mode_a, mode_b, usable, shared,
                [first[k][0] for k in shared],
                [[runs[(view, mode_a, m)]["table"][k][1] for k in shared] for m in usable],
                [[runs[(view, mode_b, m)]["table"][k][1] for k in shared] for m in usable],
                n_boot, seed + 10_000 + counter,
            ))
            counter += 1
    return jobs


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--runs-file", type=Path, default=None, help="Optional one-run-per-line manifest")
    parser.add_argument("--min-run-date", default=MIN_RUN_DATE)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-pairs", type=Path, default=RESULTS_ROOT / "_delta_paired_bootstrap_by_backbone.csv")
    parser.add_argument("--out-pooled", type=Path, default=RESULTS_ROOT / "_delta_paired_bootstrap_pooled.csv")
    args = parser.parse_args()

    runs = load_runs(args.results_root, args.min_run_date, args.runs_file)
    models = sorted({k[2] for k in runs})
    print(f"loaded {len(runs)} runs, {len(models)} backbones", flush=True)

    pair_jobs = build_pair_jobs(runs, models, int(args.n_boot), int(args.seed))
    pooled_jobs = build_pooled_jobs(runs, models, int(args.n_boot), int(args.seed))
    print(f"{len(pair_jobs)} per-backbone contrasts, {len(pooled_jobs)} pooled contrasts", flush=True)

    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        pair_rows = list(pool.map(paired_contrast, pair_jobs))
        pooled_rows = list(pool.map(pooled_contrast, pooled_jobs))

    pair_rows.sort(key=lambda r: (r["view"], r["mode_a"], r["mode_b"], r["model"]))
    pooled_rows.sort(key=lambda r: (r["view"], r["mode_a"], r["mode_b"]))
    write_csv(args.out_pairs, pair_rows)
    write_csv(args.out_pooled, pooled_rows)

    n_mismatch = sum(1 for r in pair_rows if r["pairing"] != "paired")
    n_sig = sum(1 for r in pair_rows if r["pairing"] == "paired" and r["delta_auc_excludes_zero"])
    n_paired = len(pair_rows) - n_mismatch
    print(f"\npaired contrasts: {n_paired}, split-mismatched: {n_mismatch}", flush=True)
    print(f"paired contrasts whose dAUROC CI excludes 0: {n_sig}/{n_paired}", flush=True)

    print("\npooled mean dAUROC vs raw (paired across backbones):", flush=True)
    for row in pooled_rows:
        if row["mode_a"] != BASELINE_MODE:
            continue
        print(
            f"  {row['view']} {row['mode_b']:<14} "
            f"{row['mean_delta_auc']:+.4f} [{row['mean_delta_auc_lo']:+.4f}, {row['mean_delta_auc_hi']:+.4f}] "
            f"p={row['mean_delta_auc_p']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
