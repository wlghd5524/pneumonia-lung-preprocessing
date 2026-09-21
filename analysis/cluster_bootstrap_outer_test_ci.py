"""Recompute outer-test AUROC/AUPRC confidence intervals with a subject-level bootstrap.

The confidence intervals stored in each run's summary.json resample individual
radiographs, which treats multiple studies from the same patient as independent
observations and therefore reports intervals that are too narrow. This script
leaves the trained checkpoints and their outer-test predictions untouched and
only re-derives the intervals by resampling subject_id with replacement
(cluster bootstrap), keeping every image belonging to a drawn subject.
"""

import argparse
import csv
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = Path(os.environ.get("RESULTS_DIR", PACKAGE_ROOT / "results_pneumonia")).resolve()
MIN_RUN_DATE = "20260717"

SUBJECT_RE = re.compile(r"^p\d+$")


def subject_id_from_path(img_path: str) -> str:
    """MIMIC-CXR layout: .../files/p10/p10003400/s52822559/<dicom_id>.jpg"""
    token = Path(img_path).parts[-3]
    if not SUBJECT_RE.fullmatch(token):
        raise ValueError(f"Cannot parse subject_id from {img_path!r} (got {token!r})")
    return token[1:]


def load_runs_file(path: Path) -> list[str]:
    runs = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not runs:
        raise RuntimeError(f"{path}: run list is empty")
    if len(set(runs)) != len(runs):
        raise RuntimeError(f"{path}: duplicate run entries found")
    return runs


def find_runs(results_root: Path, min_run_date: str, runs_file: Path | None = None):
    if runs_file is not None:
        run_dirs = []
        for run in load_runs_file(runs_file):
            run_dir = results_root / run
            pred_path = run_dir / "outer_test" / "outer_test_predictions.csv"
            if not (run_dir / "summary.json").is_file() or not pred_path.is_file():
                raise RuntimeError(f"Missing summary or outer-test predictions: {run_dir}")
            run_dirs.append(run_dir)
        return run_dirs

    paths = sorted(
        glob.glob(str(results_root / "*" / "summary.json"))
        + glob.glob(str(results_root / "*" / "*" / "summary.json"))
    )
    runs = []
    for path in paths:
        rel = os.path.relpath(path, results_root)
        if rel.split(os.sep)[0].split("_")[0] < min_run_date:
            continue
        run_dir = Path(path).parent
        if (run_dir / "outer_test" / "outer_test_predictions.csv").is_file():
            runs.append(run_dir)
    return runs


def load_run(run_dir: Path, results_root: Path) -> dict:
    with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    cfg = summary.get("config", {})
    outer = summary.get("outer_test", {}) or {}
    image_ci = outer.get("ci_95", {}) or {}

    subjects, labels, probs = [], [], []
    with (run_dir / "outer_test" / "outer_test_predictions.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            subjects.append(subject_id_from_path(row["img_path"]))
            labels.append(int(float(row["label"])))
            probs.append(float(row["prob_ensemble"]))

    return {
        "run": os.path.relpath(run_dir, results_root),
        "model": cfg.get("model_name"),
        "view": cfg.get("view"),
        "mode": cfg.get("data_mode"),
        "subjects": np.asarray(subjects),
        "labels": np.asarray(labels, dtype=np.int8),
        "probs": np.asarray(probs, dtype=np.float64),
        "image_ci": {
            "auc": image_ci.get("auc"),
            "auprc": image_ci.get("auprc"),
        },
    }


def cluster_bootstrap(
    subjects: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    n_boot: int,
    seed: int,
):
    """Resample subjects with replacement, keeping all images of each drawn subject."""
    codes, subject_code = np.unique(subjects, return_inverse=True)
    n_subjects = codes.size

    # Group image indices by subject as a flat array plus per-subject offsets,
    # so each bootstrap draw is a single vectorized gather.
    order = np.argsort(subject_code, kind="stable")
    counts = np.bincount(subject_code, minlength=n_subjects)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))

    rng = np.random.default_rng(seed)
    auc_samples = np.empty(n_boot, dtype=np.float64)
    ap_samples = np.empty(n_boot, dtype=np.float64)
    n_degenerate = 0
    filled = 0

    for _ in range(n_boot):
        drawn = rng.integers(0, n_subjects, size=n_subjects)
        lengths = counts[drawn]
        total = int(lengths.sum())
        starts_in_out = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        flat = np.repeat(offsets[drawn] - starts_in_out, lengths) + np.arange(total)
        picked = order[flat]

        y = labels[picked]
        p = probs[picked]
        if y.min() == y.max():
            n_degenerate += 1
            continue
        auc_samples[filled] = roc_auc_score(y, p)
        ap_samples[filled] = average_precision_score(y, p)
        filled += 1

    if filled == 0:
        raise RuntimeError("Every bootstrap replicate was single-class; cannot form an interval.")

    return {
        "n_subjects": int(n_subjects),
        "n_boot_effective": int(filled),
        "n_degenerate": int(n_degenerate),
        "auc": np.percentile(auc_samples[:filled], [2.5, 97.5]).tolist(),
        "auprc": np.percentile(ap_samples[:filled], [2.5, 97.5]).tolist(),
        "auc_boot_mean": float(auc_samples[:filled].mean()),
        "auprc_boot_mean": float(ap_samples[:filled].mean()),
    }


def process(job) -> dict:
    data, n_boot, seed = job
    labels, probs = data["labels"], data["probs"]
    point_auc = float(roc_auc_score(labels, probs))
    point_ap = float(average_precision_score(labels, probs))
    boot = cluster_bootstrap(data["subjects"], labels, probs, n_boot=n_boot, seed=seed)

    img_auc = data["image_ci"]["auc"] or [None, None]
    img_ap = data["image_ci"]["auprc"] or [None, None]

    def width(lo, hi):
        return None if lo is None or hi is None else float(hi) - float(lo)

    return {
        "run": data["run"],
        "model": data["model"],
        "view": data["view"],
        "mode": data["mode"],
        "n_images": int(labels.size),
        "n_subjects": boot["n_subjects"],
        "n_pos": int(labels.sum()),
        "test_auc": point_auc,
        "auc_cluster_lo": boot["auc"][0],
        "auc_cluster_hi": boot["auc"][1],
        "auc_image_lo": img_auc[0],
        "auc_image_hi": img_auc[1],
        "auc_cluster_width": width(boot["auc"][0], boot["auc"][1]),
        "auc_image_width": width(img_auc[0], img_auc[1]),
        "test_auprc": point_ap,
        "auprc_cluster_lo": boot["auprc"][0],
        "auprc_cluster_hi": boot["auprc"][1],
        "auprc_image_lo": img_ap[0],
        "auprc_image_hi": img_ap[1],
        "auprc_cluster_width": width(boot["auprc"][0], boot["auprc"][1]),
        "auprc_image_width": width(img_ap[0], img_ap[1]),
        "n_boot_effective": boot["n_boot_effective"],
        "n_degenerate_replicates": boot["n_degenerate"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--runs-file", type=Path, default=None, help="Optional one-run-per-line manifest")
    parser.add_argument("--min-run-date", default=MIN_RUN_DATE)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-csv", type=Path, default=RESULTS_ROOT / "_cluster_bootstrap_ci_70runs.csv")
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=RESULTS_ROOT / "_outer_test_predictions_with_subject.csv",
        help="Tidy per-image table of subject_id / label / probability used as bootstrap input",
    )
    args = parser.parse_args()

    run_dirs = find_runs(args.results_root, args.min_run_date, args.runs_file)
    print(f"found {len(run_dirs)} runs", flush=True)
    loaded = [load_run(d, args.results_root) for d in run_dirs]

    with args.predictions_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "model", "view", "mode", "subject_id", "label", "prob_ensemble"])
        for data in loaded:
            for subject, label, prob in zip(data["subjects"], data["labels"], data["probs"]):
                writer.writerow([data["run"], data["model"], data["view"], data["mode"], subject, int(label), prob])
    print(f"wrote {args.predictions_csv}", flush=True)

    # Distinct but reproducible stream per run.
    jobs = [(data, int(args.n_boot), int(args.seed) + i) for i, data in enumerate(loaded)]
    results = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        for i, res in enumerate(pool.map(process, jobs), start=1):
            results.append(res)
            print(
                f"[{i}/{len(jobs)}] {res['view']} {res['mode']:<14} {res['model']:<26} "
                f"AUROC={res['test_auc']:.4f} "
                f"cluster=[{res['auc_cluster_lo']:.4f},{res['auc_cluster_hi']:.4f}] "
                f"image=[{res['auc_image_lo']:.4f},{res['auc_image_hi']:.4f}]",
                flush=True,
            )

    results.sort(key=lambda r: r["run"])
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {args.out_csv}", flush=True)

    auc_ratio = np.mean([r["auc_cluster_width"] / r["auc_image_width"] for r in results])
    ap_ratio = np.mean([r["auprc_cluster_width"] / r["auprc_image_width"] for r in results])
    print(f"mean CI width ratio (cluster / image): AUROC={auc_ratio:.3f}, AUPRC={ap_ratio:.3f}", flush=True)


if __name__ == "__main__":
    main()
