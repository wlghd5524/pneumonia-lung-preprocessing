#!/usr/bin/env python3
"""Score MedSAM3 and CheXMask lung masks on the 200 public PA expert contours.

The reference set is the PhysioNet dataset "Heart and lung segmentations for
MIMIC-CXR, MIMIC-CXR-JPG, and Montgomery County TB databases" (200 PA
radiographs). Some of them are in the pneumonia study cohort, so masks from the
primary pipeline exist for them. The manuscript scored those images with the
primary-pipeline masks and ran the same pipeline on the rest. Pass the primary
mask builds with --medsam3-cohort-root and --chexmask-cohort-root to do the
same; without them, every image is scored with the rebuilt masks.

Two steps, both run outside this repository:

  prepare    Write a records file for build_cxr_lung_seg_dataset.py and copy the
             expert masks to <out-dir>/reference/<dicom_id>.png.
  summarize  Pair the built masks with the references and write an
             identifier-free summary (counts, mean, SD, quartiles).

The metric code is evaluate_segmentation.py in this directory, so the 200-image
scores use the same Dice, IoU, and HD95 definitions as the AP scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import evaluate_segmentation as seg  # noqa: E402


def _outside_repo(path: Path) -> Path:
    path = path.resolve()
    if path == REPO or REPO in path.parents:
        raise SystemExit(
            f"Refusing to write inside the repository: {path}. File names and record "
            "files carry DICOM identifiers. Use a directory outside it."
        )
    return path


def prepare(args: argparse.Namespace) -> None:
    out_dir = _outside_repo(args.out_dir)
    gold_root = args.gold_root.resolve()
    mimic_root = args.mimic_root.resolve()
    links = list(csv.DictReader((gold_root / "MIMIC_links.csv").open(encoding="utf-8")))
    reference_dir = out_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)

    records = []
    missing_image = 0
    for row in links:
        dicom_id = Path(row["MIMIC-CXR_path"]).stem
        rel = row["MIMIC-CXR_path"].replace(".dcm", ".jpg")
        image = mimic_root / rel
        if not image.is_file():
            missing_image += 1
        shutil.copy2(gold_root / "lungs" / f"{Path(row['lungs_mask_path']).stem}.png", reference_dir / f"{dicom_id}.png")
        records.append(
            {
                "subject_id": int(row["subject_id"]),
                "study_id": int(row["study_id"]),
                "dicom_id": dicom_id,
                "view_position": "PA",
                "split": "reference",
                # The builder keeps only records with a 0/1 label. The value does not affect the mask.
                "pneumonia": 0,
                "image_rel_path": rel,
                "image_abs_path": str(image),
            }
        )
    if missing_image:
        raise SystemExit(f"{missing_image} radiographs are missing under {mimic_root}.")
    records_path = out_dir / "pa200_records.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")
    print(f"{len(records)} records -> {records_path}")
    print(f"{len(records)} expert masks -> {reference_dir}")
    print("Next, build the masks with the same settings as the primary analysis:")
    print(
        f"  python preprocessing/build_cxr_lung_seg_dataset.py --seg-mode medsam3 --view PA \\\n"
        f"    --labels-json {records_path} --source-image-root {mimic_root} \\\n"
        f"    --config configs/medsam3_full_lora_config.yaml --weights checkpoints/best_lora_weights.pt \\\n"
        f"    --out-root {out_dir}/medsam3"
    )
    print(
        f"  python preprocessing/build_cxr_lung_seg_dataset.py --seg-mode chexmask --view PA \\\n"
        f"    --labels-json {records_path} --source-image-root {mimic_root} \\\n"
        f"    --chexmask-csv $CHEXMASK_CSV --out-root {out_dir}/chexmask"
    )


def _flat_mask_dir(build_root: Path, target: Path, cohort_root: Path | None, overlap: set[str]) -> Path:
    """Symlink every built mask as <dicom_id>.npy so evaluate_segmentation can pair by stem.

    With cohort_root, images in the study cohort link to the primary-pipeline mask at the
    same relative path instead of the rebuilt one.
    """
    target.mkdir(parents=True, exist_ok=True)
    mask_root = build_root / "lung_mask_npy"
    masks = sorted(mask_root.rglob("*_lung_mask.npy"))
    if not masks:
        raise SystemExit(f"No *_lung_mask.npy files under {mask_root}.")
    for mask in masks:
        stem = mask.name[: -len("_lung_mask.npy")]
        source = mask
        if cohort_root is not None and stem in overlap:
            source = cohort_root / "lung_mask_npy" / mask.relative_to(mask_root)
            if not source.is_file():
                raise SystemExit(f"Primary-pipeline mask missing for a cohort image: {source}")
        (target / f"{stem}.npy").symlink_to(source)
    return target


def _cohort_labels(cohort_json: Path) -> dict[str, int]:
    """dicom_id -> 0/1 pneumonia label for the PA radiographs of the study cohort."""
    records = json.loads(cohort_json.read_text(encoding="utf-8"))
    return {
        str(r["dicom_id"]): int(r["pneumonia"])
        for r in records
        if isinstance(r, dict) and str(r.get("view_position", "")).upper() == "PA" and r.get("pneumonia") in (0, 1)
    }


def _block(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "dice": seg.summarize([r["dice"] for r in rows]),
        "iou": seg.summarize([r["iou"] for r in rows]),
        "hd95_px": seg.summarize([r["hd95_px"] for r in rows]),
    }


def summarize(args: argparse.Namespace) -> None:
    labels = _cohort_labels(args.cohort_json.resolve())
    methods = {
        "MedSAM3": (args.medsam3_root.resolve(), args.medsam3_cohort_root),
        "CheXMask": (args.chexmask_root.resolve(), args.chexmask_cohort_root),
    }
    result: dict = {
        "reference": "PhysioNet heart and lung segmentations, 200 PA radiographs",
        "metric_code": "cxr-segmentation-validation/evaluate_segmentation.py",
        "methods": {},
    }
    reference_index = seg.index_masks(args.reference_dir.resolve())
    overlap = {d for d in reference_index if d in labels}
    result["n_reference"] = len(reference_index)
    result["n_overlap_with_study_cohort"] = len(overlap)
    result["n_outside_study_cohort"] = len(reference_index) - len(overlap)
    with tempfile.TemporaryDirectory() as tmp:
        for name, (root, cohort_root) in methods.items():
            cohort_root = cohort_root.resolve() if cohort_root is not None else None
            flat = _flat_mask_dir(root, Path(tmp) / name, cohort_root, overlap)
            summary, rows = seg.score_directories(args.reference_dir.resolve(), flat)
            if summary["n_paired"] != len(reference_index):
                raise SystemExit(
                    f"{name}: {summary['n_paired']} of {len(reference_index)} references have a mask."
                )
            inside = [r for r in rows if r["file_stem"] in overlap]
            outside = [r for r in rows if r["file_stem"] not in overlap]
            result["methods"][name] = {
                "overlap_masks": "primary pipeline" if cohort_root is not None else "rebuilt",
                "pairing": {k: v for k, v in summary.items() if k.startswith("n_")},
                "all": _block(rows),
                "overlap_with_study_cohort": _block(inside),
                "outside_study_cohort": _block(outside),
                "overlap_by_pneumonia_label": {
                    "negative": _block([r for r in inside if labels[r["file_stem"]] == 0]),
                    "positive": _block([r for r in inside if labels[r["file_stem"]] == 1]),
                },
            }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"summary -> {args.summary_json}")
    for name, block in result["methods"].items():
        for part in ("all", "overlap_with_study_cohort", "outside_study_cohort"):
            b = block[part]
            print(
                f"{name:9s} {part:26s} N={b['n']:3d} "
                f"Dice {b['dice']['mean']:.3f} ± {b['dice']['std']:.3f} | "
                f"IoU {b['iou']['mean']:.3f} ± {b['iou']['std']:.3f} | "
                f"HD95 {b['hd95_px']['mean']:.1f} ± {b['hd95_px']['std']:.1f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="write the records file and the reference masks")
    p.add_argument("--gold-root", type=Path, required=True, help="the dataset's mimic_masks directory")
    p.add_argument("--mimic-root", type=Path, required=True, help="MIMIC-CXR-JPG 2.1.0 root")
    p.add_argument("--out-dir", type=Path, required=True, help="a directory outside this repository")
    p.set_defaults(func=prepare)

    s = sub.add_parser("summarize", help="score the built masks and write the summary")
    s.add_argument("--reference-dir", type=Path, required=True, help="<out-dir>/reference")
    s.add_argument("--medsam3-root", type=Path, required=True, help="--out-root of the medsam3 build")
    s.add_argument("--chexmask-root", type=Path, required=True, help="--out-root of the chexmask build")
    s.add_argument(
        "--medsam3-cohort-root",
        type=Path,
        default=None,
        help="--out-root of the primary PA medsam3 build; used for the images in the study cohort",
    )
    s.add_argument(
        "--chexmask-cohort-root",
        type=Path,
        default=None,
        help="--out-root of the primary PA chexmask build; used for the images in the study cohort",
    )
    s.add_argument("--cohort-json", type=Path, required=True, help="pneumonia_labels.json of the study cohort")
    s.add_argument("--summary-json", type=Path, required=True, help="identifier-free summary")
    s.set_defaults(func=summarize)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
