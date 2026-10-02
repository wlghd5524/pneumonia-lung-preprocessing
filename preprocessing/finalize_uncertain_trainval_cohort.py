#!/usr/bin/env python3
"""Freeze the Pneumonia=-1 training cohort for the training-policy analysis.

Reads the cxr_*_uncertain_trainval_{ap,pa} manifests produced by
``SCOPE=trainval run_uncertain_endpoint_preprocess.sh`` and keeps only
radiographs that are file-backed in every requested preprocessing mode, so
Raw and cropped runs receive the identical set of added images.

Every kept image must belong to a patient already assigned to a CV fold in
splits/{VIEW}_fold_assignment.csv; outer-test patients abort the run.

Outputs (per view):
  splits/{VIEW}_uncertain_trainval.csv       subject_id, dicom_id, label(-1), view, split, fold
  splits/{VIEW}_uncertain_trainval_meta.json counts per mode / fold and dropped dicom_ids
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(STUDY_DIR, "training"))

from pneumonia_train import (  # noqa: E402
    SPLIT_CSV_COLUMNS,
    load_dicom_split_table,
    load_preprocessing_sample_identity,
    uncertain_trainval_data_root,
)

POLICY_MODES = ("raw", "medsam3_seg", "medsam3_crop", "chexmask_seg", "chexmask_crop")


def finalize_view(
    *,
    view: str,
    modes: list[str],
    data_base_dir: str,
    split_dir: str,
    source_image_root: str | None,
) -> None:
    view_key = view.upper()
    split_table = load_dicom_split_table(split_dir, view_key)
    subject_fold: dict[int, int] = {}
    outer_subjects: set[int] = set()
    for row in split_table.values():
        if row["split"] == "outer_test":
            outer_subjects.add(int(row["subject_id"]))
        else:
            subject_fold[int(row["subject_id"])] = int(row["fold"])

    identities: dict[str, dict] = {}
    for mode in modes:
        root = uncertain_trainval_data_root(data_base_dir, view_key, mode)
        identities[mode] = load_preprocessing_sample_identity(
            root,
            data_mode=mode,
            view_tag=view_key.lower(),
            source_image_root=source_image_root,
            strict_grouping=True,
            require_files=True,
        )
        print(f"[{view_key}] {mode:13s} file-backed -1 images: {len(identities[mode]):,}  ({root})")

    all_ids = set().union(*(set(v) for v in identities.values()))
    common = set.intersection(*(set(v) for v in identities.values()))
    dropped = {mode: sorted(all_ids - set(identities[mode])) for mode in modes}

    rows = []
    fold_counts: dict[int, int] = {}
    for dicom_id in sorted(common):
        info = identities[modes[0]][dicom_id]
        sid = int(info["subject_id"])
        label = int(info["label"])
        for mode in modes[1:]:
            other = identities[mode][dicom_id]
            assert int(other["subject_id"]) == sid, f"[{view_key}] subject mismatch {dicom_id} ({mode})"
            assert int(other["label"]) == label, f"[{view_key}] label mismatch {dicom_id} ({mode})"
        if label != -1:
            raise RuntimeError(f"[{view_key}] non-uncertain label {label} in uncertain manifest: {dicom_id}")
        if sid in outer_subjects:
            raise RuntimeError(f"[LEAKAGE] [{view_key}] outer-test subject in training cohort: sub{sid} ({dicom_id})")
        if sid not in subject_fold:
            raise RuntimeError(f"[{view_key}] subject not in fixed fold assignment: sub{sid} ({dicom_id})")
        if dicom_id in split_table:
            raise RuntimeError(f"[{view_key}] dicom_id already in explicit 0/1 cohort: {dicom_id}")
        fold = subject_fold[sid]
        fold_counts[fold] = fold_counts.get(fold, 0) + 1
        rows.append({
            "subject_id": sid,
            "dicom_id": dicom_id,
            "label": -1,
            "view": view_key,
            "split": "trainval",
            "fold": fold,
        })

    out_csv = os.path.join(split_dir, f"{view_key}_uncertain_trainval.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SPLIT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "view": view_key,
        "modes_intersected": modes,
        "data_roots": {m: uncertain_trainval_data_root(data_base_dir, view_key, m) for m in modes},
        "n_file_backed_by_mode": {m: len(identities[m]) for m in modes},
        "n_common": len(rows),
        "n_subjects": len({r["subject_id"] for r in rows}),
        "n_by_fold": {str(k): fold_counts[k] for k in sorted(fold_counts)},
        "dropped_dicom_ids_by_mode": dropped,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_meta = os.path.join(split_dir, f"{view_key}_uncertain_trainval_meta.json")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[{view_key}] common -1 images: {len(rows):,} / {meta['n_subjects']:,} subjects, by fold={meta['n_by_fold']}")
    for mode, ids in dropped.items():
        if ids:
            print(f"[{view_key}]   missing in {mode}: {len(ids)} (e.g. {ids[:3]})")
    print(f"[{view_key}] -> {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Pneumonia=-1 trainval cohort shared across preprocessing modes")
    parser.add_argument("--data-base-dir", default=os.environ.get("DATA_BASE_DIR", STUDY_DIR))
    parser.add_argument("--split-dir", default=os.path.join(STUDY_DIR, "splits"))
    parser.add_argument("--source-image-root", default=None, help="raw 원본 경로가 바뀐 경우 MIMIC-CXR-JPG 2.1.0 루트")
    parser.add_argument("--modes", nargs="+", default=["raw", "medsam3_crop"], choices=list(POLICY_MODES))
    parser.add_argument("--views", nargs="+", default=["AP", "PA"], choices=["AP", "PA"])
    args = parser.parse_args()

    data_base_dir = os.path.normpath(str(args.data_base_dir))
    split_dir = os.path.normpath(str(args.split_dir))
    for view in args.views:
        finalize_view(
            view=view,
            modes=list(args.modes),
            data_base_dir=data_base_dir,
            split_dir=split_dir,
            source_image_root=args.source_image_root,
        )
        print("")


if __name__ == "__main__":
    main()
