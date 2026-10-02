#!/usr/bin/env python3
"""Build an image-level JSON of CheXpert Pneumonia=-1 radiographs.

--scope outer_test (default)
    Only images whose patient is already in the outer-test split are kept.
    This is the evaluation-endpoint set: the trained models stay frozen.

--scope trainval
    Only images whose patient is already assigned to a CV fold are kept.
    These feed the training-policy sensitivity analysis (U-Zero / U-One
    retraining). Patients absent from the fixed split are dropped so the
    patient partition is unchanged, and outer-test patients never enter.
"""

from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

import pandas as pd

STUDY_DIR = Path(__file__).resolve().parents[1]
DATA_BASE_DIR = Path(os.environ.get("DATA_BASE_DIR", STUDY_DIR)).resolve()
DEFAULT_MIMIC_ROOT = Path(os.environ.get("MIMIC_CXR_ROOT", STUDY_DIR / "data" / "mimic-cxr-jpg" / "2.1.0"))
DEFAULT_OUT = {
    "outer_test": DATA_BASE_DIR / "pneumonia_labels_uncertain_outer_test.json",
    "trainval": DATA_BASE_DIR / "pneumonia_labels_uncertain_trainval.json",
}


def build_image_path(subject_id: int, study_id: int, dicom_id: str) -> str:
    sid = str(int(subject_id))
    std = str(int(study_id))
    prefix = sid[:2] if len(sid) >= 2 else sid
    return f"files/p{prefix}/p{sid}/s{std}/{dicom_id}.jpg"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pneumonia=-1 label JSON restricted to one fixed-split scope")
    parser.add_argument("--mimic-root", type=str, default=str(DEFAULT_MIMIC_ROOT))
    parser.add_argument("--split-dir", type=str, default=str(STUDY_DIR / "splits"))
    parser.add_argument("--scope", type=str, default="outer_test", choices=["outer_test", "trainval"])
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    mimic_root = Path(args.mimic_root).resolve()
    split_dir = Path(args.split_dir).resolve()
    scope = str(args.scope)
    chex = pd.read_csv(mimic_root / "mimic-cxr-2.0.0-chexpert.csv.gz", usecols=["subject_id", "study_id", "Pneumonia"])
    meta = pd.read_csv(
        mimic_root / "mimic-cxr-2.0.0-metadata.csv.gz",
        usecols=["subject_id", "study_id", "dicom_id", "ViewPosition"],
    )
    meta["ViewPosition"] = meta["ViewPosition"].fillna("").astype(str).str.upper()
    merged = meta.loc[meta["ViewPosition"].isin(["AP", "PA"])].merge(chex, on=["subject_id", "study_id"], how="inner")
    unc = merged.loc[pd.to_numeric(merged["Pneumonia"], errors="coerce") == -1].copy()
    unc["subject_id"] = unc["subject_id"].astype(int)
    unc["study_id"] = unc["study_id"].astype(int)
    unc["dicom_id"] = unc["dicom_id"].astype(str)

    records = []
    counts = {}
    for view in ("AP", "PA"):
        ot = pd.read_csv(split_dir / f"{view}_outer_test.csv")
        ot_sub = set(ot["subject_id"].astype(int))
        fa = pd.read_csv(split_dir / f"{view}_fold_assignment.csv")
        subject_fold = (
            fa[["subject_id", "fold"]].drop_duplicates().astype(int).set_index("subject_id")["fold"].to_dict()
        )
        overlap = ot_sub & set(subject_fold)
        if overlap:
            raise RuntimeError(f"{view}: outer_test and fold_assignment share subjects: {sorted(overlap)[:10]}")

        v_all = unc.loc[unc["ViewPosition"] == view]
        scope_sub = ot_sub if scope == "outer_test" else set(subject_fold)
        v = v_all.loc[v_all["subject_id"].isin(scope_sub)].copy()
        n_exist = 0
        fold_counts: dict[int, int] = {}
        for row in v.itertuples(index=False):
            rel = build_image_path(row.subject_id, row.study_id, row.dicom_id)
            abs_path = mimic_root / rel
            if not abs_path.is_file():
                continue
            n_exist += 1
            rec = {
                "subject_id": int(row.subject_id),
                "study_id": int(row.study_id),
                "dicom_id": str(row.dicom_id),
                "view_position": str(row.ViewPosition),
                "study_date": None,
                "study_time": None,
                "split": scope,
                "pneumonia": -1,
                "image_rel_path": rel,
                "image_abs_path": str(abs_path.resolve()),
            }
            if scope == "trainval":
                fold = int(subject_fold[int(row.subject_id)])
                rec["fold"] = fold
                fold_counts[fold] = fold_counts.get(fold, 0) + 1
            records.append(rec)

        n_outside = int((~v_all["subject_id"].isin(ot_sub | set(subject_fold))).sum())
        counts[view] = {
            "candidates": int(len(v)),
            "with_file": n_exist,
            "subjects": int(v["subject_id"].nunique()),
            "dropped_not_in_fixed_split": n_outside,
        }
        if scope == "trainval":
            counts[view]["with_file_by_fold"] = {str(k): fold_counts[k] for k in sorted(fold_counts)}
        print(
            f"[STAT] {view} ({scope}): candidates={len(v):,} with_file={n_exist:,} "
            f"subjects={v['subject_id'].nunique():,} dropped_not_in_fixed_split={n_outside:,}"
        )

    out_json = Path(args.out_json).resolve() if args.out_json else DEFAULT_OUT[scope]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[DONE] {out_json} rows={len(records):,}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
