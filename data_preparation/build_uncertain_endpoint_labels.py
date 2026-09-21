#!/usr/bin/env python3
"""Build an image-level JSON of CheXpert Pneumonia=-1 radiographs.

Only images whose patient is already in the outer-test split are kept.
This is the evaluation-endpoint set: the trained models stay frozen.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
DEFAULT_MIMIC_ROOT = Path(os.environ.get("MIMIC_CXR_ROOT", PACKAGE_ROOT / "data" / "mimic-cxr-jpg" / "2.1.0"))
DEFAULT_OUT = PACKAGE_ROOT / "pneumonia_labels_uncertain_outer_test.json"


def build_image_path(subject_id: int, study_id: int, dicom_id: str) -> str:
    sid = str(int(subject_id))
    std = str(int(study_id))
    prefix = sid[:2] if len(sid) >= 2 else sid
    return f"files/p{prefix}/p{sid}/s{std}/{dicom_id}.jpg"


def main() -> None:
    parser = argparse.ArgumentParser(description="Outer-test-patient Pneumonia=-1 label JSON")
    parser.add_argument("--mimic-root", type=str, default=str(DEFAULT_MIMIC_ROOT))
    parser.add_argument("--split-dir", type=str, default=str(PACKAGE_ROOT / "splits"))
    parser.add_argument("--out-json", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    mimic_root = Path(args.mimic_root).resolve()
    split_dir = Path(args.split_dir).resolve()
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
        v = unc.loc[(unc["ViewPosition"] == view) & (unc["subject_id"].isin(ot_sub))].copy()
        n_exist = 0
        for row in v.itertuples(index=False):
            rel = build_image_path(row.subject_id, row.study_id, row.dicom_id)
            abs_path = mimic_root / rel
            if not abs_path.is_file():
                continue
            n_exist += 1
            records.append(
                {
                    "subject_id": int(row.subject_id),
                    "study_id": int(row.study_id),
                    "dicom_id": str(row.dicom_id),
                    "view_position": str(row.ViewPosition),
                    "study_date": None,
                    "study_time": None,
                    "split": "outer_test",
                    "pneumonia": -1,
                    "image_rel_path": rel,
                    "image_abs_path": str(abs_path.resolve()),
                }
            )
        counts[view] = {"candidates": int(len(v)), "with_file": n_exist, "subjects": int(v["subject_id"].nunique())}
        print(f"[STAT] {view}: candidates={len(v):,} with_file={n_exist:,} subjects={v['subject_id'].nunique():,}")

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[DONE] {out_json} rows={len(records):,}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
