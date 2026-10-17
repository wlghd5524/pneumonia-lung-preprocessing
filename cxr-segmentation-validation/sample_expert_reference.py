#!/usr/bin/env python3
"""Redraw the locked 100-image AP expert-reference sample.

A credentialed user supplies the local AP outer-test CSV and a local
MIMIC-CXR-JPG v2.1.0 tree. The script writes the selected list only to a
directory outside this repository. It does not read MedSAM3 masks, CheXMask
masks, or classifier outputs.

Selection order, with one NumPy Generator created from seed 42:

1. Dual-label subjects, in subject_id order. One label is kept.
2. Pneumonia-positive challenging subjects, then one image from each.
3. Pneumonia-positive routine subjects, then one image from each.
4. Pneumonia-negative challenging subjects, then one image from each.
5. Pneumonia-negative routine subjects, then one image from each.

Subject identifiers are sorted before each draw. Within a selected subject,
candidate DICOM identifiers are sorted, and one index is drawn. A challenging
subject contributes one image that meets the challenging rule. A routine
subject contributes one image that does not.

The saved sample file is then sorted. It is not shuffled. Sort keys are
pneumonia_label descending, stratum ascending, subject_id, and dicom_id.
The annotation list is a separate permutation of that sorted file, made with
a new Generator initialized with the same seed. That second generator does
not continue the sampling stream.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_PER_CLASS = 50
IDENTITY_COLUMNS = [
    "sample_id",
    "subject_id",
    "study_id",
    "dicom_id",
    "pneumonia_label",
    "stratum",
    "challenging_reasons",
    "is_portable",
    "support_devices",
    "consolidation",
    "pleural_effusion",
    "pleural_other",
    "recumbent",
    "procedure_description",
    "patient_orientation",
]
# SHA256 of those columns, without the local image path, from the locked sample.
LOCKED_IDENTITY_SHA256 = "2762de922d87c9961c7454d75ef9dd5ad5019a96f8b4f47ec3110b6f5f0be301"
# SHA256 of review_id and dicom_id in annotation order. That order is a
# separate permutation of the sorted sample, not the sample-file order.
LOCKED_ANNOTATION_ORDER_SHA256 = "0e53756acab9416ebcb80e6b7f4c1638ebf3e91b44b9e3d0321f8e193ed99389"


def positive_flag(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").eq(1.0)


def image_path(mimic_root: Path, subject_id: int, study_id: int, dicom_id: str) -> Path:
    sid = str(int(subject_id))
    return mimic_root / "files" / f"p{sid[:2]}" / f"p{sid}" / f"s{int(study_id)}" / f"{dicom_id}.jpg"


def load_frame(outer_csv: Path, mimic_root: Path) -> pd.DataFrame:
    outer = pd.read_csv(outer_csv)
    required = {"subject_id", "dicom_id", "label", "view"}
    missing = required.difference(outer.columns)
    if missing:
        raise ValueError(f"outer-test csv is missing columns: {sorted(missing)}")
    outer = outer.loc[outer["view"].astype(str).str.upper().eq("AP")].copy()
    outer["subject_id"] = outer["subject_id"].astype(int)
    outer["dicom_id"] = outer["dicom_id"].astype(str)
    outer["label"] = pd.to_numeric(outer["label"], errors="coerce").astype("Int64")
    outer = outer.loc[outer["label"].isin([0, 1])].copy()

    meta = pd.read_csv(
        mimic_root / "mimic-cxr-2.0.0-metadata.csv.gz",
        usecols=[
            "dicom_id",
            "subject_id",
            "study_id",
            "PerformedProcedureStepDescription",
            "ViewPosition",
            "PatientOrientationCodeSequence_CodeMeaning",
        ],
    )
    meta["subject_id"] = meta["subject_id"].astype(int)
    meta["dicom_id"] = meta["dicom_id"].astype(str)
    chex = pd.read_csv(
        mimic_root / "mimic-cxr-2.0.0-chexpert.csv.gz",
        usecols=[
            "subject_id",
            "study_id",
            "Consolidation",
            "Pleural Effusion",
            "Pleural Other",
            "Support Devices",
            "Pneumonia",
        ],
    )
    chex["subject_id"] = chex["subject_id"].astype(int)

    frame = outer.merge(meta, on=["subject_id", "dicom_id"], how="left", validate="one_to_one")
    if frame["study_id"].isna().any():
        raise RuntimeError(f"{int(frame['study_id'].isna().sum())} outer-test images have no DICOM metadata row")
    frame["study_id"] = frame["study_id"].astype(int)
    frame = frame.merge(chex, on=["subject_id", "study_id"], how="left", validate="many_to_one")
    if frame["Pneumonia"].isna().any():
        raise RuntimeError(f"{int(frame['Pneumonia'].isna().sum())} outer-test images have no CheXpert study row")

    procedure = frame["PerformedProcedureStepDescription"].fillna("").astype(str).str.upper()
    orientation = frame["PatientOrientationCodeSequence_CodeMeaning"].fillna("").astype(str).str.upper()
    frame["is_portable"] = procedure.str.contains("PORTABLE", regex=False) | procedure.str.contains(
        r"\bPORT\b", regex=True
    )
    frame["consolidation"] = positive_flag(frame["Consolidation"])
    frame["pleural_effusion"] = positive_flag(frame["Pleural Effusion"])
    frame["pleural_other"] = positive_flag(frame["Pleural Other"])
    frame["support_devices"] = positive_flag(frame["Support Devices"])
    frame["recumbent"] = orientation.eq("RECUMBENT")
    frame["image_path"] = [
        str(image_path(mimic_root, subject_id, study_id, dicom_id))
        for subject_id, study_id, dicom_id in zip(frame["subject_id"], frame["study_id"], frame["dicom_id"])
    ]
    frame["image_exists"] = [Path(path).is_file() for path in frame["image_path"]]
    return frame


def reason_lists(frame: pd.DataFrame) -> pd.Series:
    reasons = []
    for row in frame.itertuples(index=False):
        found = []
        if row.consolidation:
            found.append("consolidation")
        if row.pleural_effusion:
            found.append("pleural_effusion")
        if row.pleural_other:
            found.append("pleural_other")
        if row.recumbent:
            found.append("recumbent")
        reasons.append(found)
    return pd.Series(reasons, index=frame.index)


def assign_one_label(frame: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    chosen = {}
    for subject_id, part in frame.groupby("subject_id", sort=True):
        labels = sorted(int(value) for value in part["label"].dropna().unique())
        if len(labels) == 1:
            chosen[int(subject_id)] = labels[0]
        else:
            chosen[int(subject_id)] = int(rng.choice(np.asarray(labels, dtype=int)))
    return frame["subject_id"].map(chosen).astype(int)


def proportional_quota(n_flag: int, n_pool: int, n_draw: int) -> int:
    if n_pool <= 0:
        raise RuntimeError("class pool is empty")
    quota = int(np.floor((n_flag / n_pool) * n_draw + 0.5))
    if n_flag > 0 and quota == 0:
        quota = 1
    if n_flag < n_pool and quota == n_draw:
        quota = n_draw - 1
    if quota > n_flag or (n_draw - quota) > (n_pool - n_flag):
        raise RuntimeError(f"quota {quota} does not fit the pool of {n_pool}")
    return quota


def sample_class(
    eligible: pd.DataFrame,
    rng: np.random.Generator,
    n_challenging: int,
    n_routine: int,
) -> pd.DataFrame:
    challenging_ids = sorted(int(value) for value in eligible.loc[eligible["subject_is_challenging"], "subject_id"].unique())
    routine_ids = sorted(int(value) for value in eligible.loc[~eligible["subject_is_challenging"], "subject_id"].unique())
    if len(challenging_ids) < n_challenging or len(routine_ids) < n_routine:
        raise RuntimeError(
            f"pool too small: challenging {len(challenging_ids)}/{n_challenging}, "
            f"routine {len(routine_ids)}/{n_routine}"
        )
    picked_challenging = rng.choice(np.asarray(challenging_ids, dtype=int), size=n_challenging, replace=False)
    picked_routine = rng.choice(np.asarray(routine_ids, dtype=int), size=n_routine, replace=False)
    rows = []
    strata = []
    for stratum, subject_ids in (("challenging", picked_challenging), ("routine", picked_routine)):
        for subject_id in sorted(int(value) for value in subject_ids):
            part = eligible.loc[eligible["subject_id"].eq(subject_id)].sort_values("dicom_id")
            if stratum == "challenging":
                part = part.loc[part["image_is_challenging"]]
            else:
                part = part.loc[~part["image_is_challenging"]]
            take = int(rng.choice(np.arange(len(part))))
            rows.append(part.iloc[take])
            strata.append(stratum)
    drawn = pd.DataFrame(rows)
    drawn["stratum"] = strata
    return drawn


def build_sample(frame: pd.DataFrame, seed: int, n_per_class: int) -> pd.DataFrame:
    work = frame.loc[frame["image_exists"]].copy()
    work["reasons"] = reason_lists(work)
    work["image_is_challenging"] = work["reasons"].map(bool)
    rng = np.random.default_rng(seed)
    work["assigned_label"] = assign_one_label(work, rng)
    eligible = work.loc[work["label"].astype(int).eq(work["assigned_label"])].copy()
    subject_flag = eligible.groupby("subject_id")["image_is_challenging"].any()
    eligible["subject_is_challenging"] = eligible["subject_id"].map(subject_flag).astype(bool)

    pieces = []
    for label in (1, 0):
        part = eligible.loc[eligible["assigned_label"].eq(label)].copy()
        n_subjects = int(part["subject_id"].nunique())
        n_challenging = int(part.loc[part["subject_is_challenging"], "subject_id"].nunique())
        n_draw_challenging = proportional_quota(n_challenging, n_subjects, n_per_class)
        pieces.append(sample_class(part, rng, n_draw_challenging, n_per_class - n_draw_challenging))

    selected = pd.concat(pieces, ignore_index=True)
    if selected["subject_id"].duplicated().any() or selected["dicom_id"].duplicated().any():
        raise RuntimeError("sample contains a repeated patient or image")
    selected["pneumonia_label"] = selected["assigned_label"].astype(int)
    selected["challenging_reasons"] = selected["reasons"].map(lambda values: "|".join(values) if values else "")
    challenging_rows = selected["stratum"].eq("challenging")
    if selected.loc[challenging_rows, "challenging_reasons"].eq("").any():
        raise RuntimeError("a challenging image does not meet the pre-specified rule")
    if selected.loc[~challenging_rows, "challenging_reasons"].ne("").any():
        raise RuntimeError("a routine image meets the challenging rule")
    # Sample-file order. The annotation list is permuted later by a new generator.
    selected = selected.sort_values(
        ["pneumonia_label", "stratum", "subject_id", "dicom_id"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    selected.insert(0, "sample_id", np.arange(1, len(selected) + 1))
    keep = [
        "sample_id",
        "subject_id",
        "study_id",
        "dicom_id",
        "pneumonia_label",
        "stratum",
        "challenging_reasons",
        "is_portable",
        "support_devices",
        "consolidation",
        "pleural_effusion",
        "pleural_other",
        "recumbent",
        "PerformedProcedureStepDescription",
        "PatientOrientationCodeSequence_CodeMeaning",
        "image_path",
    ]
    out = selected[keep].copy()
    for column in ("is_portable", "support_devices", "consolidation", "pleural_effusion", "pleural_other", "recumbent"):
        out[column] = out[column].astype(int)
    return out.rename(
        columns={
            "PerformedProcedureStepDescription": "procedure_description",
            "PatientOrientationCodeSequence_CodeMeaning": "patient_orientation",
        }
    )


def identity_sha256(sample: pd.DataFrame) -> str:
    payload = sample[IDENTITY_COLUMNS].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def annotation_order_sha256(sample: pd.DataFrame, seed: int) -> str:
    """Permutation used for the contouring list, independent of the sampling stream."""
    order = np.random.default_rng(seed).permutation(len(sample))
    sheet = sample.iloc[order].reset_index(drop=True)
    payload = pd.DataFrame(
        {"review_id": np.arange(1, len(sheet) + 1), "dicom_id": sheet["dicom_id"].astype(str)}
    )
    return hashlib.sha256(payload.to_csv(index=False).encode("utf-8")).hexdigest()


def assert_outside_repository(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise SystemExit("Write the sample outside this repository. The file contains MIMIC identifiers.")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redraw the locked AP expert-reference sample")
    parser.add_argument("--outer-csv", type=Path, required=True)
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = assert_outside_repository(args.out_dir)
    frame = load_frame(args.outer_csv, args.mimic_root)
    missing = int((~frame["image_exists"]).sum())
    if missing:
        raise SystemExit(f"{missing} outer-test images have no local JPEG. The locked draw excluded none.")
    first = build_sample(frame, args.seed, args.n_per_class)
    second = build_sample(frame, args.seed, args.n_per_class)
    if not first.equals(second):
        raise SystemExit("repeated draws with the same seed disagreed")
    digest = identity_sha256(first)
    annotation_digest = annotation_order_sha256(first, args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "ap_outer_test_expert_sample.csv"
    first.to_csv(sample_path, index=False)
    counts = first.groupby(["pneumonia_label", "stratum"]).size()
    print(f"n={len(first)}")
    for (label, stratum), count in counts.items():
        print(f"label_{int(label)}_{stratum}={int(count)}")
    print(f"identity_sha256={digest}")
    if digest != LOCKED_IDENTITY_SHA256:
        print(f"locked_identity_sha256={LOCKED_IDENTITY_SHA256}")
        raise SystemExit("This draw does not match the locked sample.")
    print("identity_matches_locked_sample=true")
    print(f"annotation_order_sha256={annotation_digest}")
    if annotation_digest != LOCKED_ANNOTATION_ORDER_SHA256:
        print(f"locked_annotation_order_sha256={LOCKED_ANNOTATION_ORDER_SHA256}")
        raise SystemExit("The annotation-list permutation does not match the locked review order.")
    print("annotation_order_matches_locked_sample=true")


if __name__ == "__main__":
    main()
