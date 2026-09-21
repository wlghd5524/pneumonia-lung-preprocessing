#!/usr/bin/env python3
"""
MIMIC-CXR-JPG + CheXpert 라벨에서 Pneumonia JSON 생성.

기본 동작:
- CheXpert의 Pneumonia 컬럼 사용
- 라벨은 0(음성) / 1(양성)만 유지
- 촬영 방향은 AP / PA만 유지
- 실제 이미지 파일이 존재하는 레코드만 유지
- metadata와 조인해 이미지(dicom) 단위 레코드 생성
- 각 필터 단계에서 남은 개수와 드랍 개수를 터미널에 출력
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIMIC_ROOT = Path(os.environ.get("MIMIC_CXR_ROOT", PACKAGE_ROOT / "data" / "mimic-cxr-jpg" / "2.1.0"))
DEFAULT_CHEXPERT = DEFAULT_MIMIC_ROOT / "mimic-cxr-2.0.0-chexpert.csv.gz"
DEFAULT_METADATA = DEFAULT_MIMIC_ROOT / "mimic-cxr-2.0.0-metadata.csv.gz"
DEFAULT_SPLIT = DEFAULT_MIMIC_ROOT / "mimic-cxr-2.0.0-split.csv.gz"
DEFAULT_OUT_JSON = PACKAGE_ROOT / "pneumonia_labels.json"


def build_image_path(subject_id: int, study_id: int, dicom_id: str) -> str:
    sid = str(int(subject_id))
    std = str(int(study_id))
    prefix = sid[:2] if len(sid) >= 2 else sid
    return f"files/p{prefix}/p{sid}/s{std}/{dicom_id}.jpg"


def _to_none_if_nan(x):
    return None if pd.isna(x) else x


def _study_keys(df: pd.DataFrame) -> set[tuple[int, int]]:
    return set(zip(df["subject_id"].astype(int), df["study_id"].astype(int)))


def _print_step(name: str, before: int, after: int) -> None:
    dropped = before - after
    drop_rate = (dropped / before * 100.0) if before else 0.0
    print(f"[FILTER] {name}: {before:,} -> {after:,} (drop={dropped:,}, {drop_rate:.2f}%)")


def _print_label_counts(prefix: str, values: pd.Series) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    n_pos = int((numeric == 1).sum())
    n_neg = int((numeric == 0).sum())
    n_uncertain = int((numeric == -1).sum())
    n_null = int(values.isna().sum())
    n_other = int((~values.isna() & ~numeric.isin([-1, 0, 1])).sum())
    print(
        f"[STAT] {prefix}: pneumonia=1 {n_pos:,}, pneumonia=0 {n_neg:,}, "
        f"uncertain(-1) {n_uncertain:,}, null {n_null:,}, other {n_other:,}"
    )


def _ap_pa_study_image_counts(df: pd.DataFrame) -> dict[str, int]:
    empty = {
        "img_ap_pos": 0, "img_ap_neg": 0, "img_pa_pos": 0, "img_pa_neg": 0, "img_total": 0,
        "study_ap_pos": 0, "study_ap_neg": 0, "study_pa_pos": 0, "study_pa_neg": 0,
        "study_total": 0,
    }
    if df.empty:
        return empty

    view = df["ViewPosition"].fillna("").astype(str).str.upper()
    label = pd.to_numeric(df["Pneumonia"], errors="coerce").astype("Int64")
    ap = view == "AP"
    pa = view == "PA"
    pos = label == 1
    neg = label == 0

    study_key = df[["subject_id", "study_id"]].astype(int).apply(tuple, axis=1)

    def _unique_studies(mask: pd.Series) -> int:
        if not bool(mask.any()):
            return 0
        return int(study_key.loc[mask].nunique())

    return {
        "img_ap_pos": int((ap & pos).sum()),
        "img_ap_neg": int((ap & neg).sum()),
        "img_pa_pos": int((pa & pos).sum()),
        "img_pa_neg": int((pa & neg).sum()),
        "img_total": int(len(df)),
        "study_ap_pos": _unique_studies(ap & pos),
        "study_ap_neg": _unique_studies(ap & neg),
        "study_pa_pos": _unique_studies(pa & pos),
        "study_pa_neg": _unique_studies(pa & neg),
        "study_total": int(study_key.nunique()),
    }


def _print_ap_pa_study_image_breakdown(prefix: str, df: pd.DataFrame) -> None:
    c = _ap_pa_study_image_counts(df)
    print(f"[STAT] {prefix}")
    print(
        "  images : "
        f"AP positive={c['img_ap_pos']:,}, AP negative={c['img_ap_neg']:,}, "
        f"PA positive={c['img_pa_pos']:,}, PA negative={c['img_pa_neg']:,}, "
        f"total={c['img_total']:,}"
    )
    print(
        "  studies: "
        f"AP positive={c['study_ap_pos']:,}, AP negative={c['study_ap_neg']:,}, "
        f"PA positive={c['study_pa_pos']:,}, PA negative={c['study_pa_neg']:,}, "
        f"unique total={c['study_total']:,}"
    )


def _print_ap_pa_study_image_breakdown_from_records(prefix: str, records: list[dict]) -> None:
    if not records:
        _print_ap_pa_study_image_breakdown(prefix, pd.DataFrame())
        return
    df = pd.DataFrame(records).rename(
        columns={"view_position": "ViewPosition", "pneumonia": "Pneumonia"}
    )
    _print_ap_pa_study_image_breakdown(prefix, df)


def _print_study_label_counts(prefix: str, df: pd.DataFrame) -> None:
    if df.empty:
        print(f"[STAT] {prefix}: studies positive=0, studies negative=0, total=0")
        return
    label = pd.to_numeric(df["Pneumonia"], errors="coerce").astype("Int64")
    n_pos = int((label == 1).sum())
    n_neg = int((label == 0).sum())
    print(f"[STAT] {prefix}: studies positive={n_pos:,}, studies negative={n_neg:,}, total={len(df):,}")


def load_split_df(split_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    if split_csv is None:
        return None
    if not split_csv.is_file():
        return None
    sdf = pd.read_csv(split_csv, usecols=["subject_id", "study_id", "dicom_id", "split"])
    return sdf


def main() -> None:
    parser = argparse.ArgumentParser(description="CheXpert Pneumonia AP/PA 0/1 라벨 JSON 생성")
    parser.add_argument("--mimic-root", type=str, default=str(DEFAULT_MIMIC_ROOT))
    parser.add_argument("--chexpert-csv", type=str, default=str(DEFAULT_CHEXPERT))
    parser.add_argument("--metadata-csv", type=str, default=str(DEFAULT_METADATA))
    parser.add_argument("--split-csv", type=str, default=str(DEFAULT_SPLIT))
    parser.add_argument("--no-split", action="store_true", help="split.csv 조인을 생략합니다.")
    parser.add_argument(
        "--drop-na",
        action="store_true",
        help="하위 호환용 옵션입니다. 현재는 기본적으로 0/1 라벨만 유지하므로 NaN은 항상 제외됩니다.",
    )
    parser.add_argument("--out-json", type=str, default=str(DEFAULT_OUT_JSON))
    parser.add_argument(
        "--out-study-json",
        type=str,
        default="",
        help="원하면 study 단위(JSON)도 저장합니다. 빈 문자열이면 저장 안 함.",
    )
    args = parser.parse_args()

    mimic_root = Path(args.mimic_root).resolve()
    chexpert_csv = Path(args.chexpert_csv).resolve()
    metadata_csv = Path(args.metadata_csv).resolve()
    split_csv = None if args.no_split else Path(args.split_csv).resolve()
    out_json = Path(args.out_json).resolve()
    out_study_json = Path(args.out_study_json).resolve() if str(args.out_study_json).strip() else None

    if not chexpert_csv.is_file():
        raise FileNotFoundError(f"chexpert csv 파일이 없습니다: {chexpert_csv}")
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"metadata csv 파일이 없습니다: {metadata_csv}")

    print(f"[INFO] reading chexpert: {chexpert_csv}")
    chex = pd.read_csv(chexpert_csv, usecols=["subject_id", "study_id", "Pneumonia"])
    total_chex = len(chex)

    _print_label_counts("chexpert raw study rows", chex["Pneumonia"])
    pneumonia_num = pd.to_numeric(chex["Pneumonia"], errors="coerce")
    label_mask = pneumonia_num.isin([0, 1])
    chex = chex.loc[label_mask].copy()
    chex["Pneumonia"] = pd.to_numeric(chex["Pneumonia"], errors="coerce").astype(int)
    _print_step("CheXpert label filter (keep Pneumonia 0/1 only)", total_chex, len(chex))
    _print_label_counts("chexpert after label filter", chex["Pneumonia"])
    _print_study_label_counts("chexpert after label filter", chex)

    print(f"[INFO] reading metadata: {metadata_csv}")
    meta = pd.read_csv(
        metadata_csv,
        usecols=["subject_id", "study_id", "dicom_id", "ViewPosition", "StudyDate", "StudyTime"],
    )
    print(f"[INFO] metadata rows: {len(meta):,}")
    view_counts = meta["ViewPosition"].fillna("NULL").astype(str).str.upper().value_counts()
    print(
        "[STAT] metadata view counts: "
        + ", ".join(f"{view}={count:,}" for view, count in view_counts.head(10).items())
    )

    before_view = len(meta)
    view_norm = meta["ViewPosition"].fillna("").astype(str).str.upper()
    meta = meta.loc[view_norm.isin(["AP", "PA"])].copy()
    _print_step("metadata view filter (keep AP/PA only)", before_view, len(meta))
    view_counts_after = meta["ViewPosition"].fillna("NULL").astype(str).str.upper().value_counts()
    print(
        "[STAT] metadata AP/PA after view filter: "
        + ", ".join(f"{view}={count:,}" for view, count in view_counts_after.items())
    )

    print("[INFO] merging chexpert + metadata (study -> dicom expansion)")
    chex_studies_before_merge = _study_keys(chex)
    merged = meta.merge(chex, on=["subject_id", "study_id"], how="inner")
    merged_studies = _study_keys(merged) if not merged.empty else set()
    lost_studies = len(chex_studies_before_merge - merged_studies)
    print(f"[INFO] merged image rows: {len(merged):,}")
    print(
        f"[FILTER] study merge loss after AP/PA metadata join: "
        f"{len(chex_studies_before_merge):,} studies -> {len(merged_studies):,} "
        f"(drop={lost_studies:,})"
    )
    if not merged.empty:
        _print_ap_pa_study_image_breakdown("after merge (before image existence check)", merged)

    split_df = load_split_df(split_csv)
    if split_df is not None:
        merged = merged.merge(split_df, on=["subject_id", "study_id", "dicom_id"], how="left")
        split_missing = int(merged["split"].isna().sum())
        split_rate = (split_missing / len(merged) * 100.0) if len(merged) else 0.0
        print(
            f"[INFO] split column merged. missing split={split_missing:,}/{len(merged):,} "
            f"({split_rate:.2f}%)"
        )
    else:
        merged["split"] = None
        print("[INFO] split.csv not used.")

    merged["image_rel_path"] = merged.apply(
        lambda r: build_image_path(r["subject_id"], r["study_id"], r["dicom_id"]),
        axis=1,
    )
    merged["image_abs_path"] = merged["image_rel_path"].apply(lambda p: str((mimic_root / p).resolve()))

    before_exists = len(merged)
    image_exists = [Path(p).is_file() for p in merged["image_abs_path"]]
    merged = merged.loc[image_exists].copy()
    _print_step("image file existence filter", before_exists, len(merged))
    if before_exists != len(merged):
        missing = before_exists - len(merged)
        print(f"[WARN] dropped {missing:,} rows because image_abs_path did not exist.")
    if not merged.empty:
        _print_ap_pa_study_image_breakdown("after image existence filter", merged)

    # JSON 직렬화-friendly 타입으로 변환
    def row_to_record(row):
        label = int(row["Pneumonia"])
        return {
            "subject_id": int(row["subject_id"]),
            "study_id": int(row["study_id"]),
            "dicom_id": str(row["dicom_id"]),
            "view_position": _to_none_if_nan(row.get("ViewPosition")),
            "study_date": _to_none_if_nan(row.get("StudyDate")),
            "study_time": _to_none_if_nan(row.get("StudyTime")),
            "split": _to_none_if_nan(row.get("split")),
            "pneumonia": label,
            "image_rel_path": str(row["image_rel_path"]),
            "image_abs_path": str(row["image_abs_path"]),
        }

    records = [row_to_record(r) for _, r in merged.iterrows()]

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[DONE] image-level JSON saved: {out_json} (rows={len(records):,})")

    _print_ap_pa_study_image_breakdown_from_records("final saved JSON", records)

    if out_study_json is not None:
        # 최종 image-level 필터를 통과한 study만 저장합니다.
        study_df = merged[["subject_id", "study_id", "Pneumonia"]].drop_duplicates().copy()
        study_df["pneumonia"] = study_df["Pneumonia"].astype(int)
        study_records = [
            {
                "subject_id": int(r["subject_id"]),
                "study_id": int(r["study_id"]),
                "pneumonia": int(r["pneumonia"]),
            }
            for _, r in study_df[["subject_id", "study_id", "pneumonia"]].iterrows()
        ]
        out_study_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_study_json, "w", encoding="utf-8") as f:
            json.dump(study_records, f, ensure_ascii=False, indent=2)
        print(f"[DONE] study-level JSON saved: {out_study_json} (rows={len(study_records):,})")


if __name__ == "__main__":
    main()
