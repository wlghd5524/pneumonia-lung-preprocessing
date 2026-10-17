#!/usr/bin/env python3
"""
AP cohort에서 CheXMask 폐 RLE가 없는 영상을 모든 preprocessing
manifest·split CSV에서 제외합니다. 제외 ID는 ``splits/AP_excluded_dicom_ids.json``
(``{"excluded_dicom_ids": [...]}``)에서 읽습니다. 이 파일은 배포하지 않으므로,
CheXMask 표에서 Left/Right Lung RLE가 비어 있는 AP cohort 영상의 dicom_id로
직접 만듭니다.

주의: AP 분할 CSV는 제외 전 cohort(22,899장)로 먼저 만든 뒤 이 스크립트로
5장을 제거해야 공개 SHA256과 일치합니다. 제외 후 cohort로 분할하면 다른 분할이 됩니다.

실행:
  python3 apply_ap_cohort_exclusion.py
  python3 apply_ap_cohort_exclusion.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
DATA_BASE_DIR = Path(os.environ.get("DATA_BASE_DIR", PACKAGE_ROOT)).resolve()
SPLIT_DIR = Path(os.environ.get("SPLIT_DIR", PACKAGE_ROOT / "splits")).resolve()

AP_MANIFESTS: Sequence[Path] = (
    DATA_BASE_DIR / "cxr_medsam3_lung_seg_ap" / "manifest_ap_medsam3_ok.json",
    DATA_BASE_DIR / "cxr_medsam3_lung_seg_cropped_ap" / "manifest_ap_medsam3_ok.json",
    DATA_BASE_DIR / "cxr_chexmask_lung_seg_ap" / "manifest_ap_chexmask_ok.json",
    DATA_BASE_DIR / "cxr_chexmask_lung_seg_cropped_ap" / "manifest_ap_chexmask_ok.json",
)

SPLIT_CSV_COLUMNS = ("subject_id", "dicom_id", "label", "view", "split", "fold")


def _load_excluded_dicom_ids() -> Set[str]:
    path = SPLIT_DIR / "AP_excluded_dicom_ids.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"exclusion file not found: {path}. Create it as "
            '{"excluded_dicom_ids": [...]} with the AP cohort dicom_ids whose '
            "CheXMask Left/Right Lung RLE is missing (see splits/README.md)."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get("excluded_dicom_ids", [])
    exclude = {str(value).strip() for value in values if str(value).strip()}
    if not exclude:
        raise RuntimeError(f"excluded_dicom_ids is empty: {path}")
    return exclude


def _backup(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak_{stamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def _filter_manifest_entries(entries: List[Dict], exclude: Set[str]) -> tuple[List[Dict], int]:
    kept: List[Dict] = []
    removed = 0
    for entry in entries:
        dicom_id = str(entry.get("dicom_id") or "")
        if dicom_id in exclude:
            removed += 1
            continue
        kept.append(entry)
    return kept, removed


def _filter_split_csv(path: Path, exclude: Set[str], dry_run: bool) -> tuple[int, int, Path | None]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    kept = [row for row in rows if str(row.get("dicom_id") or "") not in exclude]
    removed = len(rows) - len(kept)
    backup_path = None
    if not dry_run and removed > 0:
        backup_path = _backup(path)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SPLIT_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(kept)
    return len(kept), removed, backup_path


def _write_exclusion_meta(exclude: Set[str], dry_run: bool) -> None:
    meta_path = SPLIT_DIR / "AP_excluded_dicom_ids.json"
    payload = {
        "view": "AP",
        "reason": "ChexMask CSV Left/Right Lung RLE missing; excluded for cross-preprocessing parity",
        "excluded_dicom_ids": sorted(exclude),
        "n_excluded": len(exclude),
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_cohort_size": 22894,
    }
    if dry_run:
        print(f"[dry-run] would write {meta_path}")
        return
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"  wrote {meta_path}")


def _patch_split_meta(exclude: Set[str], dry_run: bool) -> None:
    meta_path = SPLIT_DIR / "AP_split_meta.json"
    if not meta_path.is_file():
        print(f"  skip meta patch (missing): {meta_path}")
        return
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    meta["cohort_exclusion"] = {
        "n_excluded": len(exclude),
        "excluded_dicom_ids": sorted(exclude),
        "reference_n_images_after_exclusion": 22894,
        "reference_n_subjects_after_exclusion": 11951,
        "note": "5 AP images removed for preprocessing parity (no ChexMask lung RLE)",
    }
    if dry_run:
        print(f"[dry-run] would patch {meta_path}")
        return
    backup = _backup(meta_path)
    print(f"  backup: {backup}")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    print(f"  patched {meta_path}")


def apply_exclusion(*, dry_run: bool = False) -> None:
    exclude = _load_excluded_dicom_ids()
    print("AP cohort exclusion")
    print(f"  exclude n={len(exclude)}")
    print(f"  dry_run={dry_run}")
    print("")

    for manifest_path in AP_MANIFESTS:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
        if not isinstance(entries, list):
            raise RuntimeError(f"manifest must be a list: {manifest_path}")
        kept, removed = _filter_manifest_entries(entries, exclude)
        print(f"{manifest_path.name}: {len(entries)} -> {len(kept)} (removed {removed})")
        if not dry_run and removed > 0:
            backup = _backup(manifest_path)
            print(f"  backup: {backup}")
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(kept, handle, ensure_ascii=False, indent=2)
        elif not dry_run and removed == 0:
            print("  unchanged")

    print("")
    for csv_name in ("AP_outer_test.csv", "AP_fold_assignment.csv"):
        csv_path = SPLIT_DIR / csv_name
        if not csv_path.is_file():
            raise FileNotFoundError(f"split csv not found: {csv_path}")
        kept, removed, backup_path = _filter_split_csv(csv_path, exclude, dry_run=dry_run)
        print(f"{csv_name}: kept={kept}, removed={removed}")
        if backup_path is not None:
            print(f"  backup: {backup_path}")

    print("")
    _write_exclusion_meta(exclude, dry_run=dry_run)
    _patch_split_meta(exclude, dry_run=dry_run)
    print("")
    print("Done." if not dry_run else "Dry run complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exclude 5 AP dicom_ids from all manifests and split CSVs")
    parser.add_argument("--dry-run", action="store_true", help="변경 없이 제거 대상만 출력")
    args = parser.parse_args()
    apply_exclusion(dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
