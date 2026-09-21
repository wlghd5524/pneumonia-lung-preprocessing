#!/usr/bin/env python3
"""
고정 patient split CSV를 한 번 생성합니다.

training seed를 여러 번 바꿔도 outer-test / fold 환자 구성이 동일하도록,
split_seed(기본 42)만 사용해 AP/PA split CSV를 미리 만듭니다.

출력 예:
  splits/AP_outer_test.csv
  splits/AP_fold_assignment.csv
  splits/AP_split_meta.json
  splits/PA_outer_test.csv
  splits/PA_fold_assignment.csv
  splits/PA_split_meta.json

각 CSV 컬럼:
  subject_id, dicom_id, label, view, split, fold
"""

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from pneumonia_train import (  # noqa: E402
    build_grouped_split_assignment,
    load_manifest_split_records,
    save_fixed_split_csvs,
)


def _default_labels_json() -> str | None:
    candidate = os.path.join(PACKAGE_ROOT, "pneumonia_labels.json")
    return candidate if os.path.isfile(candidate) else None


def _data_root_for(data_base_dir: str, view: str) -> str:
    return os.path.join(data_base_dir, f"cxr_medsam3_lung_seg_{view.lower()}")


def export_view_split(
    *,
    view: str,
    data_base_dir: str,
    split_dir: str,
    split_seed: int,
    n_folds: int,
    outer_test_ratio: float,
    labels_json: str | None,
    data_mode: str,
) -> None:
    view_tag = view.lower()
    data_root = _data_root_for(data_base_dir, view)
    records, manifest_path = load_manifest_split_records(
        data_root,
        labels_json_path=labels_json,
        data_mode=data_mode,
        view_tag=view_tag,
        strict_grouping=True,
    )
    dummy_items = [(r["dicom_id"], r["label"], r["dicom_id"]) for r in records]
    labels = [r["label"] for r in records]
    groups = [r["group_id"] for r in records]
    use_outer_test = float(outer_test_ratio) > 0.0
    assignment = build_grouped_split_assignment(
        dummy_items,
        labels,
        groups,
        n_folds=int(n_folds),
        outer_test_ratio=float(outer_test_ratio) if use_outer_test else 0.0,
        seed=int(split_seed),
    )
    meta = {
        "view": view.upper(),
        "split_seed": int(split_seed),
        "n_folds": int(n_folds),
        "outer_test_ratio": float(outer_test_ratio) if use_outer_test else 0.0,
        "reference_data_root": os.path.normpath(data_root),
        "reference_data_mode": data_mode,
        "reference_manifest": manifest_path,
        "outer_split_method": assignment.get("outer_method"),
        "reference_n_images": assignment.get("reference_n_images"),
        "reference_n_subjects": assignment.get("reference_n_subjects"),
    }
    outer_csv, fold_csv = save_fixed_split_csvs(
        split_dir=split_dir,
        view=view,
        records=records,
        assignment=assignment,
        meta=meta,
    )
    print(f"[{view}] split_seed={split_seed}")
    print(f"  manifest        : {manifest_path}")
    print(f"  images/subjects : {meta['reference_n_images']}/{meta['reference_n_subjects']}")
    print(f"  outer_test CSV  : {outer_csv}")
    print(f"  fold assign CSV : {fold_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="고정 patient split CSV 생성 (AP/PA)")
    parser.add_argument(
        "--split-dir",
        default=os.path.join(PACKAGE_ROOT, "splits"),
        help="split CSV 저장 디렉터리 (기본: 패키지 루트/splits)",
    )
    parser.add_argument("--split-seed", type=int, default=42, help="patient split seed (고정값 권장)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--outer-test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--data-base-dir",
        default=os.environ.get("DATA_BASE_DIR", PACKAGE_ROOT),
        help="cxr_medsam3_lung_seg_{ap,pa} 상위 디렉터리",
    )
    parser.add_argument("--labels-json", default=_default_labels_json())
    parser.add_argument(
        "--data-mode",
        default="raw",
        choices=["raw", "medsam3_seg", "medsam3_crop", "chexmask_seg", "chexmask_crop"],
        help="split 계산에 사용할 manifest 필드 (기본 raw)",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=["AP", "PA"],
        choices=["AP", "PA"],
        help="생성할 view 목록",
    )
    args = parser.parse_args()

    split_dir = os.path.normpath(str(args.split_dir))
    os.makedirs(split_dir, exist_ok=True)
    data_base_dir = os.path.normpath(str(args.data_base_dir))

    print("고정 patient split CSV 생성")
    print(f"  split_dir         : {split_dir}")
    print(f"  split_seed        : {args.split_seed}")
    print(f"  n_folds           : {args.n_folds}")
    print(f"  outer_test_ratio  : {args.outer_test_ratio}")
    print(f"  data_base_dir     : {data_base_dir}")
    print(f"  data_mode         : {args.data_mode}")
    print("")

    for view in args.views:
        export_view_split(
            view=view.upper(),
            data_base_dir=data_base_dir,
            split_dir=split_dir,
            split_seed=int(args.split_seed),
            n_folds=int(args.n_folds),
            outer_test_ratio=float(args.outer_test_ratio),
            labels_json=args.labels_json,
            data_mode=str(args.data_mode),
        )
        print("")

    print("완료: 이후 학습 시 --split-dir 과 --seed(training)를 분리해 사용하세요.")
    print("  예) --split-dir splits --split-seed 42 --seed 2026")


if __name__ == "__main__":
    main()
