#!/usr/bin/env python3
"""
5 preprocessing condition(raw / medsam3 seg·crop / chexmask seg·crop)이
동일한 sample·label·split을 갖는지 AP/PA 각각 검사합니다.

학습 시작 전에 한 번 실행해 두면 좋습니다.
"""

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
TRAINING_DIR = os.path.join(PACKAGE_ROOT, "training")
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)

from pneumonia_train import assert_preprocessing_sample_parity  # noqa: E402


def _default_labels_json() -> str | None:
    candidate = os.path.join(PACKAGE_ROOT, "pneumonia_labels.json")
    return candidate if os.path.isfile(candidate) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AP/PA preprocessing sample·split parity 검사",
    )
    parser.add_argument(
        "--data-base-dir",
        default=os.environ.get("DATA_BASE_DIR", PACKAGE_ROOT),
        help="cxr_*_{ap,pa} 상위 디렉터리",
    )
    parser.add_argument(
        "--split-dir",
        default=os.path.join(PACKAGE_ROOT, "splits"),
        help="고정 split CSV 디렉터리 (없으면 sample/label만 검사)",
    )
    parser.add_argument("--labels-json", default=_default_labels_json())
    parser.add_argument("--source-image-root", default=os.environ.get("MIMIC_CXR_ROOT"))
    parser.add_argument(
        "--views",
        nargs="+",
        default=["AP", "PA"],
        choices=["AP", "PA"],
    )
    parser.add_argument(
        "--no-split-check",
        action="store_true",
        help="fixed split CSV와의 outer/fold 일치 검사를 건너뜁니다.",
    )
    parser.add_argument(
        "--require-files",
        action="store_true",
        help="5 mode 모두 실제 파일 존재까지 검사 (raw는 --source-image-root 필요)",
    )
    args = parser.parse_args()

    data_base_dir = os.path.normpath(str(args.data_base_dir))
    split_dir = None if args.no_split_check else os.path.normpath(str(args.split_dir))
    if split_dir and not os.path.isdir(split_dir):
        raise FileNotFoundError(f"split dir not found: {split_dir}")

    print("Preprocessing parity verification")
    print(f"  data_base_dir     : {data_base_dir}")
    print(f"  split_dir         : {split_dir or '(sample/label only)'}")
    print("")

    for view in args.views:
        assert_preprocessing_sample_parity(
            view=view.upper(),
            data_base_dir=data_base_dir,
            split_dir=split_dir,
            labels_json_path=args.labels_json,
            source_image_root=args.source_image_root,
            strict_grouping=True,
            require_files=bool(args.require_files),
        )
        print("")

    print("All requested views passed preprocessing parity checks.")


if __name__ == "__main__":
    main()
