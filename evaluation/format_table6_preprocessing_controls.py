#!/usr/bin/env python3
"""Write manuscript Table 6 from the released 2000-replicate control contrasts.

The pooled file is the mean of ResNet-152 and EVA-X-Base. Intervals are the
paired subject-level cluster bootstrap already stored in that file.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "sensitivity" / "preprocessing_controls" / "paired_delta_pooled.csv"
DEST = ROOT / "results" / "sensitivity" / "preprocessing_controls" / "Table_6_preprocessing_controls.md"

CONTRASTS = [
    ("medsam3_center", "medsam3_crop", "Study crop − Centered crop"),
    ("medsam3_seg", "medsam3_crop", "Study crop − Hard mask"),
    ("medsam3_soft", "medsam3_crop", "Study crop − Soft mask"),
    ("medsam3_soft", "medsam3_seg", "Hard mask − Soft mask"),
    ("raw", "medsam3_seg", "Hard mask − Raw"),
    ("raw", "medsam3_soft", "Soft mask − Raw"),
]


def cell(row: dict) -> str:
    point = float(row["mean_delta_auc"])
    lo = float(row["mean_delta_auc_lo"])
    hi = float(row["mean_delta_auc_hi"])
    return f"{signed(point)} ({signed(lo)}, {signed(hi)})"


def signed(value: float) -> str:
    return f"{value:+.4f}".replace("-", "−")


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    index = {(row["view"], row["mode_a"], row["mode_b"]): row for row in rows}
    lines = [
        "# Table 6. Controlled classification experiments for separating geometric and masking-related "
        "preprocessing effects",
        "",
        "Values represent mean paired AUROC differences across ResNet-152 and EVA-X-Base evaluated on the "
        "matched outer-test cohorts. Ninety-five percent confidence intervals were estimated using 2000 "
        "subject-level paired cluster-bootstrap replicates. Positive values indicate higher AUROC for the first "
        "condition named in each contrast. AP comprised 3280 radiographs from 1763 patients and PA comprised "
        "2521 radiographs from 1743 patients. Confidence intervals were not adjusted for multiple comparisons. "
        "Detailed backbone-specific AUROC and AUPRC results are provided in Supplementary Table S9.",
        "",
        "| Contrast | AP ΔAUROC (95% CI) | PA ΔAUROC (95% CI) |",
        "| --- | ---: | ---: |",
    ]
    for mode_a, mode_b, label in CONTRASTS:
        ap = index[("AP", mode_a, mode_b)]
        pa = index[("PA", mode_a, mode_b)]
        if int(ap["n_boot_effective"]) != 2000 or int(pa["n_boot_effective"]) != 2000:
            raise RuntimeError(f"{label} is not the 2000-replicate manuscript estimate")
        lines.append(f"| {label} | {cell(ap)} | {cell(pa)} |")
    lines.extend([
        "",
        "Centered crop refers to the centered crop with segmentation-matched dimensions. Its width and height "
        "were obtained from the corresponding MedSAM3-derived study-crop window, whereas the crop window was "
        "positioned at the image center.",
        "",
    ])
    DEST.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DEST}")


if __name__ == "__main__":
    main()
