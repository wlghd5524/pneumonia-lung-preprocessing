#!/usr/bin/env python3
"""Write manuscript Table 6 from the released 2,000-replicate control contrasts.

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
    ("medsam3_center", "medsam3_crop", "Study crop − Matched center crop", "Little effect of crop location"),
    ("medsam3_seg", "medsam3_crop", "Study crop − Hard mask", "Crop higher than hard masking"),
    ("medsam3_soft", "medsam3_crop", "Study crop − Soft mask", "Boundary smoothing does not close the gap"),
    ("medsam3_soft", "medsam3_seg", "Hard mask − Soft mask", "Limited effect of the sharp boundary alone"),
    ("raw", "medsam3_seg", "Hard mask − Raw", "Masking lower than raw, particularly on PA"),
    ("raw", "medsam3_soft", "Soft mask − Raw", "Decrease remains after boundary softening"),
]


def cell(row: dict) -> str:
    point = float(row["mean_delta_auc"])
    lo = float(row["mean_delta_auc_lo"])
    hi = float(row["mean_delta_auc_hi"])
    return f"{point:+.4f} ({lo:+.4f}, {hi:+.4f})"


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    index = {(row["view"], row["mode_a"], row["mode_b"]): row for row in rows}
    lines = [
        "# Table 6. Controlled classification experiments for separating geometric and masking-related preprocessing effects. "
        "Values represent mean paired differences across ResNet-152 and EVA-X-Base on the matched outer-test cohorts, "
        "with 95% confidence intervals estimated using 2,000 subject-level paired cluster-bootstrap replicates. "
        "Positive values indicate higher performance for the first condition in each contrast. "
        "Detailed backbone-specific AUROC and AUPRC results are provided in Supplementary Table S9.",
        "",
        "| Contrast | AP ΔAUROC (95% CI) | PA ΔAUROC (95% CI) | Interpretation |",
        "| --- | ---: | ---: | --- |",
    ]
    for mode_a, mode_b, label, interpretation in CONTRASTS:
        ap = index[("AP", mode_a, mode_b)]
        pa = index[("PA", mode_a, mode_b)]
        if int(ap["n_boot_effective"]) != 2000 or int(pa["n_boot_effective"]) != 2000:
            raise RuntimeError(f"{label} is not the 2,000-replicate manuscript estimate")
        lines.append(f"| {label} | {cell(ap)} | {cell(pa)} | {interpretation} |")
    lines.extend([
        "",
        "AP uses 3,280 images from 1,763 patients; PA uses 2,521 images from 1,743 patients. "
        "The matched center crop uses the width and height of the corresponding MedSAM3 study-crop window "
        "after its standard padding and before black-border trimming and square padding. "
        "Intervals are not adjusted for multiple comparisons.",
        "",
    ])
    DEST.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DEST}")


if __name__ == "__main__":
    main()
