#!/usr/bin/env python3
"""Write Supplementary Table S9 from the released 2000-replicate control contrasts.

One row per backbone, view, and contrast (ResNet-152 and EVA-X-Base). As in
the manuscript, the ΔAUROC point estimate is the difference of the two AUROC
values rounded to four decimals; intervals and p values come from the
unrounded paired bootstrap.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "sensitivity" / "preprocessing_controls" / "paired_delta_by_backbone.csv"
DEST = ROOT / "results" / "sensitivity" / "preprocessing_controls" / "Table_S9_preprocessing_controls_by_backbone.md"

BACKBONES = [("resnet152", "ResNet-152"), ("eva_x_base", "EVA-X-Base")]
LABELS = {
    "medsam3_crop": "Study crop",
    "medsam3_center": "Centered crop",
    "medsam3_seg": "Hard mask",
    "medsam3_soft": "Soft mask",
    "raw": "Raw",
    "medsam3_margin0": "0% margin crop",
    "medsam3_margin10": "10% margin crop",
    "medsam3_margin20": "20% margin crop",
}
# (first, second): Δ = first − second
CONTRASTS = [
    ("medsam3_crop", "medsam3_center"),
    ("medsam3_crop", "medsam3_seg"),
    ("medsam3_crop", "medsam3_soft"),
    ("medsam3_seg", "medsam3_soft"),
    ("medsam3_seg", "raw"),
    ("medsam3_soft", "raw"),
    ("medsam3_crop", "raw"),
    ("medsam3_crop", "medsam3_margin0"),
    ("medsam3_crop", "medsam3_margin10"),
    ("medsam3_crop", "medsam3_margin20"),
    ("medsam3_center", "raw"),
    ("medsam3_center", "medsam3_seg"),
    ("medsam3_margin0", "raw"),
    ("medsam3_margin10", "raw"),
    ("medsam3_margin20", "raw"),
    ("medsam3_margin0", "medsam3_margin20"),
]


def signed(value: float) -> str:
    return "0.0000" if round(value, 4) == 0 else f"{value:+.4f}"


def p_value(value: str) -> str:
    p = float(value)
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # The CSV stores "b_minus_a" with delta = b − a.
    index = {(r["view"], r["model"], r["mode_b"], r["mode_a"]): r for r in rows}
    lines = [
        "# Table S9. Backbone-specific paired differences for the preprocessing controls",
        "",
        "Each row represents one outer-test contrast for ResNet-152 or EVA-X-Base. Δ denotes the first-named "
        "condition minus the second-named condition. Ninety-five percent confidence intervals and unadjusted "
        "two-sided p values were estimated using 2000 subject-level paired cluster-bootstrap replicates. "
        "All comparisons were fully paired on the same outer-test radiographs and labels. The two-backbone mean "
        "estimates for the principal contrasts are summarized in Table 6.",
        "",
        "| View | Backbone | Contrast | AUROC, first | AUROC, second | ΔAUROC (95% CI) | ΔAUROC p | ΔAUPRC (95% CI) | ΔAUPRC p |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for view in ("AP", "PA"):
        for model, label in BACKBONES:
            for first, second in CONTRASTS:
                r = index[(view, model, first, second)]
                if int(r["n_boot_effective"]) != 2000:
                    raise RuntimeError(f"{view} {model} {first} − {second} is not the 2000-replicate estimate")
                auc_first = round(float(r["auc_b"]), 4)
                auc_second = round(float(r["auc_a"]), 4)
                d_auc = (
                    f"{signed(auc_first - auc_second)} "
                    f"({float(r['delta_auc_lo']):+.4f}, {float(r['delta_auc_hi']):+.4f})"
                )
                d_ap = (
                    f"{signed(float(r['delta_auprc']))} "
                    f"({float(r['delta_auprc_lo']):+.4f}, {float(r['delta_auprc_hi']):+.4f})"
                )
                lines.append(
                    f"| {view} | {label} | {LABELS[first]} − {LABELS[second]} | {auc_first:.4f} | "
                    f"{auc_second:.4f} | {d_auc} | {p_value(r['delta_auc_p'])} | {d_ap} | "
                    f"{p_value(r['delta_auprc_p'])} |"
                )
    lines.extend([
        "",
        "AP: 3280 radiographs from 1763 patients; PA: 2521 radiographs from 1743 patients. All comparisons "
        "were fully paired on the same outer-test radiographs and labels. Δ denotes the first-named condition "
        "minus the second-named condition. AUROC values are reported to four decimal places, and the displayed "
        "ΔAUROC point estimates were calculated from these four-decimal values for numerical consistency; "
        "confidence intervals were estimated from the original unrounded predictions using 2000 subject-level "
        "paired cluster-bootstrap replicates. p values were not adjusted for multiple comparisons and are "
        "reported to three decimal places; values below 0.001 are reported as p < 0.001.",
        "",
    ])
    DEST.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DEST}")


if __name__ == "__main__":
    main()
