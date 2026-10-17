#!/usr/bin/env python3
"""Write manuscript Table 7 from the released two-backbone seed summary."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "sensitivity" / "seed_robustness" / "seed_paired_delta_two_backbone_mean.csv"
DEST = ROOT / "results" / "sensitivity" / "seed_robustness" / "Table_7_training_seed_robustness.md"

ORDER = [
    ("AP", "MedSAM3 crop"),
    ("AP", "CheXMask crop"),
    ("AP", "MedSAM3 mask"),
    ("AP", "CheXMask mask"),
    ("PA", "MedSAM3 crop"),
    ("PA", "CheXMask crop"),
    ("PA", "MedSAM3 mask"),
    ("PA", "CheXMask mask"),
]


def signed(value: str) -> str:
    return f"{float(value):+.4f}".replace("-", "−")


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = {(row["view"], row["mode"]): row for row in csv.DictReader(handle)}
    lines = [
        "# Table 7. Independent training-seed robustness of paired preprocessing effects",
        "",
        "Each entry represents the mean paired outer-test ΔAUROC across ResNet-152 and EVA-X-Base. "
        "Patient partitions were held fixed, and training seeds 42, 123, and 2026 were evaluated. "
        "Seed-specific values represent preprocessing minus raw within the same acquisition view, backbone, "
        "and training seed. Mean ΔAUROC and SD summarize the three training seeds, with SD calculated using "
        "the sample convention (n − 1). Positive values indicate higher AUROC than raw input. "
        "Detailed backbone-specific results are provided in Supplementary Table S10.",
        "",
        "| View | Preprocessing | Seed 42 | Seed 123 | Seed 2026 | Mean ΔAUROC | SD | Direction |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in ORDER:
        row = rows[key]
        lines.append(
            f"| {row['view']} | {row['mode']} | {signed(row['d42'])} | {signed(row['d123'])} | "
            f"{signed(row['d2026'])} | {signed(row['mean'])} | {float(row['sd']):.4f} | {row['direction']} |"
        )
    lines.append("")
    DEST.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DEST}")


if __name__ == "__main__":
    main()
