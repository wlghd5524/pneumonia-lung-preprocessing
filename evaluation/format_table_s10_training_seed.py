#!/usr/bin/env python3
"""Write Supplementary Table S10 (panels A–D) from the released seed summaries.

Sources are the CSV files written by summarize_training_seed_robustness.py.
Values are printed to four decimals as in the manuscript.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = ROOT / "results" / "sensitivity" / "seed_robustness"
DEST = SEED_DIR / "Table_S10_training_seed_robustness.md"

BACKBONES = ["ResNet-152", "EVA-X-Base"]
DELTA_ORDER = ["MedSAM3 crop", "CheXMask crop", "MedSAM3 mask", "CheXMask mask"]
INPUT_ORDER = ["Raw", "MedSAM3 mask", "MedSAM3 crop", "CheXMask mask", "CheXMask crop"]


def read(name: str) -> list[dict]:
    with (SEED_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(value: str) -> str:
    return f"{float(value):.4f}"


def main() -> None:
    by_backbone = {(r["view"], r["arch"], r["mode"]): r for r in read("seed_paired_delta_by_backbone.csv")}
    crop_mask = {(r["view"], r["arch"], r["contrast"]): r for r in read("seed_crop_minus_mask.csv")}
    absolute = {
        (r["view"], r["arch"], r["mode"]): r for r in read("seed_absolute_metrics.csv") if r["metric"] == "AUROC"
    }
    lines = [
        "# Table S10. Independent training-seed robustness of paired preprocessing effects",
        "",
        "Main-text Table 7 reports the two-backbone mean ΔAUROC. This table provides the backbone-specific "
        "paired differences and the absolute outer-test AUROC at each training seed. Patient partitions were "
        "fixed (split seed 42). Training seeds were 42, 123, and 2026. Each score is the five-model ensemble on "
        "the same outer-test radiographs. Δ is preprocessed minus raw, or crop minus the matching hard mask, "
        "within the same view, backbone, and training seed. Mean and SD are across the three seeds; SD divides "
        "by n−1. Positive values indicate a higher score for the first condition. Only ResNet-152 and "
        "EVA-X-Base were retrained.",
        "",
    ]
    for title, prefix, mean_col, sd_col, dir_col, metric in [
        ("A. Backbone-specific ΔAUROC versus raw", "d", "mean", "sd", "direction", "ΔAUROC"),
        ("C. Backbone-specific ΔAUPRC versus raw", "a", "amean", "asd", "adir", "ΔAUPRC"),
    ]:
        if prefix == "a":
            lines.extend(crop_mask_panel(crop_mask))
        lines.extend([
            f"## {title}",
            "",
            f"| View | Backbone | Preprocessing | Seed 42 | Seed 123 | Seed 2026 | Mean {metric} | SD | Direction |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for view in ("AP", "PA"):
            for arch in BACKBONES:
                for mode in DELTA_ORDER:
                    r = by_backbone[(view, arch, mode)]
                    lines.append(
                        f"| {view} | {arch} | {mode} | {num(r[prefix + '42'])} | {num(r[prefix + '123'])} | "
                        f"{num(r[prefix + '2026'])} | {num(r[mean_col])} | {num(r[sd_col])} | {r[dir_col]} |"
                    )
        lines.append("")
    lines.extend([
        "## D. Absolute outer-test AUROC by training seed",
        "",
        "| View | Backbone | Input | Seed 42 | Seed 123 | Seed 2026 | Mean AUROC | SD |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for view in ("AP", "PA"):
        for arch in BACKBONES:
            for mode in INPUT_ORDER:
                r = absolute[(view, arch, mode)]
                lines.append(
                    f"| {view} | {arch} | {mode} | {num(r['s42'])} | {num(r['s123'])} | {num(r['s2026'])} | "
                    f"{num(r['mean'])} | {num(r['sd'])} |"
                )
    lines.append("")
    DEST.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DEST}")


def crop_mask_panel(crop_mask: dict) -> list[str]:
    lines = [
        "## B. Crop minus hard mask, ΔAUROC",
        "",
        "| View | Backbone | Contrast | Seed 42 | Seed 123 | Seed 2026 | Mean ΔAUROC | SD | Direction |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for source in ("MedSAM3", "CheXMask"):
        contrast = f"{source} crop − mask"
        for view in ("AP", "PA"):
            for arch in BACKBONES + ["Two-backbone mean"]:
                r = crop_mask[(view, arch, contrast)]
                lines.append(
                    f"| {view} | {arch} | {source} crop - mask | {num(r['d42'])} | {num(r['d123'])} | "
                    f"{num(r['d2026'])} | {num(r['mean'])} | {num(r['sd'])} | {r['direction']} |"
                )
    lines.append("")
    return lines


if __name__ == "__main__":
    main()
