#!/usr/bin/env python3
"""Write manuscript Table 5 from the released 70-run outer-test metrics.

Each cell averages the seven backbones of one view and preprocessing. The
rounding follows the manuscript:

  - Mean AUROC and mean AUPRC are means of the per-backbone values rounded to
    four decimals.
  - ΔAUROC is the mean of the seven per-backbone ΔAUROC values printed in
    Table 4 (four decimals).
  - ΔAUPRC is the displayed mean AUPRC minus the displayed raw mean AUPRC.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "primary" / "aggregate_metrics.csv"
DEST = ROOT / "results" / "primary" / "Table_5_mean_by_preprocessing.md"

ORDER = [
    ("raw", "Raw"),
    ("chexmask_seg", "CheXMask mask"),
    ("chexmask_crop", "CheXMask crop"),
    ("medsam3_seg", "MedSAM3 mask"),
    ("medsam3_crop", "MedSAM3 crop"),
]


def signed(value: float) -> str:
    text = f"{value:+.4f}"
    return text.replace("-", "−")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        "# Table 5. Mean held-out test performance across the seven evaluated backbone architectures "
        "according to acquisition view and lung preprocessing method",
        "",
        "AUROC was the primary discrimination metric, and AUPRC was evaluated as a secondary metric. "
        "Differences relative to raw input are shown descriptively using the corresponding backbone-averaged "
        "values within each acquisition view.",
        "",
        "| View | Preprocessing | Mean AUROC | ΔAUROC vs. Raw | Mean AUPRC | ΔAUPRC vs. Raw |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for view in ("AP", "PA"):
        by_mode = {}
        for mode, _ in ORDER:
            picked = {r["model"]: r for r in rows if r["view"] == view and r["mode"] == mode}
            if len(picked) != 7:
                raise RuntimeError(f"{view} {mode}: expected 7 backbones, found {len(picked)}")
            by_mode[mode] = picked
        raw = by_mode["raw"]
        raw_auprc = round(mean([round(float(r["test_auprc"]), 4) for r in raw.values()]), 4)
        for mode, label in ORDER:
            picked = by_mode[mode]
            auroc = round(mean([round(float(r["test_auc"]), 4) for r in picked.values()]), 4)
            auprc = round(mean([round(float(r["test_auprc"]), 4) for r in picked.values()]), 4)
            if mode == "raw":
                d_auroc = d_auprc = "Reference"
            else:
                deltas = [
                    round(float(picked[m]["test_auc"]) - float(raw[m]["test_auc"]), 4) for m in picked
                ]
                d_auroc = signed(round(mean(deltas), 4))
                d_auprc = signed(round(auprc - raw_auprc, 4))
            lines.append(f"| {view} | {label} | {auroc:.4f} | {d_auroc} | {auprc:.4f} | {d_auprc} |")
    lines.extend([
        "",
        "Mean AUROC and mean AUPRC average the seven per-backbone values rounded to four decimals. "
        "ΔAUROC is the mean of the seven per-backbone ΔAUROC values in Table 4. "
        "ΔAUPRC is the displayed mean AUPRC minus the displayed raw mean AUPRC.",
        "",
    ])
    DEST.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DEST}")


if __name__ == "__main__":
    main()
