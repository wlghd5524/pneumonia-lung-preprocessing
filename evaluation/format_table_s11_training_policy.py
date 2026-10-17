#!/usr/bin/env python3
"""Write Supplementary Table S11 from the released training-policy summary.

Source is the output of evaluate_training_policy_sensitivity.py. Values are
printed to three decimals as in the manuscript.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "sensitivity" / "training_policy" / "training_policy_sensitivity.csv"
DEST = ROOT / "results" / "sensitivity" / "training_policy" / "Table_S11_training_policy_sensitivity.md"

BACKBONES = [("resnet152", "ResNet-152"), ("eva_x_base", "EVA-X-Base")]
INPUTS = [("raw", "Raw"), ("medsam3_crop", "MedSAM3 crop")]
POLICIES = ["Explicit-only", "U-Zero", "U-One"]


def interval(row: dict, key: str, sign: bool) -> str:
    fmt = "{:+.3f}" if sign else "{:.3f}"
    return (
        f"{fmt.format(float(row[key]))} "
        f"({fmt.format(float(row[key + '_lo']))}, {fmt.format(float(row[key + '_hi']))})"
    )


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = {
            (r["view"], r["model"], r["input"], r["training_policy"]): r for r in csv.DictReader(handle)
        }
    lines = [
        "# Table S11. Training-Policy Sensitivity to CheXpert Uncertain Pneumonia Labels in Representative "
        "Model Configurations",
        "",
        "ResNet-152 and EVA-X-Base models were retrained for raw and MedSAM3-cropped inputs in the AP and PA "
        "cohorts under three training-label policies: Explicit-only, U-Zero, and U-One. Patient-level data "
        "partitions were preserved across policies, and validation and outer-test evaluation used the same "
        "explicit 0/1 Pneumonia endpoint. AUROC and AUPRC are reported with 95% confidence intervals estimated "
        "using 2,000 subject-level cluster bootstrap replicates. ΔAUROC and ΔAUPRC represent paired differences "
        "relative to the corresponding Explicit-only model, estimated by resampling the same patients for both "
        "models.",
        "",
        "| View | Backbone | Input | Training Policy | AUROC (95% CI) | ΔAUROC vs. Explicit-only (95% CI) | "
        "AUPRC (95% CI) | ΔAUPRC vs. Explicit-only (95% CI) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for view in ("AP", "PA"):
        for model, model_label in BACKBONES:
            for mode, mode_label in INPUTS:
                for policy in POLICIES:
                    r = rows[(view, model, mode, policy)]
                    if int(r["n_boot_effective"]) != 2000:
                        raise RuntimeError(f"{view} {model} {mode} {policy} is not the 2,000-replicate estimate")
                    ref = policy == "Explicit-only"
                    lines.append(
                        f"| {view} | {model_label} | {mode_label} | {policy} | {interval(r, 'auroc', False)} | "
                        f"{'Ref.' if ref else interval(r, 'delta_auroc', True)} | {interval(r, 'auprc', False)} | "
                        f"{'Ref.' if ref else interval(r, 'delta_auprc', True)} |"
                    )
    lines.append("")
    DEST.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DEST}")


if __name__ == "__main__":
    main()
