#!/usr/bin/env python3
"""Fold-level AUROC/AUPRC mean±SD and paired Δ vs raw (5 CV folds)."""
from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parents[1] / "results_pneumonia")).resolve()
CANON = ROOT / "aggregates" / "_canonical_70runs.txt"
OUT_MD = ROOT / "paper_tables" / "Supplementary_Table_fold_auroc_auprc_vs_raw.md"
OUT_CSV = ROOT / "paper_tables" / "Supplementary_Table_fold_auroc_auprc_vs_raw.csv"

RUN_RE = re.compile(
    r"_(AP|PA)_(raw|medsam3_seg|medsam3_crop|chexmask_seg|chexmask_crop)_cv_5fold"
)

ARCH_LABEL = {
    "resnet152": "ResNet-152",
    "densenet": "DenseNet-201",
    "convnextv2_base": "ConvNeXt V2-Base",
    "swint_base": "Swin V2-Base",
    "dinov3_base": "DINOv3-Base",
    "eva_x_base": "EVA-X-Base",
    "rad_dino": "RAD-DINO",
}
MODE_LABEL = {
    "raw": "Raw",
    "medsam3_seg": "MedSAM3 mask",
    "medsam3_crop": "MedSAM3 crop",
    "chexmask_seg": "CheXMask mask",
    "chexmask_crop": "CheXMask crop",
}
ARCH_ORDER = list(ARCH_LABEL)
MODE_ORDER = list(MODE_LABEL)
VIEW_ORDER = ["AP", "PA"]


def mean_sd(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    mu = sum(xs) / n
    # Match summary.json std_fold_auc / cv_auc_std (population SD, divide by n).
    var = sum((x - mu) ** 2 for x in xs) / n
    return mu, math.sqrt(var)


def fmt_mean_sd(mu: float, sd: float, signed: bool = False) -> str:
    if signed:
        return f"{mu:+.4f} ± {sd:.4f}"
    return f"{mu:.4f} ± {sd:.4f}"


def same_direction(deltas: list[float]) -> str:
    mu = sum(deltas) / len(deltas)
    if all(abs(d) < 1e-15 for d in deltas):
        return "—"
    if abs(mu) < 1e-15:
        n_zero = sum(1 for d in deltas if abs(d) < 1e-15)
        return f"{n_zero}/{len(deltas)}"
    n = sum(1 for d in deltas if (d > 0) == (mu > 0) and abs(d) >= 1e-15)
    return f"{n}/{len(deltas)}"


def load_run(run: str) -> dict:
    m = RUN_RE.search(run)
    if not m:
        raise ValueError(f"Cannot parse view/mode from {run}")
    view, mode = m.group(1), m.group(2)
    summary_path = ROOT / run / "summary.json"
    with summary_path.open() as f:
        s = json.load(f)
    cfg = s.get("run_config") or s.get("config") or {}
    meta_run = (s.get("run_meta") or {}).get("run") or {}
    arch = meta_run.get("arch") or cfg.get("arch")
    if not arch:
        raise ValueError(f"Missing arch in {summary_path}")
    auc = [float(x) for x in s["per_fold_auc"]]
    auprc = [float(x) for x in s["per_fold_auprc"]]
    if len(auc) != 5 or len(auprc) != 5:
        raise ValueError(f"{run}: expected 5 folds, got {len(auc)}/{len(auprc)}")
    return {
        "run": run,
        "view": view,
        "arch": arch,
        "mode": mode,
        "auc": auc,
        "auprc": auprc,
    }


def main() -> None:
    runs = [ln.strip() for ln in CANON.read_text().splitlines() if ln.strip()]
    if len(runs) != 70:
        raise SystemExit(f"Expected 70 canonical runs, got {len(runs)}")

    records = [load_run(r) for r in runs]
    by_key = {(r["view"], r["arch"], r["mode"]): r for r in records}
    if len(by_key) != 70:
        raise SystemExit(f"Duplicate view/arch/mode keys: {len(by_key)}")

    rows = []
    for view in VIEW_ORDER:
        for arch in ARCH_ORDER:
            raw = by_key[(view, arch, "raw")]
            for mode in MODE_ORDER:
                rec = by_key[(view, arch, mode)]
                auc_mu, auc_sd = mean_sd(rec["auc"])
                prc_mu, prc_sd = mean_sd(rec["auprc"])
                d_auc = [a - b for a, b in zip(rec["auc"], raw["auc"])]
                d_prc = [a - b for a, b in zip(rec["auprc"], raw["auprc"])]
                da_mu, da_sd = mean_sd(d_auc)
                dp_mu, dp_sd = mean_sd(d_prc)
                rows.append(
                    {
                        "view": view,
                        "backbone": ARCH_LABEL[arch],
                        "condition": MODE_LABEL[mode],
                        "arch": arch,
                        "mode": mode,
                        "fold_auroc_mean": auc_mu,
                        "fold_auroc_sd": auc_sd,
                        "fold_auprc_mean": prc_mu,
                        "fold_auprc_sd": prc_sd,
                        "delta_auroc_mean": da_mu,
                        "delta_auroc_sd": da_sd,
                        "delta_auprc_mean": dp_mu,
                        "delta_auprc_sd": dp_sd,
                        "same_direction_folds": same_direction(d_auc),
                        "n_folds": 5,
                        "run": rec["run"],
                        "raw_run": raw["run"],
                    }
                )

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "view",
                "backbone",
                "condition",
                "arch",
                "mode",
                "fold_auroc_mean",
                "fold_auroc_sd",
                "fold_auprc_mean",
                "fold_auprc_sd",
                "delta_auroc_mean",
                "delta_auroc_sd",
                "delta_auprc_mean",
                "delta_auprc_sd",
                "same_direction_folds",
                "n_folds",
                "run",
                "raw_run",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Fold-level AUROC / AUPRC versus raw",
        "",
        "Five-fold CV scores from the training pool (not the outer test set). "
        "Mean ± SD is across the five folds, using the same SD as `cv_auc_std` "
        "in the unified 70-run report (divide by n = 5). "
        "Δ is the fold-wise difference versus the matching raw run of the same "
        "view and backbone (same fold index). Same-direction folds counts how "
        "many of the five ΔAUROC values have the same sign as the mean ΔAUROC.",
        "",
        "| View | Backbone | Condition | Fold AUROC mean ± SD | Fold AUPRC mean ± SD | ΔAUROC vs raw mean ± SD | ΔAUPRC vs raw mean ± SD | Same-direction folds |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        if r["mode"] == "raw":
            d_auc = "—"
            d_prc = "—"
            same = "—"
        else:
            d_auc = fmt_mean_sd(r["delta_auroc_mean"], r["delta_auroc_sd"], signed=True)
            d_prc = fmt_mean_sd(r["delta_auprc_mean"], r["delta_auprc_sd"], signed=True)
            same = r["same_direction_folds"]
        lines.append(
            "| {view} | {bb} | {cond} | {auc} | {prc} | {da} | {dp} | {same} |".format(
                view=r["view"],
                bb=r["backbone"],
                cond=r["condition"],
                auc=fmt_mean_sd(r["fold_auroc_mean"], r["fold_auroc_sd"]),
                prc=fmt_mean_sd(r["fold_auprc_mean"], r["fold_auprc_sd"]),
                da=d_auc,
                dp=d_prc,
                same=same,
            )
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {len(rows)} rows -> {OUT_MD}")
    print(f"Wrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
