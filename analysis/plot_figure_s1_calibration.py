#!/usr/bin/env python3
"""Supplementary Figure S1: calibration curves of five-model ensemble predictions."""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
RESULTS = Path(os.environ.get("RESULTS_DIR", PACKAGE_ROOT / "results_pneumonia")).resolve()
CANON = RESULTS / "_canonical_70runs.txt"
PRED_CSV = RESULTS / "_outer_test_predictions_with_subject.csv"
STEM = RESULTS / "Supplementary_Figure_S1_calibration"
CAPTION = RESULTS / "Supplementary_Figure_S1_caption.md"
BIN_CSV = RESULTS / "Supplementary_Figure_S1_calibration_bins.csv"

RUN_RE = re.compile(
    r"_(AP|PA)_(raw|medsam3_seg|medsam3_crop|chexmask_seg|chexmask_crop)_cv_5fold"
)
ARCH_RE = re.compile(
    r"^\d{8}_\d{6}_(.+)_(?:AP|PA)_(?:raw|medsam3_seg|medsam3_crop|chexmask_seg|chexmask_crop)_cv_5fold"
)

BACKBONES = [
    ("resnet152", "ResNet-152", "#0072B2", "o"),
    ("densenet", "DenseNet-201", "#E69F00", "s"),
    ("convnextv2_base", "ConvNeXt V2-B", "#009E73", "D"),
    ("swint_base", "Swin V2-B", "#CC79A7", "^"),
    ("dinov3_base", "DINOv3-B", "#56B4E9", "v"),
    ("eva_x_base", "EVA-X-B", "#D55E00", "P"),
    ("rad_dino", "RAD-DINO", "#000000", "X"),
]
PREPROCESS = [
    ("raw", "Raw"),
    ("medsam3_seg", "MedSAM3 masking"),
    ("medsam3_crop", "MedSAM3 cropping"),
    ("chexmask_seg", "CheXMask masking"),
    ("chexmask_crop", "CheXMask cropping"),
]
N_BINS = 10


def parse_run(run: str) -> tuple[str, str, str]:
    m = RUN_RE.search(run)
    a = ARCH_RE.match(run)
    if not m or not a:
        raise ValueError(f"Cannot parse run name: {run}")
    return a.group(1), m.group(1), m.group(2)


def reliability_points(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    y = y.astype(np.float64)
    p = np.clip(p.astype(np.float64), 1e-7, 1.0 - 1e-7)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(
            {
                "bin": i + 1,
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": n,
                "mean_predicted": float(p[mask].mean()),
                "observed_frequency": float(y[mask].mean()),
            }
        )
    return rows


def draw_panel(
    ax,
    curves: dict[str, list[dict]],
    show_ylabel: bool,
    show_xlabel_ticks: bool,
    col_idx: int,
    n_cols: int,
) -> None:
    ax.plot([0, 1], [0, 1], linestyle="--", color="#444444", linewidth=0.9, zorder=1)
    for arch, _label, color, marker in BACKBONES:
        pts = curves.get(arch, [])
        if not pts:
            continue
        xs = [d["mean_predicted"] for d in pts]
        ys = [d["observed_frequency"] for d in pts]
        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            markersize=4.0,
            markerfacecolor=color,
            markeredgewidth=0.0,
            linewidth=1.15,
            zorder=3,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    xlabels = ["0", "0.5", "1"]
    if col_idx > 0:
        xlabels[0] = ""
    if col_idx < n_cols - 1:
        xlabels[2] = ""
    ax.set_xticklabels(xlabels if show_xlabel_ticks else [])
    ax.set_yticklabels(["0", "0.5", "1"] if show_ylabel else [])
    ax.tick_params(axis="both", labelsize=7.4, length=3, pad=1.6)
    if not show_xlabel_ticks:
        ax.tick_params(labelbottom=False)
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    ax.grid(color="#eeeeee", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#777777")
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["left"].set_linewidth(0.7)
    if show_ylabel:
        ax.set_ylabel("Observed frequency", fontsize=8.4, labelpad=3)


def main() -> None:
    canon = [ln.strip() for ln in CANON.read_text().splitlines() if ln.strip()]
    if len(canon) != 70:
        raise SystemExit(f"Expected 70 canonical runs, got {len(canon)}")

    df = pd.read_csv(
        PRED_CSV,
        usecols=["run", "view", "mode", "label", "prob_ensemble"],
        dtype={"run": str, "view": str, "mode": str, "label": np.int8, "prob_ensemble": np.float64},
    )
    df = df[df["run"].isin(canon)].copy()
    meta = {run: parse_run(run) for run in canon}
    df["arch"] = df["run"].map(lambda r: meta[r][0])

    grouped: dict[tuple[str, str, str], list[dict]] = {}
    bin_rows = []
    for (view, mode, arch), sub in df.groupby(["view", "mode", "arch"], sort=False):
        pts = reliability_points(sub["label"].to_numpy(), sub["prob_ensemble"].to_numpy())
        grouped[(view, mode, arch)] = pts
        for pt in pts:
            bin_rows.append(
                {
                    "view": view,
                    "mode": mode,
                    "arch": arch,
                    **pt,
                }
            )

    n_subj = {"AP": 1763, "PA": 1743}
    n_img = {"AP": 3280, "PA": 2521}

    with BIN_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "view",
                "mode",
                "arch",
                "bin",
                "bin_lo",
                "bin_hi",
                "n",
                "mean_predicted",
                "observed_frequency",
            ],
        )
        w.writeheader()
        w.writerows(bin_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(13.4, 8.15))
    gs = fig.add_gridspec(
        4,
        5,
        height_ratios=[0.26, 1.00, 0.16, 1.00],
        left=0.048,
        right=0.988,
        top=0.975,
        bottom=0.155,
        wspace=0.16,
        hspace=0.10,
    )

    first_row_axes = []
    panel_rows = {0: 1, 1: 3}
    header_rows = {0: 0, 1: 2}

    for row_idx, view in enumerate(("AP", "PA")):
        header = fig.add_subplot(gs[header_rows[row_idx], :])
        header.axis("off")
        header.text(
            0.0,
            0.70,
            f"({'A' if row_idx == 0 else 'B'})  {view} view",
            transform=header.transAxes,
            ha="left",
            va="center",
            fontsize=12.4,
            fontweight="bold",
        )
        header.text(
            1.0,
            0.70,
            f"{n_subj[view]:,} subjects, {n_img[view]:,} images",
            transform=header.transAxes,
            ha="right",
            va="center",
            fontsize=8.3,
            color="#555555",
        )

        for col_idx, (mode, mode_label) in enumerate(PREPROCESS):
            ax = fig.add_subplot(gs[panel_rows[row_idx], col_idx])
            if row_idx == 0:
                first_row_axes.append(ax)
            curves = {
                arch: grouped[(view, mode, arch)]
                for arch, *_ in BACKBONES
                if (view, mode, arch) in grouped
            }
            if len(curves) != 7:
                missing = [a for a, *_ in BACKBONES if a not in curves]
                raise RuntimeError(f"{view}/{mode}: missing {missing}")
            draw_panel(
                ax,
                curves,
                show_ylabel=(col_idx == 0),
                show_xlabel_ticks=(row_idx == 1),
                col_idx=col_idx,
                n_cols=len(PREPROCESS),
            )
            if row_idx == 0:
                ax.set_title(mode_label, fontsize=9.6, pad=4)
            if row_idx == 1:
                ax.set_xlabel("Mean predicted probability", fontsize=8.0, labelpad=3)

    handles = [
        mlines.Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=1.15,
            markersize=5.4,
            markerfacecolor=color,
            markeredgewidth=0.0,
            label=label,
        )
        for _arch, label, color, marker in BACKBONES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=8.0,
        bbox_to_anchor=(0.52, 0.018),
        handletextpad=0.4,
        columnspacing=1.15,
    )
    fig.text(
        0.52,
        0.078,
        "Points are mean predicted probability versus observed event frequency in 10 equal-width bins. "
        "The dashed line is perfect calibration. Brier scores and expected calibration error are in Supplementary Table S5.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#333333",
    )

    for ext in ("png", "pdf", "svg"):
        path = STEM.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300)
        print(f"saved {path}")

    CAPTION.write_text(
        "Supplementary Figure S1. Calibration curves of the five-model ensemble predictions. "
        "Calibration curves are shown separately for AP and PA radiographs across the evaluated "
        "preprocessing conditions and backbone architectures. "
        "(A) AP view. (B) PA view. "
        "Observed event frequencies are plotted against mean predicted probabilities within "
        "10 probability bins; the diagonal line represents perfect calibration. "
        "Corresponding Brier scores and expected calibration errors are reported in Supplementary Table S5.\n",
        encoding="utf-8",
    )
    print(f"saved {CAPTION}")
    print(f"saved {BIN_CSV}")
    plt.close(fig)


if __name__ == "__main__":
    main()
