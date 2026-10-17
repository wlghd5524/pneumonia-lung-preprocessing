#!/usr/bin/env python3
"""Figure 5: backbone-specific ΔAUROC versus raw, by view and preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

STUDY_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = STUDY_DIR / "results"

BACKBONES = [
    ("resnet152", "ResNet-152"),
    ("densenet201", "DenseNet-201"),
    ("convnextv2_base", "ConvNeXt V2-B"),
    ("swinv2_base_window16_256", "Swin V2-B"),
    ("vit_base_patch16_dinov3", "DINOv3-B"),
    ("eva_x_base", "EVA-X-B"),
    ("microsoft/rad-dino", "RAD-DINO"),
]
PREPROCESS = [
    ("medsam3_seg", "MedSAM3 masking", "#c0392b"),
    ("medsam3_crop", "MedSAM3 cropping", "#2471a3"),
    ("chexmask_seg", "CheXMask masking", "#c0392b"),
    ("chexmask_crop", "CheXMask cropping", "#2471a3"),
]


def draw_panel(ax, frame: pd.DataFrame, color: str, show_ylabel: bool) -> None:
    by_model = {row.model: row for row in frame.itertuples()}
    y_pos = list(range(len(BACKBONES)))[::-1]
    ax.axvline(0.0, color="#222222", linewidth=0.9, zorder=1)
    for y, (model, _label) in zip(y_pos, BACKBONES):
        ax.axhline(y, color="#f2f2f2", linewidth=7.0, zorder=0)
        row = by_model[model]
        ax.plot(
            [row.delta_auc_lo, row.delta_auc_hi],
            [y, y],
            color=color,
            linewidth=1.8,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.plot(
            row.delta_auc,
            y,
            "o",
            markersize=5.8,
            color=color,
            markerfacecolor=color,
            markeredgewidth=1.2,
            zorder=3,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [label for _, label in BACKBONES] if show_ylabel else [],
        fontsize=8.2,
    )
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", labelsize=7.6, length=3, pad=2)
    ax.set_ylim(-0.7, len(BACKBONES) - 0.3)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#777777")
    ax.spines["bottom"].set_linewidth(0.7)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    pairs = pd.read_csv(RESULTS_ROOT / "primary" / "paired_delta_by_backbone.csv")
    vs_raw = pairs["vs_raw"].astype(str).str.lower().isin({"true", "1"})
    pairs = pairs[vs_raw & (pairs["pairing"] == "paired")].copy()
    if len(pairs) != 56:
        raise RuntimeError(f"expected 56 vs-raw paired rows, got {len(pairs)}")

    xmin = float(pairs["delta_auc_lo"].min())
    xmax = float(pairs["delta_auc_hi"].max())
    xlim = (xmin - 0.0035, xmax + 0.0035)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(11.8, 7.85))
    gs = fig.add_gridspec(
        4, 4,
        height_ratios=[0.28, 1.00, 0.16, 1.00],
        left=0.125,
        right=0.985,
        top=0.97,
        bottom=0.165,
        wspace=0.10,
        hspace=0.05,
    )

    first_row_axes = []
    panel_rows = {0: 1, 1: 3}
    header_rows = {0: 0, 1: 2}

    for row_idx, view in enumerate(("AP", "PA")):
        n_subj = int(pairs.loc[pairs.view == view, "n_subjects_paired"].iloc[0])
        n_img = int(pairs.loc[pairs.view == view, "n_images_paired"].iloc[0])
        header = fig.add_subplot(gs[header_rows[row_idx], :])
        header.axis("off")
        header.text(
            0.0, 0.72,
            f"({'A' if row_idx == 0 else 'B'})  {view} view",
            transform=header.transAxes,
            ha="left",
            va="center",
            fontsize=12.5,
            fontweight="bold",
        )
        header.text(
            1.0, 0.72,
            f"{n_subj:,} subjects, {n_img:,} images",
            transform=header.transAxes,
            ha="right",
            va="center",
            fontsize=8.4,
            color="#555555",
        )

        for col_idx, (mode, mode_label, color) in enumerate(PREPROCESS):
            sharex = first_row_axes[col_idx] if row_idx == 1 else (first_row_axes[0] if first_row_axes else None)
            ax = fig.add_subplot(gs[panel_rows[row_idx], col_idx], sharex=sharex)
            if row_idx == 0:
                first_row_axes.append(ax)
            sub = pairs[(pairs.view == view) & (pairs.mode_b == mode)]
            draw_panel(ax, sub, color, show_ylabel=(col_idx == 0))
            ax.set_xlim(*xlim)
            if row_idx == 0:
                ax.set_title(mode_label, fontsize=10.3, pad=5)
            if row_idx == 1:
                ax.set_xlabel(r"$\Delta$AUROC vs. raw", fontsize=8.6, labelpad=4)
            else:
                ax.tick_params(labelbottom=False)

    handles = [
        mlines.Line2D(
            [], [], color="#c0392b", marker="o", linestyle="None",
            markersize=6.2, markerfacecolor="#c0392b", markeredgewidth=1.2,
            label="Masking",
        ),
        mlines.Line2D(
            [], [], color="#2471a3", marker="o", linestyle="None",
            markersize=6.2, markerfacecolor="#2471a3", markeredgewidth=1.2,
            label="Cropping",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=8.4,
        bbox_to_anchor=(0.555, 0.002),
        handletextpad=0.5,
        columnspacing=2.0,
    )
    fig.text(
        0.555, 0.078,
        "Positive $\\Delta$AUROC indicates higher discrimination than the matched raw-input model. "
        "Error bars represent paired subject-level cluster-bootstrap 95% confidence intervals based on 20,000 replicates.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#333333",
    )
    fig.text(
        0.555, 0.052,
        "Multiplicity-adjusted statistical inference is reported separately using Holm-adjusted $p$ values in Table 4.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#333333",
    )

    stem = RESULTS_ROOT / "figures" / "Figure_5_delta_auroc_vs_raw"
    for ext in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300)
        print(f"saved {path}")

    caption_path = RESULTS_ROOT / "figures" / "Figure_5_caption.md"
    caption_path.write_text(
        "Figure 5. Backbone-specific changes in held-out test AUROC relative to raw input "
        "for all four lung preprocessing conditions within each acquisition view. "
        "(A) AP view. (B) PA view. "
        "Differences are shown for MedSAM3 masking, MedSAM3 cropping, CheXMask masking, "
        "and CheXMask cropping. "
        "Positive $\\Delta$AUROC indicates higher discrimination than the matched raw-input model. "
        "Error bars represent paired subject-level cluster-bootstrap 95% confidence intervals "
        "based on 20,000 replicates and are shown to convey effect size and uncertainty only. "
        "Multiplicity-adjusted statistical inference is reported separately "
        "using Holm-adjusted $p$ values in Table 4.\n",
        encoding="utf-8",
    )
    print(f"saved {caption_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
