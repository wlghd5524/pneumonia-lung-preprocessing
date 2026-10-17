"""Clustered DeLong (Obuchowski) test for preprocessing-versus-raw dAUROC.

Primary hypothesis test for Table 4. The estimand is unchanged: image-level
outer-test AUROC of the fixed ensemble, raw versus each lung preprocessing
condition, for the same backbone and view. Predictions are read from the saved
outer-test files, so nothing is retrained.

Two correlations are handled at once:
  - pairing: both models score the same radiographs, so the structural
    components of the two AUROCs are differenced image by image;
  - clustering: radiographs from one subject are not independent, so the
    variance is a sum over subjects of the subject-level structural components
    (Obuchowski, Biometrics 1997;53:567-578), not over images.

With one image per subject the variance reduces to the paired DeLong variance.
The test statistic z = dAUROC / SE is referred to the standard normal, so the
p-value is an asymptotic approximation, not an exact test. Holm adjustment is
applied across the 56 view x backbone x preprocessing-versus-raw contrasts.

Pointwise 95% CIs are taken from the paired subject-level cluster bootstrap
(paired_cluster_bootstrap_delta.py) and are not adjusted for multiplicity. The
run directories and AUROC point estimates of the two sources must agree.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import norm

from paired_cluster_bootstrap_delta import (
    BASELINE_MODE,
    MIN_RUN_DATE,
    RESULTS_ROOT,
    fast_auc,
    load_runs,
    subject_of,
    write_csv,
)

MODEL_LABELS = {
    "resnet152": "ResNet-152",
    "densenet201": "DenseNet-201",
    "convnextv2_base": "ConvNeXt V2-Base",
    "swinv2_base_window16_256": "Swin V2-Base",
    "vit_base_patch16_dinov3": "DINOv3-Base",
    "eva_x_base": "EVA-X-Base",
    "microsoft/rad-dino": "RAD-DINO",
}
MODEL_ORDER = list(MODEL_LABELS)
MODE_LABELS = {
    "medsam3_seg": "MedSAM3 mask",
    "medsam3_crop": "MedSAM3 crop",
    "chexmask_seg": "CheXMask mask",
    "chexmask_crop": "CheXMask crop",
}
MODE_ORDER = list(MODE_LABELS)


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, capped at 1, in the input order."""
    m = int(pvals.size)
    order = np.argsort(pvals, kind="mergesort")
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * float(pvals[idx]))
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def fmt_bound(x: float) -> str:
    shown = float(f"{x:.4f}")
    if x != 0.0 and shown == 0.0:
        return f"{x:+.5f}"
    return f"{x:+.4f}"

N_CONTRASTS = 56


def placement_values(y: np.ndarray, p: np.ndarray):
    """DeLong structural components: V10 per positive image, V01 per negative image."""
    pos = p[y == 1]
    neg = p[y == 0]
    neg_sorted = np.sort(neg)
    pos_sorted = np.sort(pos)
    below = np.searchsorted(neg_sorted, pos, side="left")
    below_or_tied = np.searchsorted(neg_sorted, pos, side="right")
    v10 = (below + 0.5 * (below_or_tied - below)) / neg.size
    above = pos.size - np.searchsorted(pos_sorted, neg, side="right")
    above_or_tied = pos.size - np.searchsorted(pos_sorted, neg, side="left")
    v01 = (above + 0.5 * (above_or_tied - above)) / pos.size
    return v10, v01


def cluster_components(y, p, codes, n_clusters):
    """AUROC and per-subject centred structural-component sums."""
    v10, v01 = placement_values(y, p)
    theta = float(v10.mean())
    pos = y == 1
    m_i = np.bincount(codes[pos], minlength=n_clusters)
    n_i = np.bincount(codes[~pos], minlength=n_clusters)
    c10 = np.bincount(codes[pos], weights=v10, minlength=n_clusters) - m_i * theta
    c01 = np.bincount(codes[~pos], weights=v01, minlength=n_clusters) - n_i * theta
    return theta, c10, c01, m_i, n_i


def clustered_variance(d10, d01, m_i, n_i) -> float:
    """Obuchowski variance of a (difference of) clustered AUROC from centred component sums."""
    big_m = float(m_i.sum())
    big_n = float(n_i.sum())
    i10 = int(np.count_nonzero(m_i))
    i01 = int(np.count_nonzero(n_i))
    i_all = int(m_i.size)
    s10 = i10 / (i10 - 1) * float(np.sum(d10 * d10))
    s01 = i01 / (i01 - 1) * float(np.sum(d01 * d01))
    s11 = i_all / (i_all - 1) * float(np.sum(d10 * d01))
    return s10 / big_m**2 + s01 / big_n**2 + 2.0 * s11 / (big_m * big_n)


def delong_contrast(y, pa, pb, codes):
    n_clusters = int(codes.max()) + 1
    auc_a, a10, a01, m_i, n_i = cluster_components(y, pa, codes, n_clusters)
    auc_b, b10, b01, _, _ = cluster_components(y, pb, codes, n_clusters)
    var_a = clustered_variance(a10, a01, m_i, n_i)
    var_b = clustered_variance(b10, b01, m_i, n_i)
    var_d = clustered_variance(b10 - a10, b01 - a01, m_i, n_i)
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "se_auc_a": float(np.sqrt(var_a)),
        "se_auc_b": float(np.sqrt(var_b)),
        "corr_auc_a_b": float((var_a + var_b - var_d) / (2.0 * np.sqrt(var_a * var_b))),
        "se": float(np.sqrt(var_d)),
        "n_pos": int(m_i.sum()),
        "n_neg": int(n_i.sum()),
        "n_subjects": n_clusters,
        "n_subjects_with_pos": int(np.count_nonzero(m_i)),
        "n_subjects_with_neg": int(np.count_nonzero(n_i)),
        "n_subjects_with_both": int(np.count_nonzero((m_i > 0) & (n_i > 0))),
    }


def two_sided(delta: float, se: float) -> tuple[float, float]:
    """Two-sided normal p from the survival function, which keeps precision in the far tail."""
    z = delta / se
    p = float(2.0 * norm.sf(abs(z)))
    if p == 0.0:
        raise RuntimeError(f"two-sided p underflowed to 0 at z = {z:.2f}")
    return float(z), p


def load_bootstrap(path: Path) -> dict:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["mode_a"] == BASELINE_MODE and row["pairing"] == "paired":
                rows[(row["view"], row["model"], row["mode_b"])] = row
    if len(rows) != N_CONTRASTS:
        raise RuntimeError(f"{path}: expected {N_CONTRASTS} paired vs-raw rows, got {len(rows)}")
    return rows


def build_rows(runs: dict, boot: dict) -> list[dict]:
    rows = []
    for view in ("AP", "PA"):
        for model in MODEL_ORDER:
            a = runs[(view, BASELINE_MODE, model)]
            for mode in MODE_ORDER:
                b = runs[(view, mode, model)]
                if set(a["table"]) != set(b["table"]):
                    raise RuntimeError(f"{view} {model} {mode}: outer-test images differ from raw")
                keys = sorted(a["table"])
                y = np.array([a["table"][k][0] for k in keys], dtype=np.int8)
                if any(b["table"][k][0] != a["table"][k][0] for k in keys):
                    raise RuntimeError(f"{view} {model} {mode}: labels differ from raw")
                pa = np.array([a["table"][k][1] for k in keys])
                pb = np.array([b["table"][k][1] for k in keys])
                _, codes = np.unique([subject_of(k) for k in keys], return_inverse=True)

                res = delong_contrast(y, pa, pb, codes)
                if abs(res["auc_a"] - fast_auc(y, pa)) > 1e-12 or abs(res["auc_b"] - fast_auc(y, pb)) > 1e-12:
                    raise RuntimeError(f"{view} {model} {mode}: placement AUROC disagrees with rank AUROC")
                delta = res["auc_b"] - res["auc_a"]
                z, p = two_sided(delta, res["se"])

                # Same paired DeLong with every image treated as its own cluster,
                # kept only to show how much the subject clustering widens the SE.
                flat = delong_contrast(y, pa, pb, np.arange(y.size))
                z_flat, p_flat = two_sided(delta, flat["se"])

                bt = boot[(view, model, mode)]
                if bt["run_a"] != a["run"] or bt["run_b"] != b["run"]:
                    raise RuntimeError(f"{view} {model} {mode}: bootstrap CSV was built from different runs")
                if abs(float(bt["delta_auc"]) - delta) > 1e-9:
                    raise RuntimeError(f"{view} {model} {mode}: bootstrap dAUROC disagrees")

                rows.append({
                    "view": view,
                    "model": model,
                    "mode_a": BASELINE_MODE,
                    "mode_b": mode,
                    "n_images": int(y.size),
                    "n_pos": res["n_pos"],
                    "n_neg": res["n_neg"],
                    "n_subjects": res["n_subjects"],
                    "n_subjects_with_pos": res["n_subjects_with_pos"],
                    "n_subjects_with_neg": res["n_subjects_with_neg"],
                    "n_subjects_with_both": res["n_subjects_with_both"],
                    "auc_a": res["auc_a"],
                    "auc_b": res["auc_b"],
                    "delta_auc": delta,
                    "se_auc_a": res["se_auc_a"],
                    "se_auc_b": res["se_auc_b"],
                    "corr_auc_a_b": res["corr_auc_a_b"],
                    "delta_auc_se": res["se"],
                    "delta_auc_z": z,
                    "delta_auc_p": p,
                    "delta_auc_p_holm": float("nan"),
                    "delta_auc_se_unclustered": flat["se"],
                    "delta_auc_p_unclustered": p_flat,
                    "se_ratio_clustered_to_unclustered": res["se"] / flat["se"],
                    "boot_delta_auc_lo": float(bt["delta_auc_lo"]),
                    "boot_delta_auc_hi": float(bt["delta_auc_hi"]),
                    "boot_delta_auc_se": float(bt["delta_auc_se"]),
                    "boot_n_boot": int(bt["n_boot"]),
                    "run_a": a["run"],
                    "run_b": b["run"],
                })
    if len(rows) != N_CONTRASTS:
        raise RuntimeError(f"expected {N_CONTRASTS} contrasts, got {len(rows)}")
    holm = holm_adjust(np.array([r["delta_auc_p"] for r in rows]))
    for row, value in zip(rows, holm):
        row["delta_auc_p_holm"] = float(value)
    return rows


SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def fmt_p(p: float) -> str:
    """Three decimals from 0.001 up; below that, three significant figures in a × 10⁻ⁿ form.

    Display rounding only. Holm adjustment is applied to the unrounded values beforehand.
    """
    if p >= 0.001:
        return f"{p:.3f}"
    mantissa, exponent = f"{p:.2e}".split("e")
    return f"{mantissa} × 10{str(int(exponent)).translate(SUPERSCRIPT)}"


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


def write_table(path: Path, rows: list[dict]) -> None:
    n_boot = rows[0]["boot_n_boot"]
    lines = [
        "# Table 4. Per-backbone ΔAUROC versus raw input",
        "",
        (
            "Paired comparisons of each lung preprocessing condition with raw input on the held-out internal "
            "test set for each backbone and acquisition view. ΔAUROC is the image-level AUROC of the "
            "preprocessed model minus that of the corresponding raw-input model, calculated from unrounded "
            f"estimates. Pointwise 95% CIs are percentile intervals obtained from {n_boot:,} paired "
            "subject-level cluster bootstrap replicates and are not adjusted for multiple comparisons. "
            "Unadjusted two-sided p-values were calculated using a subject-clustered DeLong-type test that "
            "accounts for paired predictions on the same radiographs and correlations among radiographs from "
            f"the same patient. The Holm procedure was applied jointly to the {len(rows)} "
            "preprocessing-versus-raw comparisons using unrounded p-values. Both unadjusted and Holm-adjusted "
            "p-values are reported."
        ),
        "",
        "| View | Backbone | Preprocessing | Raw AUROC | Preprocessed AUROC | ΔAUROC vs. Raw [95% CI] | Unadjusted p | Holm-adjusted p |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        ci = f"{row['delta_auc']:+.4f} [{fmt_bound(row['boot_delta_auc_lo'])}, {fmt_bound(row['boot_delta_auc_hi'])}]"
        lines.append(
            f"| {row['view']} | {MODEL_LABELS[row['model']]} | {MODE_LABELS[row['mode_b']]} | "
            f"{row['auc_a']:.4f} | {row['auc_b']:.4f} | {ci} | "
            f"{fmt_p(row['delta_auc_p'])} | {fmt_p(row['delta_auc_p_holm'])} |"
        )
    write_lines(path, lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--runs-file", type=Path, default=None, help="Optional one-run-per-line manifest")
    parser.add_argument("--min-run-date", default=MIN_RUN_DATE)
    parser.add_argument(
        "--bootstrap-csv",
        type=Path,
        default=RESULTS_ROOT / "aggregates" / "_delta_paired_bootstrap_by_backbone.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=RESULTS_ROOT / "aggregates" / "_delta_auroc_vs_raw_clustered_delong.csv",
    )
    parser.add_argument(
        "--out-table",
        type=Path,
        default=RESULTS_ROOT / "paper_tables" / "Table4_delta_auroc_vs_raw_clustered_delong.md",
    )
    args = parser.parse_args()

    runs = load_runs(args.results_root, args.min_run_date, args.runs_file)
    rows = build_rows(runs, load_bootstrap(args.bootstrap_csv))
    write_csv(args.out_csv, rows)
    write_table(args.out_table, rows)

    print("\ncontrasts with Holm p < 0.05:", flush=True)
    shown = sorted((r for r in rows if r["delta_auc_p_holm"] < 0.05), key=lambda r: r["delta_auc_p"])
    if not shown:
        print("  none", flush=True)
    for row in shown:
        print(
            f"  {row['view']} {row['model']:<28} {row['mode_b']:<14} d={row['delta_auc']:+.4f} "
            f"se={row['delta_auc_se']:.4f} z={row['delta_auc_z']:+.2f} "
            f"p={fmt_p(row['delta_auc_p'])} holm={fmt_p(row['delta_auc_p_holm'])}",
            flush=True,
        )
    ratio = np.array([r["se_ratio_clustered_to_unclustered"] for r in rows])
    boot = np.array([r["delta_auc_se"] / r["boot_delta_auc_se"] for r in rows])
    print(f"\nSE clustered / unclustered: median {np.median(ratio):.3f} (range {ratio.min():.3f}-{ratio.max():.3f})")
    print(f"SE clustered DeLong / bootstrap: median {np.median(boot):.3f} (range {boot.min():.3f}-{boot.max():.3f})")


if __name__ == "__main__":
    main()
