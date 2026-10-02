"""Summarize paired preprocessing effects across training seeds 42, 123, and 2026.

Patient partitions stay fixed. Seed 42 is the canonical results_pneumonia run.
Seeds 123 and 2026 live in results_pneumonia_seed123 and results_pneumonia_seed2026.
"""

from __future__ import annotations

import os

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", STUDY / "results_pneumonia")).resolve()
ARCHS = ["resnet152", "eva_x_base"]
MODES = ["raw", "medsam3_seg", "medsam3_crop", "chexmask_seg", "chexmask_crop"]
MODE_LABEL = {
    "raw": "Raw",
    "medsam3_seg": "MedSAM3 mask",
    "medsam3_crop": "MedSAM3 crop",
    "chexmask_seg": "CheXMask mask",
    "chexmask_crop": "CheXMask crop",
}
ARCH_LABEL = {"resnet152": "ResNet-152", "eva_x_base": "EVA-X-Base"}
ORDER = ["medsam3_crop", "chexmask_crop", "medsam3_seg", "chexmask_seg"]
SEEDS = (42, 123, 2026)
TEXT_COLS = {"View", "Backbone", "Preprocessing", "Contrast", "Direction", "Input"}


def load_summary(root: Path, run: str) -> dict:
    with (root / run / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    cfg = summary["config"]
    outer = summary["outer_test"]
    return {
        "run": run,
        "arch": cfg["arch"],
        "view": cfg["view"],
        "mode": cfg["data_mode"],
        "seed": int(cfg["seed"]),
        "auc": float(outer["auc"]),
        "auprc": float(outer["auprc"]),
        "n_samples": int(outer["n_samples"]),
        "n_pos": int(outer["n_pos"]),
        "n_models": int(outer["n_models_ensembled"]),
        "executed_folds": int(summary["executed_folds"]),
    }


def groups_hash(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        assignment = json.load(handle)["assignment"]
    folds = [
        {
            "i": index,
            "train": sorted(fold["train_groups"]),
            "val": sorted(fold["val_groups"]),
        }
        for index, fold in enumerate(assignment["folds"])
    ]
    payload = {
        "outer_test": sorted(assignment["outer_test_groups"]),
        "outer_trainval": sorted(assignment["outer_trainval_groups"]),
        "folds": folds,
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def prediction_id(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    path_index = header.index("img_path")
    label_index = header.index("label")
    pairs = []
    for line in lines[1:]:
        parts = line.split(",")
        pairs.append((Path(parts[path_index]).name, parts[label_index]))
    pairs.sort()
    encoded = "\n".join(f"{name}|{label}" for name, label in pairs).encode()
    return hashlib.sha256(encoded).hexdigest()


def collect() -> pd.DataFrame:
    rows = []
    root42 = RESULTS_DIR
    canon = [
        line.strip()
        for line in (root42 / "aggregates" / "_canonical_70runs.txt").read_text().splitlines()
        if line.strip()
    ]
    allowed_arch = set(ARCHS)
    allowed_mode = set(MODES)
    for run in canon:
        record = load_summary(root42, run)
        if record["arch"] in allowed_arch and record["mode"] in allowed_mode:
            record.update(seed_label=42, root=str(root42))
            rows.append(record)
    for seed, folder in ((123, "results_pneumonia_seed123"), (2026, "results_pneumonia_seed2026")):
        root = RESULTS_DIR.parent / folder
        for summary_path in sorted(root.glob("*/summary.json")):
            record = load_summary(root, summary_path.parent.name)
            if record["seed"] != seed:
                raise RuntimeError(f"seed mismatch in {summary_path}: {record['seed']}")
            record.update(seed_label=seed, root=str(root))
            rows.append(record)
    frame = pd.DataFrame(rows)
    if len(frame) != 60:
        raise RuntimeError(f"expected 60 runs, found {len(frame)}")
    if set(frame["seed_label"]) != set(SEEDS):
        raise RuntimeError(f"unexpected seeds: {sorted(frame['seed_label'].unique())}")
    if frame.duplicated(["view", "arch", "mode", "seed_label"]).any():
        raise RuntimeError("duplicate view/arch/mode/seed rows")
    return frame


def assert_fixed_partitions(frame: pd.DataFrame) -> None:
    split_hashes: dict[str, set[str]] = {}
    image_hashes: dict[str, set[str]] = {}
    for record in frame.to_dict(orient="records"):
        root = Path(record["root"])
        run = record["run"]
        split_hashes.setdefault(record["view"], set()).add(
            groups_hash(root / run / "split_assignment.json")
        )
        image_hashes.setdefault(record["view"], set()).add(
            prediction_id(root / run / "outer_test" / "outer_test_predictions.csv")
        )
    for view, hashes in split_hashes.items():
        if len(hashes) != 1:
            raise RuntimeError(f"{view} patient partitions differ across runs")
    for view, hashes in image_hashes.items():
        if len(hashes) != 1:
            raise RuntimeError(f"{view} outer-test images or labels differ across runs")


def seed_values(group: pd.DataFrame, column: str) -> np.ndarray:
    by_seed = group.set_index("seed_label")[column]
    return np.array([float(by_seed.loc[seed]) for seed in SEEDS])


def direction_vs_raw(values: np.ndarray) -> str:
    positive = int((values > 0).sum())
    negative = int((values < 0).sum())
    if positive == 3:
        return "3/3 higher than raw"
    if negative == 3:
        return "3/3 lower than raw"
    if positive > negative:
        return f"{positive}/3 higher than raw"
    return f"{negative}/3 lower than raw"


def direction_crop_mask(values: np.ndarray) -> str:
    positive = int((values > 0).sum())
    negative = int((values < 0).sum())
    if positive == 3:
        return "3/3 crop higher"
    if negative == 3:
        return "3/3 mask higher"
    if positive > negative:
        return f"{positive}/3 crop higher"
    return f"{negative}/3 mask higher"


def paired_vs_raw(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = frame[frame["mode"] == "raw"][
        ["seed_label", "view", "arch", "auc", "auprc"]
    ].rename(columns={"auc": "raw_auc", "auprc": "raw_auprc"})
    paired = frame[frame["mode"] != "raw"].merge(raw, on=["seed_label", "view", "arch"])
    paired["d_auc"] = paired["auc"] - paired["raw_auc"]
    paired["d_auprc"] = paired["auprc"] - paired["raw_auprc"]

    backbone_rows = []
    for view in ("AP", "PA"):
        for arch in ARCHS:
            for mode in ORDER:
                group = paired[
                    (paired["view"] == view)
                    & (paired["arch"] == arch)
                    & (paired["mode"] == mode)
                ]
                delta_auc = seed_values(group, "d_auc")
                delta_auprc = seed_values(group, "d_auprc")
                backbone_rows.append(
                    {
                        "view": view,
                        "arch": ARCH_LABEL[arch],
                        "mode": MODE_LABEL[mode],
                        "d42": delta_auc[0],
                        "d123": delta_auc[1],
                        "d2026": delta_auc[2],
                        "mean": float(delta_auc.mean()),
                        "sd": float(delta_auc.std(ddof=1)),
                        "range": float(delta_auc.max() - delta_auc.min()),
                        "direction": direction_vs_raw(delta_auc),
                        "a42": delta_auprc[0],
                        "a123": delta_auprc[1],
                        "a2026": delta_auprc[2],
                        "amean": float(delta_auprc.mean()),
                        "asd": float(delta_auprc.std(ddof=1)),
                        "adir": direction_vs_raw(delta_auprc),
                    }
                )
    by_backbone = pd.DataFrame(backbone_rows)

    mean_rows = []
    for view in ("AP", "PA"):
        for mode in ORDER:
            auc_means = []
            auprc_means = []
            for seed in SEEDS:
                subset = paired[
                    (paired["view"] == view)
                    & (paired["mode"] == mode)
                    & (paired["seed_label"] == seed)
                ]
                auc_means.append(float(subset["d_auc"].mean()))
                auprc_means.append(float(subset["d_auprc"].mean()))
            auc_means = np.array(auc_means)
            auprc_means = np.array(auprc_means)
            mean_rows.append(
                {
                    "view": view,
                    "mode": MODE_LABEL[mode],
                    "d42": float(auc_means[0]),
                    "d123": float(auc_means[1]),
                    "d2026": float(auc_means[2]),
                    "mean": float(auc_means.mean()),
                    "sd": float(auc_means.std(ddof=1)),
                    "range": float(auc_means.max() - auc_means.min()),
                    "direction": direction_vs_raw(auc_means),
                    "a42": float(auprc_means[0]),
                    "a123": float(auprc_means[1]),
                    "a2026": float(auprc_means[2]),
                    "amean": float(auprc_means.mean()),
                    "asd": float(auprc_means.std(ddof=1)),
                    "adir": direction_vs_raw(auprc_means),
                }
            )
    return paired, by_backbone, pd.DataFrame(mean_rows)


def crop_minus_mask(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = (
        ("MedSAM3", "medsam3_crop", "medsam3_seg"),
        ("CheXMask", "chexmask_crop", "chexmask_seg"),
    )
    for segmenter, crop_mode, mask_mode in pairs:
        crop = frame[frame["mode"] == crop_mode][
            ["seed_label", "view", "arch", "auc", "auprc"]
        ].rename(columns={"auc": "crop_auc", "auprc": "crop_auprc"})
        mask = frame[frame["mode"] == mask_mode][
            ["seed_label", "view", "arch", "auc", "auprc"]
        ].rename(columns={"auc": "mask_auc", "auprc": "mask_auprc"})
        joined = crop.merge(mask, on=["seed_label", "view", "arch"])
        joined["d_auc"] = joined["crop_auc"] - joined["mask_auc"]
        joined["d_auprc"] = joined["crop_auprc"] - joined["mask_auprc"]
        contrast = f"{segmenter} crop − mask"
        for view in ("AP", "PA"):
            for arch in ARCHS:
                group = joined[(joined["view"] == view) & (joined["arch"] == arch)]
                delta_auc = seed_values(group, "d_auc")
                delta_auprc = seed_values(group, "d_auprc")
                rows.append(
                    {
                        "view": view,
                        "arch": ARCH_LABEL[arch],
                        "contrast": contrast,
                        "d42": delta_auc[0],
                        "d123": delta_auc[1],
                        "d2026": delta_auc[2],
                        "mean": float(delta_auc.mean()),
                        "sd": float(delta_auc.std(ddof=1)),
                        "range": float(delta_auc.max() - delta_auc.min()),
                        "direction": direction_crop_mask(delta_auc),
                        "a42": delta_auprc[0],
                        "a123": delta_auprc[1],
                        "a2026": delta_auprc[2],
                        "amean": float(delta_auprc.mean()),
                        "asd": float(delta_auprc.std(ddof=1)),
                    }
                )
            auc_means = []
            auprc_means = []
            for seed in SEEDS:
                subset = joined[(joined["view"] == view) & (joined["seed_label"] == seed)]
                auc_means.append(float(subset["d_auc"].mean()))
                auprc_means.append(float(subset["d_auprc"].mean()))
            auc_means = np.array(auc_means)
            auprc_means = np.array(auprc_means)
            rows.append(
                {
                    "view": view,
                    "arch": "Two-backbone mean",
                    "contrast": contrast,
                    "d42": float(auc_means[0]),
                    "d123": float(auc_means[1]),
                    "d2026": float(auc_means[2]),
                    "mean": float(auc_means.mean()),
                    "sd": float(auc_means.std(ddof=1)),
                    "range": float(auc_means.max() - auc_means.min()),
                    "direction": direction_crop_mask(auc_means),
                    "a42": float(auprc_means[0]),
                    "a123": float(auprc_means[1]),
                    "a2026": float(auprc_means[2]),
                    "amean": float(auprc_means.mean()),
                    "asd": float(auprc_means.std(ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def absolute_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for view in ("AP", "PA"):
        for arch in ARCHS:
            for mode in MODES:
                group = frame[
                    (frame["view"] == view) & (frame["arch"] == arch) & (frame["mode"] == mode)
                ].set_index("seed_label")
                for metric, column in (("AUROC", "auc"), ("AUPRC", "auprc")):
                    values = np.array([float(group.loc[seed, column]) for seed in SEEDS])
                    rows.append(
                        {
                            "view": view,
                            "arch": ARCH_LABEL[arch],
                            "mode": MODE_LABEL[mode],
                            "metric": metric,
                            "s42": float(values[0]),
                            "s123": float(values[1]),
                            "s2026": float(values[2]),
                            "mean": float(values.mean()),
                            "sd": float(values.std(ddof=1)),
                            "range": float(values.max() - values.min()),
                        }
                    )
    return pd.DataFrame(rows)


def architecture_gap(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for view in ("AP", "PA"):
        for mode in MODES:
            for seed in SEEDS:
                eva = float(
                    frame[
                        (frame["view"] == view)
                        & (frame["arch"] == "eva_x_base")
                        & (frame["mode"] == mode)
                        & (frame["seed_label"] == seed)
                    ]["auc"].iloc[0]
                )
                resnet = float(
                    frame[
                        (frame["view"] == view)
                        & (frame["arch"] == "resnet152")
                        & (frame["mode"] == mode)
                        & (frame["seed_label"] == seed)
                    ]["auc"].iloc[0]
                )
                rows.append(
                    {
                        "view": view,
                        "mode": MODE_LABEL[mode],
                        "seed": seed,
                        "eva_auc": eva,
                        "resnet_auc": resnet,
                        "gap": eva - resnet,
                    }
                )
    return pd.DataFrame(rows)


def signed(value: float) -> str:
    return f"{value:+.4f}"


def plain(value: float) -> str:
    return f"{value:.4f}"


def markdown_table(headers: list[str], records: list[dict], getters) -> str:
    aligns = ["---" if header in TEXT_COLS else "---:" for header in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for record in records:
        lines.append("| " + " | ".join(getters(record)) + " |")
    return "\n".join(lines)


def write_markdown(
    by_backbone: pd.DataFrame,
    two_backbone: pd.DataFrame,
    crop_mask: pd.DataFrame,
    absolute: pd.DataFrame,
    gap: pd.DataFrame,
    destination: Path,
) -> None:
    mean_table = markdown_table(
        ["View", "Preprocessing", "Seed 42", "Seed 123", "Seed 2026", "Mean", "SD", "Direction"],
        two_backbone.to_dict("records"),
        lambda row: [
            row["view"],
            row["mode"],
            signed(row["d42"]),
            signed(row["d123"]),
            signed(row["d2026"]),
            signed(row["mean"]),
            plain(row["sd"]),
            row["direction"],
        ],
    )
    backbone_table = markdown_table(
        ["View", "Backbone", "Preprocessing", "Seed 42", "Seed 123", "Seed 2026", "Mean", "SD", "Direction"],
        by_backbone.to_dict("records"),
        lambda row: [
            row["view"],
            row["arch"],
            row["mode"],
            signed(row["d42"]),
            signed(row["d123"]),
            signed(row["d2026"]),
            signed(row["mean"]),
            plain(row["sd"]),
            row["direction"],
        ],
    )
    crop_table = markdown_table(
        ["View", "Backbone", "Contrast", "Seed 42", "Seed 123", "Seed 2026", "Mean", "SD", "Direction"],
        crop_mask.to_dict("records"),
        lambda row: [
            row["view"],
            row["arch"],
            row["contrast"],
            signed(row["d42"]),
            signed(row["d123"]),
            signed(row["d2026"]),
            signed(row["mean"]),
            plain(row["sd"]),
            row["direction"],
        ],
    )
    auprc_table = markdown_table(
        ["View", "Backbone", "Preprocessing", "Seed 42", "Seed 123", "Seed 2026", "Mean", "SD", "Direction"],
        by_backbone.to_dict("records"),
        lambda row: [
            row["view"],
            row["arch"],
            row["mode"],
            signed(row["a42"]),
            signed(row["a123"]),
            signed(row["a2026"]),
            signed(row["amean"]),
            plain(row["asd"]),
            row["adir"],
        ],
    )
    absolute_auc = absolute[absolute["metric"] == "AUROC"]
    absolute_table = markdown_table(
        ["View", "Backbone", "Input", "Seed 42", "Seed 123", "Seed 2026", "Mean", "SD"],
        absolute_auc.to_dict("records"),
        lambda row: [
            row["view"],
            row["arch"],
            row["mode"],
            plain(row["s42"]),
            plain(row["s123"]),
            plain(row["s2026"]),
            plain(row["mean"]),
            plain(row["sd"]),
        ],
    )
    gap_min = float(gap["gap"].min())
    gap_max = float(gap["gap"].max())
    text = f"""# Training-seed robustness of paired preprocessing effects

Patient partitions were held fixed. Every run of a given view used the same outer-test patients and the same five development folds (split seed 42; subject lists hashed to one value per view across all 30 runs). Only the training seed changed: 42 (the canonical run), 123, and 2026. Each AUROC is the five-model ensemble on the untouched outer-test set.

Scope of the repeat: AP and PA × ResNet-152 and EVA-X-Base × raw, MedSAM3 mask, MedSAM3 crop, CheXMask mask, and CheXMask crop. That is 20 configurations × 3 seeds = 60 runs. The other five backbones were not retrained.

Δ is preprocessed minus raw, matched on view, backbone, and training seed. Mean and SD are across the three seeds. SD divides by n−1 = 2. Positive means the preprocessed model scored higher than raw.

The outer-test images and labels were the same across seeds and preprocessing conditions within a view (matched on DICOM file name and label): AP 3,280 images, 1,582 pneumonia-positive; PA 2,521 images, 750 pneumonia-positive.

## 1. Two-backbone mean ΔAUROC versus raw

This is the same summary used in the main preprocessing comparison: the average of ResNet-152 and EVA-X-Base.

{mean_table}

## 2. Backbone-specific ΔAUROC versus raw

{backbone_table}

## 3. Crop minus hard mask, ΔAUROC

{crop_table}

## 4. Backbone-specific ΔAUPRC versus raw

{auprc_table}

## 5. Absolute outer-test AUROC

{absolute_table}

## Reading

Hard-mask versus raw was lower in all 8 backbone-specific contrasts and in all 3 seeds (24 of 24 paired comparisons). The two-backbone mean drop was about 0.003 to 0.006 on AP and about 0.015 to 0.022 on PA. Seed-to-seed SD of those mean drops was 0.0006 to 0.0040, smaller than the drop itself on PA and usually smaller on AP as well.

Crop versus raw stayed small. Across the 8 backbone-specific crop contrasts, only 3 kept the same sign in all three seeds (AP ResNet-152, both crops, slightly higher; PA ResNet-152 MedSAM3 crop, slightly lower). The other five changed sign. Two-backbone mean crop effects on AP stayed slightly positive (about +0.002) in all three seeds. On PA they sat within about ±0.002 of zero and did not keep one direction. For crops, the seed SD was often as large as the mean difference.

Crop minus the matching hard mask stayed positive in all 8 backbone-specific contrasts and all 3 seeds. That direction is the stable paired effect: cropping scored higher than hard masking, while cropping did not reliably score higher than raw.

EVA-X-Base outer-test AUROC was higher than ResNet-152 in all 30 view × input × seed comparisons (gap {gap_min:+.4f} to {gap_max:+.4f}). This repeat does not change the caution on architectural superiority. Only two backbones were retrained, and a stable ordering of these two models is not a claim that one architecture class is superior.
"""
    destination.write_text(text, encoding="utf-8")


def write_response(gap: pd.DataFrame, destination: Path) -> None:
    gap_min = float(gap["gap"].min())
    gap_max = float(gap["gap"].max())
    text = f"""# Response: independent training-seed robustness

We repeated the principal comparisons with three independent training seeds (42, 123, and 2026) while holding the patient partitions fixed. Split seed remained 42. For each view, the outer-test patients and the five development folds were identical across seeds, backbones, and preprocessing conditions. The repeated scope was AP and PA, ResNet-152 and EVA-X-Base, and the five inputs used in the main comparison (raw, MedSAM3 mask, MedSAM3 crop, CheXMask mask, and CheXMask crop): 20 configurations × 3 seeds. Each score is the five-fold ensemble AUROC on the same outer-test radiographs. Fold-to-fold variation is not used as a substitute for this training-seed variability.

Paired differences were formed within each seed (preprocessed minus raw, same view and backbone). Across seeds, hard masking remained lower than raw in every backbone-specific contrast (8 contrasts × 3 seeds). The two-backbone mean ΔAUROC was about −0.003 to −0.006 on AP and about −0.015 to −0.022 on PA, and the seed-to-seed SD of those means was smaller than the mean drop, particularly on PA. Crop versus raw remained small. Several backbone-specific crop effects changed sign across seeds, and the seed SD was often comparable to the mean difference. The two-backbone mean crop effect stayed slightly positive on AP (about +0.002) and was indistinguishable from zero on PA. Crop minus the matched hard mask stayed positive in all 8 backbone-specific contrasts and all 3 seeds. We therefore treat the masking-related decrease, and the advantage of crop over hard mask, as directionally consistent, and we do not interpret the small crop-versus-raw differences as a stable gain.

The caution regarding architectural superiority is unchanged. EVA-X-Base remained above ResNet-152 in these repeats (outer-test AUROC gap {gap_min:+.4f} to {gap_max:+.4f} across 30 view × input × seed comparisons), but only two backbones were retrained, and that ordering is not a claim of architectural superiority.
"""
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    frame = collect()
    assert_fixed_partitions(frame)
    paired, by_backbone, two_backbone = paired_vs_raw(frame)
    crop_mask = crop_minus_mask(frame)
    absolute = absolute_metrics(frame)
    gap = architecture_gap(frame)
    if not bool((gap["gap"] > 0).all()):
        raise RuntimeError("EVA-X was not above ResNet in every comparison")
    if not bool(crop_mask["direction"].eq("3/3 crop higher").all()):
        raise RuntimeError("crop-minus-mask direction was not uniformly positive")

    out = RESULTS_DIR / "analysis_outputs" / "seed_robustness"
    out.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=["root"]).to_csv(out / "seed_outer_metrics.csv", index=False)
    paired.to_csv(out / "seed_paired_deltas_long.csv", index=False)
    by_backbone.to_csv(out / "seed_paired_delta_by_backbone.csv", index=False)
    two_backbone.to_csv(out / "seed_paired_delta_two_backbone_mean.csv", index=False)
    crop_mask.to_csv(out / "seed_crop_minus_mask.csv", index=False)
    absolute.to_csv(out / "seed_absolute_metrics.csv", index=False)
    gap.to_csv(out / "seed_eva_minus_resnet.csv", index=False)
    write_markdown(
        by_backbone,
        two_backbone,
        crop_mask,
        absolute,
        gap,
        out / "Training_seed_robustness.md",
    )
    write_response(
        gap,
        RESULTS_DIR / "paper_text" / "Response_letter_training_seed.md",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
