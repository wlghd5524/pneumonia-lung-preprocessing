#!/usr/bin/env python3
"""Score lung masks against a reference mask directory.

Dice, IoU, and HD95 follow evaluation_protocol.md. The script reads masks
only from directories the caller supplies. It does not download data and it
does not record filenames in the default summary.

PNG lung pixels are gray values greater than 0. A .npy mask is converted
with astype(bool). When the two masks differ in size, the prediction is
resized to the reference with nearest-neighbor interpolation and then
binarized again.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

MASK_SUFFIXES = {".png", ".npy"}


def load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.asarray(np.load(path))
        if array.ndim != 2:
            raise ValueError(f"{path.name} is not a 2-D mask")
        return array.astype(bool)
    image = np.array(Image.open(path).convert("L"))
    if image.ndim != 2:
        raise ValueError(f"{path.name} is not a 2-D mask")
    return image > 0


def align(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    if reference.shape == other.shape:
        return other
    image = Image.fromarray(other.astype(np.uint8) * 255)
    image = image.resize((reference.shape[1], reference.shape[0]), resample=Image.Resampling.NEAREST)
    return np.array(image) > 0


def dice_iou(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    reference = reference.astype(bool)
    prediction = prediction.astype(bool)
    intersection = int(np.logical_and(reference, prediction).sum())
    reference_area = int(reference.sum())
    prediction_area = int(prediction.sum())
    union = reference_area + prediction_area - intersection
    dice = (2.0 * intersection / (reference_area + prediction_area)) if (reference_area + prediction_area) else float("nan")
    iou = (intersection / union) if union else float("nan")
    return float(dice), float(iou)


def hd95(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Manuscript HD95: pooled boundary-to-foreground distances.

    Each boundary is the foreground removed by one erosion. The distance at a
    boundary pixel is the Euclidean distance to the nearest foreground pixel
    of the other mask, including that mask's interior. Both directions are
    pooled, and the score is one 95th percentile.

    This is the definition behind the reported 100-image results. It is not
    boundary-to-boundary surface distance, and it is not the maximum of the
    two directed 95th percentiles used by MONAI. Do not replace the distance
    target without recomputing those results.
    """
    reference = reference.astype(bool)
    prediction = prediction.astype(bool)
    if reference.sum() == 0 or prediction.sum() == 0:
        return float("nan")
    reference_boundary = reference ^ ndimage.binary_erosion(reference, iterations=1, border_value=0)
    prediction_boundary = prediction ^ ndimage.binary_erosion(prediction, iterations=1, border_value=0)
    if reference_boundary.sum() == 0:
        reference_boundary = reference
    if prediction_boundary.sum() == 0:
        prediction_boundary = prediction
    reference_to_prediction = ndimage.distance_transform_edt(~prediction)[reference_boundary]
    prediction_to_reference = ndimage.distance_transform_edt(~reference)[prediction_boundary]
    distances = np.concatenate([reference_to_prediction.ravel(), prediction_to_reference.ravel()])
    return float(np.percentile(distances, 95))


def summarize(values: list[float]) -> dict:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)),
        "q1": float(np.percentile(finite, 25)),
        "q3": float(np.percentile(finite, 75)),
    }


def index_masks(directory: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in MASK_SUFFIXES:
            continue
        if path.name.endswith(".paint.png"):
            continue
        if path.stem in found:
            raise SystemExit(f"Two mask files in {directory} share the same stem.")
        found[path.stem] = path
    return found


def _summary(
    reference_index: dict[str, Path],
    prediction_index: dict[str, Path],
    paired_stems: list[str],
    rows: list[dict],
    resampled: int,
    empty_reference: int,
    empty_prediction: int,
) -> dict:
    return {
        "n_reference": len(reference_index),
        "n_prediction": len(prediction_index),
        "n_paired": len(paired_stems),
        "n_unmatched_reference": len(reference_index) - len(paired_stems),
        "n_unmatched_prediction": len(prediction_index) - len(paired_stems),
        "n_resampled": resampled,
        "n_empty_reference": empty_reference,
        "n_empty_prediction": empty_prediction,
        "dice": summarize([row["dice"] for row in rows]),
        "iou": summarize([row["iou"] for row in rows]),
        "hd95_px": summarize([row["hd95_px"] for row in rows]),
    }


def score_pair(reference: np.ndarray, prediction: np.ndarray) -> dict:
    resampled = reference.shape != prediction.shape
    aligned = align(reference, prediction)
    dice, iou = dice_iou(reference, aligned)
    return {
        "dice": dice,
        "iou": iou,
        "hd95_px": hd95(reference, aligned),
        "resampled": resampled,
        "empty_reference": int(reference.astype(bool).sum()) == 0,
        "empty_prediction": int(aligned.sum()) == 0,
    }


def score_directories(reference_dir: Path, prediction_dir: Path) -> tuple[dict, list[dict]]:
    reference_index = index_masks(reference_dir)
    prediction_index = index_masks(prediction_dir)
    paired_stems = sorted(set(reference_index) & set(prediction_index))
    rows: list[dict] = []
    resampled = 0
    empty_reference = 0
    empty_prediction = 0
    for stem in paired_stems:
        reference = load_mask(reference_index[stem])
        prediction = load_mask(prediction_index[stem])
        metrics = score_pair(reference, prediction)
        resampled += int(metrics["resampled"])
        empty_reference += int(metrics["empty_reference"])
        empty_prediction += int(metrics["empty_prediction"])
        rows.append({"file_stem": stem, "dice": metrics["dice"], "iou": metrics["iou"], "hd95_px": metrics["hd95_px"]})
    summary = _summary(
        reference_index,
        prediction_index,
        paired_stems,
        rows,
        resampled,
        empty_reference,
        empty_prediction,
    )
    return summary, rows


def _close(left: float, right: float) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)


def self_check() -> None:
    identical = np.zeros((16, 16), dtype=bool)
    identical[4:12, 4:12] = True
    same = score_pair(identical, identical.copy())
    if not (_close(same["dice"], 1.0) and _close(same["iou"], 1.0) and _close(same["hd95_px"], 0.0)):
        raise SystemExit("identical masks did not score as exact overlap")

    empty = np.zeros((16, 16), dtype=bool)
    both_empty = score_pair(empty, empty.copy())
    if not (math.isnan(both_empty["dice"]) and math.isnan(both_empty["iou"]) and math.isnan(both_empty["hd95_px"])):
        raise SystemExit("two empty masks were scored")

    one_empty = score_pair(identical, empty)
    if not (_close(one_empty["dice"], 0.0) and _close(one_empty["iou"], 0.0) and math.isnan(one_empty["hd95_px"])):
        raise SystemExit("an empty prediction was scored incorrectly")

    reference = np.zeros((10, 10), dtype=bool)
    prediction = np.zeros((10, 10), dtype=bool)
    reference[0:5, 0:2] = True
    prediction[0:5, 1:3] = True
    overlap = score_pair(reference, prediction)
    if not (_close(overlap["dice"], 0.5) and _close(overlap["iou"], 5.0 / 15.0)):
        raise SystemExit("partial overlap Dice or IoU changed")

    left = np.zeros((8, 8), dtype=bool)
    right = np.zeros((8, 8), dtype=bool)
    left[0, 0] = True
    right[0, 3] = True
    if not _close(hd95(left, right), 3.0):
        raise SystemExit("single-pixel HD95 changed")

    # Radii 10 and 12, centers one column apart. Boundary-to-foreground HD95
    # is 2.828 px. Boundary-to-boundary HD95 on the same pair is 3 px.
    grid = np.ogrid[:64, :64]
    disk_a = (grid[0] - 32) ** 2 + (grid[1] - 32) ** 2 <= 10 ** 2
    disk_b = (grid[0] - 32) ** 2 + (grid[1] - 33) ** 2 <= 12 ** 2
    boundary_a = disk_a ^ ndimage.binary_erosion(disk_a, iterations=1, border_value=0)
    boundary_b = disk_b ^ ndimage.binary_erosion(disk_b, iterations=1, border_value=0)
    surface = np.concatenate(
        [
            ndimage.distance_transform_edt(~boundary_b)[boundary_a].ravel(),
            ndimage.distance_transform_edt(~boundary_a)[boundary_b].ravel(),
        ]
    )
    paper_hd95 = hd95(disk_a, disk_b)
    surface_hd95 = float(np.percentile(surface, 95))
    if not _close(paper_hd95, 2.8284271247461903):
        raise SystemExit("HD95 no longer matches the manuscript definition")
    if not _close(surface_hd95, 3.0):
        raise SystemExit("synthetic surface-distance reference changed")
    if _close(paper_hd95, surface_hd95):
        raise SystemExit("HD95 collapsed onto boundary-to-boundary distance")

    small = np.zeros((2, 2), dtype=bool)
    small[0, 0] = True
    large = np.zeros((4, 4), dtype=bool)
    aligned = align(large, small)
    expected = np.zeros((4, 4), dtype=bool)
    expected[0:2, 0:2] = True
    if aligned.shape != expected.shape or not np.array_equal(aligned, expected):
        raise SystemExit("nearest-neighbor alignment changed")
    resized = score_pair(expected, small)
    if not resized["resampled"] or not _close(resized["dice"], 1.0):
        raise SystemExit("resized overlap was not exact")

    stats = summarize([1.0, float("nan"), 3.0])
    if stats["n"] != 2 or not _close(stats["mean"], 2.0):
        raise SystemExit("summary kept a missing value")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        reference_dir = root / "reference"
        prediction_dir = root / "prediction"
        reference_dir.mkdir()
        prediction_dir.mkdir()
        Image.fromarray(identical.astype(np.uint8) * 255).save(reference_dir / "case.png")
        np.save(prediction_dir / "case.npy", identical)
        faint = np.zeros((16, 16), dtype=np.uint8)
        faint[0, 0] = 1
        Image.fromarray(faint).save(reference_dir / "faint.png")
        np.save(prediction_dir / "faint.npy", faint > 0)
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(prediction_dir / "extra.png")
        summary, rows = score_directories(reference_dir, prediction_dir)
        if summary["n_paired"] != 2 or summary["n_unmatched_prediction"] != 1:
            raise SystemExit("file pairing changed")
        by_stem = {row["file_stem"]: row for row in rows}
        if not _close(by_stem["case"]["dice"], 1.0) or not _close(by_stem["faint"]["dice"], 1.0):
            raise SystemExit("PNG or NPY binarization changed")
    print("self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score lung masks against a reference directory")
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--summary-json", type=Path, help="Write the identifier-free summary to this path")
    parser.add_argument(
        "--per-image-csv",
        type=Path,
        help="Optional per-file table. Keep it outside this repository when names identify radiographs.",
    )
    parser.add_argument("--self-check", action="store_true", help="Run the synthetic mask checks and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    if args.reference_dir is None or args.prediction_dir is None:
        raise SystemExit("Pass --reference-dir and --prediction-dir, or pass --self-check.")
    summary, rows = score_directories(args.reference_dir, args.prediction_dir)
    text = json.dumps(summary, indent=2)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.per_image_csv is not None:
        args.per_image_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.per_image_csv.open("w", encoding="utf-8", newline="") as handle:
            handle.write("file_stem,dice,iou,hd95_px\n")
            for row in rows:
                handle.write(f"{row['file_stem']},{row['dice']},{row['iou']},{row['hd95_px']}\n")


if __name__ == "__main__":
    main()
