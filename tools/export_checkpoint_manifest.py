#!/usr/bin/env python3
"""Create a portable SHA256 manifest for fine-tuned fold checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/trained_checkpoint_manifest.csv"))
    parser.add_argument("--file-name", default="best_model.pth")
    parser.add_argument(
        "--runs-file",
        type=Path,
        default=None,
        help="Optional one-run-name-per-line file; use results/primary/canonical_70runs.txt for the paper models.",
    )
    args = parser.parse_args()

    root = args.results_root.expanduser().resolve()
    if args.runs_file:
        run_names = [
            line.strip()
            for line in args.runs_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        run_dirs = [root / name for name in run_names]
    else:
        run_dirs = sorted(path for path in root.iterdir() if path.is_dir())

    rows = []
    missing = []
    for run_dir in run_dirs:
        checkpoints = sorted(run_dir.glob(f"fold_*/{args.file_name}"))
        if len(checkpoints) != 5:
            missing.append(f"{run_dir.name}: expected 5, found {len(checkpoints)}")
            continue
        split_assignment = read_json(run_dir / "split_assignment.json")
        for checkpoint in checkpoints:
            fold_dir = checkpoint.parent
            run_meta = read_json(run_dir / "run_meta.json")
            fold_results = read_json(fold_dir / "fold_results.json")
            run = run_meta.get("run", {})
            dataset = run_meta.get("dataset", {})
            config = fold_results.get("config", {})
            split_seed = run.get("split_seed", config.get("split_seed"))
            if split_seed is None:
                split_seed = split_assignment.get("seed", "")
            rows.append(
                {
                    "run_name": run_dir.name,
                    "fold": fold_dir.name.replace("fold_", ""),
                    "arch": run.get("arch", config.get("arch", "")),
                    "model_name": run.get("model_name", config.get("model_name", "")),
                    "view": dataset.get("view", config.get("view", "")),
                    "data_mode": config.get("data_mode", ""),
                    "training_seed": run.get("seed", config.get("seed", "")),
                    "split_seed": split_seed,
                    "relative_path": checkpoint.relative_to(root).as_posix(),
                    "size_bytes": checkpoint.stat().st_size,
                    "sha256": sha256_file(checkpoint),
                }
            )

    if missing:
        raise SystemExit("Incomplete checkpoint runs:\n" + "\n".join(missing))

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "fold",
        "arch",
        "model_name",
        "view",
        "data_mode",
        "training_seed",
        "split_seed",
        "relative_path",
        "size_bytes",
        "sha256",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} checkpoint records to {output}")


if __name__ == "__main__":
    main()
