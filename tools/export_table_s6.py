#!/usr/bin/env python3
"""Assemble Supplementary Table S6 from the public model and environment records.

Fold weight files are not read. The checkpoint manifest contributes its row
count only.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "configs" / "models.json"
ENV = ROOT / "configs" / "reference_environment.json"
MANIFEST = ROOT / "checkpoints" / "trained_checkpoint_manifest.csv"
DEST_CSV = ROOT / "results" / "supplement" / "Table_S6_model_checkpoint_software.csv"
DEST_MD = ROOT / "results" / "supplement" / "Table_S6_model_checkpoint_software.md"

ARCH_CONFIG = {
    "resnet152": "configs/ap/raw_resnet152.json",
    "densenet201": "configs/ap/raw_densenet201.json",
    "convnextv2_base": "configs/ap/raw_convnextv2_base.json",
    "swinv2_base": "configs/ap/raw_swinv2_base.json",
    "dinov3_base": "configs/ap/raw_dinov3_base.json",
    "eva_x_base": "configs/ap/raw_eva_x_base.json",
    "rad_dino": "configs/ap/raw_rad_dino.json",
}


def main() -> None:
    models = json.loads(MODELS.read_text(encoding="utf-8"))
    env = json.loads(ENV.read_text(encoding="utf-8"))
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        n_checkpoints = sum(1 for _ in csv.DictReader(handle))
    rows = []
    for key, record in models["classification_backbones"].items():
        cfg_path = ROOT / ARCH_CONFIG[key]
        if not cfg_path.is_file():
            raise FileNotFoundError(cfg_path)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        source = record.get("pretrained_repository", "")
        pin = record.get("commit") or record.get("sha256") or ""
        rows.append({
            "backbone": key,
            "model_name": record["model_name"],
            "source": source,
            "pin": pin,
            "weight_file": record.get("weight_file", ""),
            "input_size": cfg.get("input_size", ""),
            "training_seed": cfg.get("seed", ""),
            "precision": "float32" if key == "rad_dino" else "bfloat16",
            "checkpoint_rule": "best validation AUROC",
        })
    seg = models["lung_segmentation"]
    rows.append({
        "backbone": "sam3",
        "model_name": "sam3",
        "source": seg["base_model"]["huggingface_repository"],
        "pin": seg["base_model"]["huggingface_commit"],
        "weight_file": seg["base_model"]["weight_file"],
        "input_size": "",
        "training_seed": "",
        "precision": "",
        "checkpoint_rule": "base weights; the MedSAM3-v1 LoRA is the upstream public release (lal-Joey/MedSAM3_v1), used unmodified, not retrained here",
    })
    DEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with DEST_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Supplementary Table S6. Model, checkpoint, and software record",
        "",
        f"Python {env['python']}; CUDA {env['cuda_runtime']}; cuDNN {env['cudnn']}; "
        f"PyTorch {env['packages']['torch']}.",
        f"The checkpoint manifest lists {n_checkpoints} fold files and stores sizes and SHA256 values only. "
        "The weight files are not in this repository. Retraining uses the configs, seeds, and "
        "initialization sources above; the resulting weights are not expected to match those checksums exactly.",
        "",
        "| Backbone | Model | Source | Pin | Precision | Checkpoint rule |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['backbone']} | {row['model_name']} | {row['source']} | {row['pin']} | "
            f"{row['precision']} | {row['checkpoint_rule']} |"
        )
    lines.append("")
    DEST_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DEST_CSV}")
    print(f"wrote {DEST_MD}")


if __name__ == "__main__":
    main()
