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

BACKBONE_LABELS = {
    "resnet152": "ResNet-152",
    "densenet201": "DenseNet-201",
    "convnextv2_base": "ConvNeXt V2-Base",
    "swinv2_base": "Swin V2-Base",
    "dinov3_base": "DINOv3-Base",
    "eva_x_base": "EVA-X-Base",
    "rad_dino": "RAD-DINO",
}

ARCH_CONFIG = {
    "resnet152": "configs/ap/raw_resnet152.json",
    "densenet201": "configs/ap/raw_densenet201.json",
    "convnextv2_base": "configs/ap/raw_convnextv2_base.json",
    "swinv2_base": "configs/ap/raw_swinv2_base.json",
    "dinov3_base": "configs/ap/raw_dinov3_base.json",
    "eva_x_base": "configs/ap/raw_eva_x_base.json",
    "rad_dino": "configs/ap/raw_rad_dino.json",
}


def lr_text(value: float) -> str:
    mantissa, exponent = f"{value:.0e}".split("e")
    superscript = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{mantissa} × 10{str(int(exponent)).translate(superscript)}"


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
    cfg = json.loads((ROOT / ARCH_CONFIG["resnet152"]).read_text(encoding="utf-8"))
    pkgs = env["packages"]
    head = models["classifier_head"]
    infer = seg["inference"]
    runtime = infer["runtime"]
    lines = [
        "# Table S6. Model, Checkpoint, and Software Reproducibility Details",
        "",
        "## A. Classification training configuration",
        "",
        "| Parameter | Setting |",
        "| --- | --- |",
        f"| Input resolution | {cfg['input_size']} × {cfg['input_size']} |",
        f"| Maximum epochs | {cfg['epochs']} |",
        f"| Batch size | {cfg['batch_size']} |",
        f"| Learning rate | {lr_text(cfg['learning_rate'])} |",
        f"| Weight decay | {cfg['weight_decay']} |",
        "| Optimizer | AdamW |",
        f"| Dropout | {cfg['dropout']} |",
        "| Loss | Weighted BCE |",
        "| LR scheduler | Cosine annealing |",
        f"| Warm-up | {cfg['warmup_epochs']} epochs |",
        f"| Training seed | {cfg['seed']} |",
        f"| Split seed | {cfg['split_seed']} |",
        "",
        "## B. Backbones",
        "",
        "All seven backbones used 512×512 inputs and were fully fine-tuned. Best checkpoints were selected by "
        "validation AUROC.",
        "",
        "| Backbone | Pretrained checkpoint/source | Classifier head |",
        "| --- | --- | --- |",
    ]
    for key, label in BACKBONE_LABELS.items():
        record = models["classification_backbones"][key]
        if key == "eva_x_base":
            source = record["weight_file"]
        elif key == "rad_dino":
            source = (
                f"Hugging Face {record['pretrained_repository']}; cached snapshot used in this study: "
                f"{record['commit']}; model.safetensors SHA-256: {record['weight_sha256']}"
            )
        else:
            source = record["pretrained_repository"]
        lines.append(f"| {label} | {source} | {head} |")
    lines.extend([
        "",
        "For RAD-DINO, the model was loaded using AutoModel.from_pretrained(\"microsoft/rad-dino\") without an "
        "explicitly pinned revision; the snapshot identifier and SHA-256 hash reported above identify the locally "
        "cached public checkpoint used for the experiments reported in this study.",
        "",
        "## C. Lung segmentation inference",
        "",
        "| Item | Recorded value |",
        "| --- | --- |",
        f"| MedSAM3 base | {seg['base_model']['huggingface_repository']}; revision {seg['base_model']['huggingface_commit']} |",
        f"| SAM3 checkpoint | {seg['base_model']['weight_file']}; SHA-256 unavailable on this classification host |",
        "| LoRA weights | MedSAM3/weights/best_lora_weights.pt; SHA-256 unavailable on this classification host |",
        f"| LoRA configuration | rank {infer['lora_rank']}; alpha {infer['lora_alpha']}; dropout "
        f"{infer['lora_dropout_at_inference']} during inference |",
        f"| Repository commits | MedSAM3 {seg['medsam3_code']['git_commit']}; SAM3 {seg['base_model']['git_commit']} |",
        f"| MedSAM3 input / prompt | {infer['input_size']}×{infer['input_size']}; channel-wise mean and SD "
        f"{infer['normalization_mean']}; text prompt {infer['text_prompt']} |",
        f"| Detection / NMS | confidence >{infer['detection_confidence_threshold']}; NMS IoU {infer['nms_iou']} |",
        "| Mask decoding | sigmoid and threshold 0.5; bilinear native-size resize; threshold 0.5 again; logical union |",
        f"| MedSAM3 inference runtime | Python {runtime['python']}; PyTorch {runtime['torch']}; torchvision "
        f"{runtime['torchvision']}; CUDA {runtime['cuda']}; {runtime['gpu']} |",
        f"| CheXMask | {seg['chexmask']} |",
        "",
        "## D. Classification software/runtime",
        "",
        "| Component | Version/details |",
        "| --- | --- |",
        f"| Python | {env['python']} |",
        f"| PyTorch | {pkgs['torch']} |",
        f"| torchvision | {pkgs['torchvision']} |",
        f"| timm | {pkgs['timm']} |",
        f"| transformers | {pkgs['transformers']} |",
        f"| NumPy | {pkgs['numpy']} |",
        f"| SciPy | {pkgs['scipy']} |",
        f"| scikit-learn | {pkgs['scikit-learn']} |",
        f"| Pillow | {pkgs['Pillow']} |",
        f"| CUDA reported by PyTorch | {env['cuda_runtime']} |",
        f"| Classification GPU | {env['gpu']} |",
        "",
        "## Additional record in this repository",
        "",
        "These entries are not in the manuscript table.",
        "",
        f"- The MedSAM3 LoRA file is the upstream public release {seg['lora_checkpoint']['huggingface_repository']} "
        f"at revision {seg['lora_checkpoint']['huggingface_revision']}; the file downloaded from that revision has "
        f"SHA-256 {seg['lora_checkpoint']['sha256']} (`checkpoints/README.md`).",
        f"- cuDNN {env['cudnn']}. Hugging Face commits and SHA-256 values of every initialization source are in "
        "`Table_S6_model_checkpoint_software.csv` and `configs/models.json`.",
        f"- The checkpoint manifest lists {n_checkpoints} fold files and stores sizes and SHA256 values only. "
        "The weight files are not in this repository. Retraining uses the configs, seeds, and initialization "
        "sources above; the resulting weights are not expected to match those checksums exactly.",
        "",
    ])
    DEST_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DEST_CSV}")
    print(f"wrote {DEST_MD}")


if __name__ == "__main__":
    main()
