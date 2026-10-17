# Table S6. Model, Checkpoint, and Software Reproducibility Details

## A. Classification training configuration

| Parameter | Setting |
| --- | --- |
| Input resolution | 512 × 512 |
| Maximum epochs | 50 |
| Batch size | 64 |
| Learning rate | 2 × 10⁻⁵ |
| Weight decay | 0.05 |
| Optimizer | AdamW |
| Dropout | 0.2 |
| Loss | Weighted BCE |
| LR scheduler | Cosine annealing |
| Warm-up | 3 epochs |
| Training seed | 42 |
| Split seed | 42 |

## B. Backbones

All seven backbones used 512×512 inputs and were fully fine-tuned. Best checkpoints were selected by validation AUROC.

| Backbone | Pretrained checkpoint/source | Classifier head |
| --- | --- | --- |
| ResNet-152 | timm/resnet152.a1h_in1k | adaptive GAP → LayerNorm → Dropout(0.2) → Linear(C,256) → GELU → LayerNorm → Dropout(0.2) → Linear(256,2) |
| DenseNet-201 | timm/densenet201.tv_in1k | adaptive GAP → LayerNorm → Dropout(0.2) → Linear(C,256) → GELU → LayerNorm → Dropout(0.2) → Linear(256,2) |
| ConvNeXt V2-Base | timm/convnextv2_base.fcmae_ft_in22k_in1k | adaptive GAP → LayerNorm → Dropout(0.2) → Linear(C,256) → GELU → LayerNorm → Dropout(0.2) → Linear(256,2) |
| Swin V2-Base | timm/swinv2_base_window16_256.ms_in1k | adaptive GAP → LayerNorm → Dropout(0.2) → Linear(C,256) → GELU → LayerNorm → Dropout(0.2) → Linear(256,2) |
| DINOv3-Base | timm/vit_base_patch16_dinov3.lvd1689m | adaptive GAP → LayerNorm → Dropout(0.2) → Linear(C,256) → GELU → LayerNorm → Dropout(0.2) → Linear(256,2) |
| EVA-X-Base | eva_x_base_patch16_merged520k_mim.pt | adaptive GAP → LayerNorm → Dropout(0.2) → Linear(C,256) → GELU → LayerNorm → Dropout(0.2) → Linear(256,2) |
| RAD-DINO | Hugging Face microsoft/rad-dino; cached snapshot used in this study: 110cbc18d5133582e320b43d53bf5c44e410c936; model.safetensors SHA-256: dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae | adaptive GAP → LayerNorm → Dropout(0.2) → Linear(C,256) → GELU → LayerNorm → Dropout(0.2) → Linear(256,2) |

For RAD-DINO, the model was loaded using AutoModel.from_pretrained("microsoft/rad-dino") without an explicitly pinned revision; the snapshot identifier and SHA-256 hash reported above identify the locally cached public checkpoint used for the experiments reported in this study.

## C. Lung segmentation inference

| Item | Recorded value |
| --- | --- |
| MedSAM3 base | facebook/sam3; revision 3c879f39826c281e95690f02c7821c4de09afae7 |
| SAM3 checkpoint | sam3.pt; SHA-256 unavailable on this classification host |
| LoRA weights | MedSAM3/weights/best_lora_weights.pt; SHA-256 unavailable on this classification host |
| LoRA configuration | rank 16; alpha 32; dropout 0.0 during inference |
| Repository commits | MedSAM3 f79eef3f4ccfc880e63a0a5a758153137e75d34b; SAM3 f6e51f59500a87c576c2df2323ce56b9fd7a12de |
| MedSAM3 input / prompt | 1008×1008; channel-wise mean and SD [0.5, 0.5, 0.5]; text prompt lung |
| Detection / NMS | confidence >0.5; NMS IoU 0.5 |
| Mask decoding | sigmoid and threshold 0.5; bilinear native-size resize; threshold 0.5 again; logical union |
| MedSAM3 inference runtime | Python 3.11.14; PyTorch 2.10.0+cu128; torchvision 0.25.0+cu128; CUDA 12.8; NVIDIA GeForce RTX 4090 |
| CheXMask | Stored PhysioNet CheXMask left- and right-lung RLE masks combined by logical OR; nearest-neighbor resize when required |

## D. Classification software/runtime

| Component | Version/details |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.10.0a0+b558c986e8.nv25.11 |
| torchvision | 0.25.0a0+7a13ad0f |
| timm | 1.0.22 |
| transformers | 5.3.0 |
| NumPy | 1.26.4 |
| SciPy | 1.16.3 |
| scikit-learn | 1.7.2 |
| Pillow | 12.0.0 |
| CUDA reported by PyTorch | 13.0 |
| Classification GPU | NVIDIA H200 |

## Additional record in this repository

These entries are not in the manuscript table.

- The MedSAM3 LoRA file is the upstream public release lal-Joey/MedSAM3_v1 at revision f579e37e4db9f2c7e85a133b7970d59907e36fec; the file downloaded from that revision has SHA-256 499e638bb7c51dbe0dcc3bfb9dbfada74fc2d725e953fbb5bdb2dd1b72106f91 (`checkpoints/README.md`).
- cuDNN 9.15.0. Hugging Face commits and SHA-256 values of every initialization source are in `Table_S6_model_checkpoint_software.csv` and `configs/models.json`.
- The checkpoint manifest lists 350 fold files and stores sizes and SHA256 values only. The weight files are not in this repository. Retraining uses the configs, seeds, and initialization sources above; the resulting weights are not expected to match those checksums exactly.
