# Supplementary Table S6. Model, checkpoint, and software record

Python 3.12.3; CUDA 13.0; cuDNN 9.15.0; PyTorch 2.10.0a0+b558c986e8.nv25.11.
The checkpoint manifest lists 350 fold files and stores sizes and SHA256 values only. The weight files are not in this repository. Retraining uses the configs, seeds, and initialization sources above; the resulting weights are not expected to match those checksums exactly.

| Backbone | Model | Source | Pin | Precision | Checkpoint rule |
| --- | --- | --- | --- | --- | --- |
| resnet152 | resnet152 | timm/resnet152.a1h_in1k | 394d5cec41d60cbe669afd3e09cee61693582595 | bfloat16 | best validation AUROC |
| densenet201 | densenet201 | timm/densenet201.tv_in1k | 8d2e0f14ef7363343a1a15d6535db01e161ab3a9 | bfloat16 | best validation AUROC |
| convnextv2_base | convnextv2_base | timm/convnextv2_base.fcmae_ft_in22k_in1k | 9e250ed8f88b436472d2b24af26f82a8aa8c719d | bfloat16 | best validation AUROC |
| swinv2_base | swinv2_base_window16_256 | timm/swinv2_base_window16_256.ms_in1k | 9e2bc70d011c17e8ae6720bf731990a8970c9f0e | bfloat16 | best validation AUROC |
| dinov3_base | vit_base_patch16_dinov3 | timm/vit_base_patch16_dinov3.lvd1689m | c6a5fb7d12bbd3cf3b0079253141c3332aaed7da | bfloat16 | best validation AUROC |
| eva_x_base | eva_x_base | MapleF/eva_x | 35dc7012c1a49cba01f5a31a8d777db39258078c85cfad3f3c9a41deb172e1b4 | bfloat16 | best validation AUROC |
| rad_dino | microsoft/rad-dino | microsoft/rad-dino | 110cbc18d5133582e320b43d53bf5c44e410c936 | float32 | best validation AUROC |
| sam3 | sam3 | facebook/sam3 | 3c879f39826c281e95690f02c7821c4de09afae7 |  | base weights; the MedSAM3-v1 LoRA is the upstream public release (lal-Joey/MedSAM3_v1), used unmodified, not retrained here |
