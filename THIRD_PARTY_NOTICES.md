# Third-party code and weights

The MIT License in `LICENSE` covers the code and documents written for this
study. It does not cover the items below, and it does not cover data.

## Code included in this repository

| File | Origin | License |
| --- | --- | --- |
| `eva-x/eva_x.py` | EVA-X backbone, https://github.com/hustvl/EVA-X | Apache-2.0 (the repository lists Apache-2.0 and a `LICENSE.txt`; its README badge says MIT) |
| `preprocessing/lora_layers.py` | Copied unchanged from https://github.com/Joey-S-Liu/MedSAM3 at commit `f79eef3f4ccfc880e63a0a5a758153137e75d34b` (SHA256 in `configs/models.json`) | The MedSAM3 repository declares no license. Its README says it is built on https://github.com/Sompote/SAM3_LoRA, which declares Apache-2.0 |

These two files keep the licenses of their origins and are not relicensed
under MIT. Check the upstream terms before reusing them. For
`preprocessing/lora_layers.py` no license grant could be confirmed, so reuse
outside running this study's code is at your own risk until the upstream
authors state a license.

## Weights and models that are not included

These are downloaded separately and keep their own terms: SAM3
(`facebook/sam3`, SAM License), the MedSAM3-v1 LoRA
(`lal-Joey/MedSAM3_v1`, Apache-2.0 per its model card), the EVA-X weights, and
the timm and Hugging Face pretrained backbones named in `configs/models.json`.
Details and checksums are in `checkpoints/README.md`.

## Data

MIMIC-CXR-JPG, the CheXpert labels, the CheXMask table, and the PhysioNet
heart and lung segmentation dataset are not included and are governed by their
own data use agreements. See `DATA_GOVERNANCE.md`.
