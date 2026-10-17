# Checkpoint placement

Large weight files are intentionally not committed to Git.

## MedSAM3 LoRA (`checkpoints/best_lora_weights.pt`)

This file is the public MedSAM3-v1 LoRA release by its authors
(https://huggingface.co/lal-Joey/MedSAM3_v1, code
https://github.com/Joey-S-Liu/MedSAM3). We used it unmodified. We did not
train or fine-tune it, on MIMIC data or on anything else, so there is no
training script for it in this repository. The upstream card describes its
training data as 658,094 images, 2,863,974 instance annotations and 330 text
concepts across CT, MRI, PET, X-ray, microscopy, ultrasound, endoscopy and
other modalities. It does not list the source datasets, so we cannot state
whether MIMIC-CXR images were among them. The reported lung-mask quality rests
on the expert-contour comparison in `cxr-segmentation-validation/`, not on
any claim about upstream training data.

Download and verify:

```bash
hf download lal-Joey/MedSAM3_v1 best_lora_weights.pt \
  --revision f579e37e4db9f2c7e85a133b7970d59907e36fec --local-dir checkpoints
echo "499e638bb7c51dbe0dcc3bfb9dbfada74fc2d725e953fbb5bdb2dd1b72106f91  checkpoints/best_lora_weights.pt" | sha256sum -c
```

Expected SHA256:

`499e638bb7c51dbe0dcc3bfb9dbfada74fc2d725e953fbb5bdb2dd1b72106f91`

The base SAM3 weights (`facebook/sam3`, revision in `configs/models.json`) are
separate and are also obtained from their own sources.
`configs/medsam3_full_lora_config.yaml` is upstream's config with local paths
changed. Its training section does not describe anything we ran.

Place the EVA-X weights at:

`eva-x/eva_x_base_patch16_merged520k_mim.pt`

Expected SHA256:

`35dc7012c1a49cba01f5a31a8d777db39258078c85cfad3f3c9a41deb172e1b4`

Fine-tuned classifier checkpoints retain the training output layout:

`<results_root>/<run_name>/fold_<1-5>/best_model.pth`

`trained_checkpoint_manifest.csv` contains the file size and SHA256 for all
350 paper checkpoints (70 canonical runs × 5 folds). Its own SHA256 is:

`1997eb174087c69df2d70db32055f3f3dc5962c100bc4928b3dd38c177a2cd6b`

The fold-specific trained checkpoints are not redistributed through this
public repository, because PhysioNet asks that models derived from MIMIC be
shared on PhysioNet under the source data agreement (see `DATA_GOVERNANCE.md`).
`trained_checkpoint_manifest.csv` records sizes and checksums only. Retraining from the released configs and seeds can repeat the
experiment, but the new weight files will not match these checksums exactly,
and Grad-CAM figures drawn from the retrained weights will not match Figure 6. Regenerate that list from a local results folder with:

```bash
python tools/export_checkpoint_manifest.py \
  --results-root /path/to/results_pneumonia \
  --runs-file results/primary/canonical_70runs.txt \
  --output checkpoints/trained_checkpoint_manifest.csv
```
