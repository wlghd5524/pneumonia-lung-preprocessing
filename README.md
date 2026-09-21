# Reproducibility package: pneumonia classification with lung preprocessing

This repository reconstructs the study pipeline from cohort creation to lung
preprocessing, patient-level splitting, five-fold training, outer-test
evaluation, and Grad-CAM analysis.

This public package contains code and non-identifying reproducibility metadata.
Exact split identifiers and trained models are intentionally absent. PhysioNet
requires MIMIC-derived datasets and models to be shared under the same
credentialed agreement as MIMIC. See `DATA_GOVERNANCE.md` for the required
two-part release.

## What answers the reproducibility request

- Credentialed patient split placement and checksums: `splits/`
- Cohort construction: `data_preparation/build_pneumonia_json.py`
- Lung segmentation and crop preprocessing: `preprocessing/`
- Training, five-fold CV, and outer-test evaluation: `training/`
- Exact backbone IDs, pretrained revisions, and weight hashes: `configs/models.json`
- Exact study hyperparameters: `configs/training_config.json`
- Reference software versions: `configs/reference_environment.json`
- Grad-CAM and attribution code: `interpretability/`
- Bootstrap, calibration, uncertainty, and geometry analyses: `analysis/`
- Integrity hashes for all released text/code files: `CHECKSUMS.sha256`

## Repository layout

```text
data_preparation/     CheXpert label and AP/PA cohort construction
preprocessing/        MedSAM3/CheXMask masks and mask-based crops
training/             fixed patient splits, 5-fold CV, outer-test evaluation
analysis/             bootstrap CIs, paired comparisons, calibration, sensitivity
interpretability/     Grad-CAM, Figure 6, lung-attribution concentration
splits/               released AP and PA image/patient assignments
configs/              exact model, training, LoRA, and environment metadata
eva-x/                EVA-X loader used by the study
tools/                checkpoint manifest utility
checkpoints/          placement instructions; no large binaries in Git
```

## 1. Environment

Python 3.12 is recommended. Install all Python dependencies with:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU execution, the PyTorch wheel selected by pip must support the local
NVIDIA driver. If a CUDA-specific PyTorch index is required, install matching
`torch` and `torchvision` wheels first, then run the requirements command. The
reference versions are recorded in `configs/reference_environment.json`.

## 2. Restricted data and external weights

Obtain authorized access to:

1. MIMIC-CXR-JPG v2.1.0 and its CheXpert, metadata, and split CSV files.
2. The CheXMask MIMIC-CXR-JPG CSV.
3. SAM3/MedSAM3 and EVA-X weights under their respective licenses.
4. The study's MedSAM3 LoRA and fine-tuned fold checkpoints from the companion
   credentialed PhysioNet project.

Set portable paths instead of editing source files:

```bash
export MIMIC_CXR_ROOT=/path/to/mimic-cxr-jpg/2.1.0
export CHEXMASK_CSV=/path/to/ChexMask_MIMIC-CXR-JPG.csv
export DATA_BASE_DIR=/path/to/generated_datasets
export RESULTS_ROOT=/path/to/results_pneumonia
export RESULTS_DIR="$RESULTS_ROOT"
```

Place the LoRA and EVA-X files as described in `checkpoints/README.md`, then
verify them:

```bash
sha256sum checkpoints/best_lora_weights.pt
sha256sum eva-x/eva_x_base_patch16_merged520k_mim.pt
```

`requirements.txt` installs the exact SAM3 Git revision. The MedSAM3 LoRA
runtime module used by this pipeline is included as
`preprocessing/lora_layers.py`, so no separate Python environment or manual
source checkout is required.

## 3. Build the image-level cohort

Only CheXpert Pneumonia labels 0 and 1 are retained; uncertain or missing
labels are excluded from the primary cohort. Only AP and PA radiographs are
retained.

```bash
python data_preparation/build_pneumonia_json.py \
  --mimic-root "$MIMIC_CXR_ROOT" \
  --out-json pneumonia_labels.json
```

## 4. Generate lung masks

Run each view separately for MedSAM3:

```bash
python preprocessing/build_cxr_lung_seg_dataset.py \
  --seg-mode medsam3 --view AP \
  --labels-json pneumonia_labels.json \
  --source-image-root "$MIMIC_CXR_ROOT" \
  --config configs/medsam3_full_lora_config.yaml \
  --weights checkpoints/best_lora_weights.pt

python preprocessing/build_cxr_lung_seg_dataset.py \
  --seg-mode medsam3 --view PA \
  --labels-json pneumonia_labels.json \
  --source-image-root "$MIMIC_CXR_ROOT" \
  --config configs/medsam3_full_lora_config.yaml \
  --weights checkpoints/best_lora_weights.pt
```

Repeat with `--seg-mode chexmask --chexmask-csv "$CHEXMASK_CSV"` to generate
the CheXMask conditions.

## 5. Generate lung crops

```bash
python preprocessing/build_cxr_lung_crop_dataset.py --seg-mode medsam3 --view AP
python preprocessing/build_cxr_lung_crop_dataset.py --seg-mode medsam3 --view PA
python preprocessing/build_cxr_lung_crop_dataset.py \
  --seg-mode chexmask --view AP --chexmask-csv "$CHEXMASK_CSV"
python preprocessing/build_cxr_lung_crop_dataset.py \
  --seg-mode chexmask --view PA --chexmask-csv "$CHEXMASK_CSV"
```

The study defaults include two-component filtering, asymmetric crop margins,
mask-quality checking, center-crop fallback, and square padding. Their exact
values are the CLI defaults in the preprocessing script and are summarized in
the script help (`--help`).

## 6. Released patient splits

Download the exact files from the credentialed companion PhysioNet project and
place them under `splits/`. Their expected SHA256 values are in
`splits/README.md`.

The released CSVs are the authoritative study assignments:

- AP: 22,894 images from 11,951 subjects after excluding five images without
  usable CheXMask lung RLE (the pre-exclusion cohort had 11,953 subjects).
- PA: 18,609 images from 12,385 subjects.
- Split seed: 42.
- Outer test: approximately 15%, created by the first split of
  `StratifiedGroupKFold(n_splits=7)`.
- Remaining subjects: five-fold `StratifiedGroupKFold`.

Each CSV row contains `subject_id`, `dicom_id`, label, view, split, and fold.
The same subject never appears in both outer test and train/validation folds.

To regenerate rather than reuse the released CSVs:

```bash
python training/build_fixed_patient_splits.py \
  --data-base-dir "$DATA_BASE_DIR" \
  --split-dir splits \
  --split-seed 42 --n-folds 5 --outer-test-ratio 0.15

python preprocessing/apply_ap_cohort_exclusion.py --dry-run
python preprocessing/apply_ap_cohort_exclusion.py
python preprocessing/verify_preprocessing_sample_parity.py \
  --data-base-dir "$DATA_BASE_DIR" --split-dir splits
```

## 7. Train and evaluate

Example for one experiment:

```bash
python training/pneumonia_train.py \
  --arch resnet152 \
  --view AP \
  --data-mode raw \
  --data-root "$DATA_BASE_DIR/cxr_medsam3_lung_seg_ap" \
  --source-image-root "$MIMIC_CXR_ROOT" \
  --data-base-dir "$DATA_BASE_DIR" \
  --split-dir splits \
  --n-folds 5 --outer-test-ratio 0.15 \
  --split-seed 42 --seed 42 \
  --batch-size 64 --learning-rate 2e-5 \
  --bf16 --no-compile
```

Run the seven architecture keys in `configs/models.json`, both views, and all
five preprocessing conditions for the full experiment grid. The training
script writes fold-best checkpoints, OOF predictions, outer-test predictions,
AUROC, AUPRC, operating-point metrics, and run metadata.

The exact 70-run grid can be launched after installing `requirements.txt`:

```bash
bash training/run_all_base_backbones_sweep_h200.sh
```

For EVA-X, add:

```bash
--eva-x-ckpt-dir eva-x/eva_x_base_patch16_merged520k_mim.pt
```

RAD-DINO was run in float32; use `--fp32` instead of `--bf16`.

## 8. Statistical and interpretability analyses

```bash
python analysis/cluster_bootstrap_outer_test_ci.py --results-root "$RESULTS_ROOT"
python analysis/paired_cluster_bootstrap_delta.py --results-root "$RESULTS_ROOT"
python analysis/analyze_crossfit_operating_point.py
python analysis/plot_figure_s1_calibration.py
python analysis/analyze_input_geometry.py

python interpretability/generate_gradcam_pneumonia.py --results-root "$RESULTS_ROOT"
python interpretability/generate_figure6_gradcam.py
```

Use `python <script> --help` for script-specific inputs. Analyses use
subject-level cluster resampling where repeated images from one patient must
remain together.

## 9. Checkpoint release manifest

Large trained weights are MIMIC-derived models. Publish them in a versioned,
credentialed PhysioNet project—not GitHub or an unrestricted model archive.
Generate a SHA256 manifest before submission:

```bash
python tools/export_checkpoint_manifest.py \
  --results-root "$RESULTS_ROOT" \
  --runs-file configs/canonical_70runs.txt \
  --output checkpoints/trained_checkpoint_manifest.csv
```

Report the PhysioNet project DOI and version in the manuscript's Code and Data
Availability statement.
