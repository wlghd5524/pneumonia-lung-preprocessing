# Context-Dependent Effects of Lung Segmentation Preprocessing on Radiology-Report-Derived Pneumonia Classification in Chest Radiographs: A Comparative Study Across Acquisition Settings and Backbone Architectures

Reproducibility package for the revised manuscript.

| Item | Value |
| --- | --- |
| Release tag | `v1.0` (snapshot corresponding to the manuscript) |
| Zenodo DOI | Not issued yet. Add it here after the GitHub release is archived. |
| Python | 3.12 (reference run: 3.12.3) |
| Reference environment | `configs/reference_environment.json` and `requirements.txt` |
| GPU stack used for the reported runs | CUDA 13.0, cuDNN 9.15.0, PyTorch 2.10.0 (NVIDIA build) |

The reported training used one NVIDIA H200, batch size 64, and bfloat16
(RAD-DINO used float32). CPU-only execution can regenerate the tables from
saved predictions; it cannot repeat the training.

## What is and is not in this repository

| Material | Public repository |
| --- | --- |
| Training, preprocessing, and evaluation code | Yes |
| One JSON config per executed experiment (146 files) | Yes, under `configs/` |
| Aggregate and fold-level metrics | Yes, under `results/`. Numbered supplementary tables are not stored; the scripts that build them are in section 7. |
| Exact patient-partition CSV files | No. Checksums and the builder are in `splits/`. Credentialed users place the CSVs locally. See `DATA_GOVERNANCE.md`. |
| Image-level or patient-level prediction files | No. They contain MIMIC-CXR identifiers. Credentialed users can regenerate them. |
| Fold-specific trained checkpoints | Not currently redistributed. Architectures, initial weights, configs, seeds, and the checkpoint rule are in this repository and Supplementary Table S6. |

MIMIC-CXR-JPG v2.1.0 is available only to credentialed PhysioNet users:
https://physionet.org/content/mimic-cxr-jpg/2.1.0/

## Repository layout

```text
data_preparation/   CheXpert pneumonia labels and the AP/PA cohort
preprocessing/      MedSAM3 and CheXMask lung masks and crops
training/           patient-level 5-fold training and outer-test evaluation
analysis/           scripts from the previous public snapshot
evaluation/         bootstrap intervals, paired contrasts, calibration, sensitivity
mask_analysis/      input geometry, mask failure, segmentation-quality summaries
interpretability/   Grad-CAM and lung-attribution concentration
configs/            executed hyperparameters, one JSON per experiment
results/            released aggregate and fold-level tables
splits/             checksums and counts for the fixed patient partition
checkpoints/        where to place external weights; no trained binaries
eva-x/              EVA-X loader
tools/              checkpoint checksum manifest
```

## 1. Environment

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a CUDA build of `torch` and `torchvision` that matches the local
driver before that command if pip would otherwise select a CPU wheel.
Package versions pinned for the study are listed in
`configs/reference_environment.json`.

## 2. Data you must obtain separately

1. MIMIC-CXR-JPG v2.1.0, including the CheXpert labels, metadata, and the official split CSV.
2. The CheXMask MIMIC-CXR-JPG table.
3. SAM3 / MedSAM3 and EVA-X weights, under their own licenses.
4. The study's MedSAM3 LoRA file, placed at `checkpoints/best_lora_weights.pt` (checksum in `checkpoints/README.md`).
5. The four patient-partition CSVs, placed in `splits/` (checksums in `splits/README.md`).

```bash
export MIMIC_CXR_ROOT=/path/to/mimic-cxr-jpg/2.1.0
export CHEXMASK_CSV=/path/to/ChexMask_MIMIC-CXR-JPG.csv
export DATA_BASE_DIR=/path/to/generated_datasets
export RESULTS_ROOT=/path/to/results_pneumonia
export SPLIT_DIR=/path/to/splits
```

`DATA_BASE_DIR` is where generated mask and crop folders are written. It
defaults to the repository root. `RESULTS_ROOT` (also read as `RESULTS_DIR`
by the analysis scripts) is where training and tables are written. It
defaults to `results_pneumonia/` inside the repository. The released tables
already in `results/` are separate from that working directory.

## 3. Cohort

The primary cohort keeps CheXpert Pneumonia labels 0 and 1 and drops
uncertain or missing labels. Only AP and PA radiographs are kept.

```bash
python data_preparation/build_pneumonia_json.py \
  --mimic-root "$MIMIC_CXR_ROOT" \
  --out-json "$DATA_BASE_DIR/pneumonia_labels.json"
```

Released counts after the AP exclusion below: AP 22,894 images from 11,951
subjects; PA 18,609 images from 12,385 subjects.

## 4. Preprocessing

Lung masks, then lung crops. Run AP and PA separately.

```bash
python preprocessing/build_cxr_lung_seg_dataset.py \
  --seg-mode medsam3 --view AP \
  --labels-json "$DATA_BASE_DIR/pneumonia_labels.json" \
  --source-image-root "$MIMIC_CXR_ROOT" \
  --config configs/medsam3_full_lora_config.yaml \
  --weights checkpoints/best_lora_weights.pt

python preprocessing/build_cxr_lung_seg_dataset.py \
  --seg-mode medsam3 --view PA \
  --labels-json "$DATA_BASE_DIR/pneumonia_labels.json" \
  --source-image-root "$MIMIC_CXR_ROOT" \
  --config configs/medsam3_full_lora_config.yaml \
  --weights checkpoints/best_lora_weights.pt
```

Repeat with `--seg-mode chexmask --chexmask-csv "$CHEXMASK_CSV"`.

```bash
python preprocessing/build_cxr_lung_crop_dataset.py --seg-mode medsam3 --view AP
python preprocessing/build_cxr_lung_crop_dataset.py --seg-mode medsam3 --view PA
python preprocessing/build_cxr_lung_crop_dataset.py --seg-mode chexmask --view AP --chexmask-csv "$CHEXMASK_CSV"
python preprocessing/build_cxr_lung_crop_dataset.py --seg-mode chexmask --view PA --chexmask-csv "$CHEXMASK_CSV"
```

`bash preprocessing/run_rebuild_lung_seg_crop_datasets.sh` runs the same eight
jobs. Crop defaults (component filter, margins, quality check, center-crop
fallback, square padding) are the script's CLI defaults; `python
preprocessing/build_cxr_lung_crop_dataset.py --help` prints them.

Five AP images had no usable CheXMask lung RLE and were removed so that every
preprocessing shared the same images:

```bash
python preprocessing/apply_ap_cohort_exclusion.py --dry-run
python preprocessing/apply_ap_cohort_exclusion.py
python preprocessing/verify_preprocessing_sample_parity.py \
  --data-base-dir "$DATA_BASE_DIR" --split-dir "$SPLIT_DIR"
```

## 5. Patient partition

The study split is by `subject_id`, not by image.

- Split seed 42.
- Outer test: first split of `StratifiedGroupKFold(n_splits=7)`, about 15% of subjects.
- The remaining subjects: five-fold `StratifiedGroupKFold`.
- One subject is never in both the outer test and a training fold.

The authoritative files are:

```text
splits/AP_outer_test.csv
splits/AP_fold_assignment.csv
splits/PA_outer_test.csv
splits/PA_fold_assignment.csv
```

Columns: `subject_id`, `dicom_id`, `label`, `view`, `split`, `fold`.
These CSVs are not in the public repository. Their SHA256 values are in
`splits/README.md`. Place the credentialed copies in `splits/` before
training. Do not rely on re-running the split with seed 42 alone: a different
scikit-learn build can shuffle groups differently.

To rebuild the CSVs from an authorized cohort and then check them:

```bash
python training/build_fixed_patient_splits.py \
  --data-base-dir "$DATA_BASE_DIR" \
  --split-dir "$SPLIT_DIR" \
  --split-seed 42 --n-folds 5 --outer-test-ratio 0.15

sha256sum "$SPLIT_DIR"/{AP,PA}_{outer_test,fold_assignment}.csv
```

Compare the hashes with `splits/README.md`.

## 6. Training

Executed hyperparameters for every run are in `configs/`. The primary grid
is 2 views × 5 preprocessing conditions × 7 backbones = 70 files:

```text
configs/ap/raw_resnet152.json
configs/ap/chexmask_crop_rad_dino.json
configs/pa/...
```

Additional files cover the preprocessing controls (`configs/controls/`),
training seeds 123 and 2026 (`configs/seeds/`), and the uncertain-label
training policies (`configs/uncertainty_training_policy/`).

Shared values: input 512, learning rate 2e-5, weight decay 0.05, batch size
64, dropout 0.2, 50 epochs, cosine schedule, 3 warmup epochs, weighted BCE,
early stopping on validation AUROC (patience 8; 9 for EVA-X and RAD-DINO),
decision threshold from Youden J on validation predictions. The full executed
record, including augmentation, is inside each JSON.

One experiment:

```bash
python training/pneumonia_train.py \
  --config configs/ap/raw_resnet152.json \
  --data-root "$DATA_BASE_DIR/cxr_medsam3_lung_seg_ap" \
  --data-base-dir "$DATA_BASE_DIR" \
  --split-dir "$SPLIT_DIR" \
  --source-image-root "$MIMIC_CXR_ROOT" \
  --split-mode cv5 \
  --n-folds 5 \
  --outer-test-ratio 0.15 \
  --bf16 \
  --no-compile
```

The full 70-run grid, as launched for the study:

```bash
bash training/run_all_base_backbones_sweep_h200.sh
```

Controls, seed repeats, and training-policy repeats:

```bash
bash training/run_preprocessing_control_sweep_h200.sh
bash training/run_seed_variability_sweep_h200.sh
bash training/run_uncertainty_training_policy_sweep_h200.sh
```

`DRY_RUN=1` prints the commands. `MAX_FOLDS=1` trains only the first fold.

## 7. Evaluation

```bash
python evaluation/cluster_bootstrap_outer_test_ci.py --results-root "$RESULTS_ROOT"
python evaluation/paired_cluster_bootstrap_delta.py --results-root "$RESULTS_ROOT"
python evaluation/analyze_crossfit_operating_point.py
python evaluation/extract_fold_vs_raw_table.py
python evaluation/evaluate_uncertainty_endpoint_sensitivity.py
python evaluation/summarize_training_seed_robustness.py
python evaluation/evaluate_training_policy_sensitivity.py
python mask_analysis/analyze_input_geometry.py
```

Bootstrap resampling keeps all images from one subject together
(`n_boot=2000`, seed 42). `python <script> --help` lists inputs.

### Which script builds which table or figure

| Reported item | Script |
| --- | --- |
| Table S1, fold-level AUROC/AUPRC | `evaluation/extract_fold_vs_raw_table.py` |
| Table S3, mask failure and agreement | `mask_analysis/summarize_mask_source_failure_table.py`, `mask_analysis/analyze_segmentation_quality_proxies.py` |
| Table S4, input geometry | `mask_analysis/analyze_input_geometry.py` |
| Table S5, operating point and calibration | `evaluation/analyze_crossfit_operating_point.py` |
| Table S6, model and software record | `configs/models.json`, `configs/reference_environment.json`, `checkpoints/trained_checkpoint_manifest.csv` |
| Table S9 and Table 6, preprocessing controls | `evaluation/paired_cluster_bootstrap_delta.py` on the control runs |
| Table S10 and Table 7, training seeds | `evaluation/summarize_training_seed_robustness.py` |
| Uncertainty-endpoint sensitivity | `evaluation/evaluate_uncertainty_endpoint_sensitivity.py` |
| Training-policy sensitivity | `evaluation/evaluate_training_policy_sensitivity.py` |
| Grad-CAM figures | `interpretability/generate_gradcam_pneumonia.py`, `interpretability/generate_figure6_gradcam.py` |
| Lung-attribution table | `interpretability/analyze_lung_attribution_concentration.py` |

`results/` holds the aggregate numbers. It does not hold the numbered
supplementary tables; those are produced by the scripts above. Per-image
expert tables and Grad-CAM figures identify individual radiographs and are
not included there.

## 8. Checkpoints and predictions

**Trained model checkpoints.** The fold-specific trained checkpoints used in
this study are not redistributed through the public repository. All model
architectures, pretrained initialization sources, training configurations,
patient-partition procedures, random seeds, checkpoint-selection criteria,
and evaluation code required to reproduce the trained models are provided in
the repository. The model identifiers, software versions, and checkpoint
checksums are in `configs/models.json`, `configs/reference_environment.json`,
and `checkpoints/trained_checkpoint_manifest.csv`.

**Evaluation outputs.** Aggregate and fold-level evaluation results, including
discrimination metrics, operating-point metrics, calibration metrics,
preprocessing contrasts, and sensitivity analyses, are provided in
`results/`. The numbered supplementary tables are not included as files;
section 7 lists the scripts that rebuild them. Patient-level or
image-level prediction files containing MIMIC-CXR identifiers are not
redistributed. Credentialed users of MIMIC-CXR-JPG can regenerate these
outputs using the released evaluation scripts and fixed patient-partition
files.

A SHA256 list of the 350 fold checkpoints (70 runs × 5 folds), without the
weight files, is in `checkpoints/trained_checkpoint_manifest.csv`.
`CHECKSUMS.sha256` is the checksum list from the previous public snapshot.
`configs/training_config.json` is the shared hyperparameter record from that
same snapshot. The per-run files under `configs/ap/` and `configs/pa/` are
the executed configurations.
