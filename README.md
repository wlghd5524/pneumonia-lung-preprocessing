# Context-Dependent Effects of Lung Segmentation Preprocessing on Radiology-Report-Derived Pneumonia Classification in Chest Radiographs: A Comparative Study Across Acquisition Settings and Backbone Architectures

Reproducibility package for the manuscript.

| Item | Value |
| --- | --- |
| Release tag | `v1.1` (snapshot corresponding to the manuscript) |
| DOI | None. This release is identified by its Git tag and commit. |
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
| Aggregate metrics and identifier-free manuscript tables | Yes, under `results/`: Tables 3 to 7 and Supplementary Tables S1 to S11. Table 4 uses the 20,000-replicate paired intervals. Per-image tables are not. |
| Exact patient-partition CSV files | No. They hold MIMIC identifiers. The builder and SHA256 checksums are in `splits/` and `training/`; credentialed users generate the CSVs locally and verify them (Section 5). |
| Image-level or patient-level prediction files | No. They contain MIMIC-CXR identifiers. Credentialed users can regenerate them. |
| Expert lung masks and per-image expert tables | No. The annotation procedure and Dice, IoU, and HD95 code are in `cxr-segmentation-validation/`. The 100 AP expert masks are provided on request only to credentialed PhysioNet users of MIMIC-CXR-JPG (`DATA_GOVERNANCE.md`). |
| Fold-specific trained checkpoints | No. PhysioNet's policy treats models derived from MIMIC data as sensitive and asks that they be shared on PhysioNet under the source data agreement, so they are not placed on GitHub (`DATA_GOVERNANCE.md`). Architectures, initial weights, configs, seeds, and the checkpoint rule are in this repository and Supplementary Table S6. |
| MedSAM3 LoRA (`best_lora_weights.pt`) | Not a study checkpoint. It is the authors' public MedSAM3-v1 release; the download command is in Section 2. |

MIMIC-CXR-JPG v2.1.0 is available only to credentialed PhysioNet users:
https://physionet.org/content/mimic-cxr-jpg/2.1.0/

## License

The code and documents written for this study are released under the MIT
License (`LICENSE`). `eva-x/eva_x.py` and `preprocessing/lora_layers.py` come
from other projects and keep their own terms, and model weights and data are
not covered. See `THIRD_PARTY_NOTICES.md`.

## Repository layout

```text
data_preparation/   CheXpert pneumonia labels and the AP/PA cohort
preprocessing/      MedSAM3 and CheXMask lung masks and crops
training/           patient-level 5-fold training and outer-test evaluation
analysis/           scripts from the previous public snapshot
figures/            Figure 5 and Supplementary Figure S1 drawing code
evaluation/         bootstrap intervals, paired contrasts, calibration, sensitivity
mask_analysis/      input geometry, mask failure, segmentation-quality summaries
cxr-segmentation-validation/  expert-contour procedure and Dice, IoU, HD95 code
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
4. The MedSAM3-v1 LoRA file from the authors' public release, placed at `checkpoints/best_lora_weights.pt`. It was not trained or modified in this study. Download and verify:

   ```bash
   hf download lal-Joey/MedSAM3_v1 best_lora_weights.pt \
     --revision f579e37e4db9f2c7e85a133b7970d59907e36fec --local-dir checkpoints
   echo "499e638bb7c51dbe0dcc3bfb9dbfada74fc2d725e953fbb5bdb2dd1b72106f91  checkpoints/best_lora_weights.pt" | sha256sum -c
   ```
5. The four patient-partition CSVs in `splits/`. They are not distributed. Generate them with the builder in Section 5 and check them against `splits/README.md`.

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

Cohort flow (manuscript Figure 2). MIMIC-CXR-JPG metadata contained 377,110
image records, of which 243,334 AP/PA image records remained after excluding
non-frontal views. The CheXpert label table contained 227,827 study records. Of
these, 18,291 studies had an uncertain Pneumonia label (−1) and 168,642 had a
missing Pneumonia label; these 186,933 studies were excluded. The remaining
40,894 studies with explicit Pneumonia labels of 0 or 1 were linked to AP/PA
image metadata, and records without an accessible JPG file were excluded.
Metadata linkage and image file verification yielded a final cohort of 41,503
radiographs from 21,612 patients, comprising 17,207 report-derived
Pneumonia-positive and 24,296 report-derived Pneumonia-negative images: AP
22,894 images from 11,951 subjects and PA 18,609 images from 12,385 subjects
(after the AP exclusion below).

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
These CSVs are not in the public repository because they contain MIMIC
identifiers, and there is no separate download for them. Generate them
locally and compare their SHA256 values with `splits/README.md`. With
scikit-learn 1.7.2 both views reproduce those checksums byte for byte. A
different scikit-learn build can shuffle groups differently, so the checksum
comparison is required.

**Order matters for AP.** The study drew the AP split on the cohort before the
five-image exclusion of Section 4 (22,899 images, 11,953 subjects) and removed
those images afterwards. Building the AP split on the 22,894-image cohort gives
a different partition (only 8% of the outer-test subjects coincide). So build
the splits first, then run `apply_ap_cohort_exclusion.py`, which removes the five
rows from the split CSVs as well as from the manifests. The PA split has no
exclusion.

1. Build the MedSAM3 and CheXMask datasets (Section 4) up to, but not including,
   `apply_ap_cohort_exclusion.py`.
2. Build the splits:

```bash
python training/build_fixed_patient_splits.py \
  --data-base-dir "$DATA_BASE_DIR" \
  --labels-json "$DATA_BASE_DIR/pneumonia_labels.json" \
  --split-dir "$SPLIT_DIR" \
  --split-seed 42 --n-folds 5 --outer-test-ratio 0.15
```

3. Create `$SPLIT_DIR/AP_excluded_dicom_ids.json` as
   `{"excluded_dicom_ids": [...]}` listing the AP cohort images whose CheXMask
   Left or Right Lung RLE is empty. All 22,899 pre-exclusion AP images are in the
   CheXMask table, and exactly five of them have an empty RLE: these are the five
   excluded images. Then run
   `python preprocessing/apply_ap_cohort_exclusion.py`.
4. Check:

```bash
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
python evaluation/cluster_bootstrap_outer_test_ci.py --results-root "$RESULTS_ROOT" --n-boot 2000
python evaluation/paired_cluster_bootstrap_delta.py --results-root "$RESULTS_ROOT" --n-boot 20000
python evaluation/clustered_delong_delta_auroc.py --results-root "$RESULTS_ROOT"
python evaluation/analyze_crossfit_operating_point.py
python evaluation/extract_fold_vs_raw_table.py
python evaluation/evaluate_uncertainty_endpoint_sensitivity.py
python evaluation/summarize_training_seed_robustness.py
python evaluation/evaluate_training_policy_sensitivity.py
python mask_analysis/analyze_input_geometry.py
```

The scripts create `aggregates/` and `paper_tables/` under `$RESULTS_ROOT` when
they are missing. If `aggregates/_canonical_70runs.txt` is absent they use the
run list released as `results/primary/canonical_70runs.txt`. Run them in the
order above: `cluster_bootstrap_outer_test_ci.py` writes the per-image prediction
table that `analyze_crossfit_operating_point.py` reads, and
`clustered_delong_delta_auroc.py` reads the paired-bootstrap CSV.

Bootstrap resampling keeps all images from one subject together (seed 42).
Marginal AUROC and AUPRC intervals, operating points (Table S5), preprocessing
controls (Table 6 and Table S9), and lung-attribution intervals use 2,000
replicates. The paired intervals in Table 4 and Figure 5 use 20,000 replicates.
Table 4 p-values are the clustered DeLong tests. Rebuild the control contrasts
with `--n-boot 2000`; the released control files are the 2,000-replicate
estimates. `python <script> --help` lists inputs.

### Recomputation check

On 2026-10-17 the statistics below were recomputed from the stored outer-test
predictions of the 70 primary runs (no retraining) and compared with the files in
`results/`:

| Item | Result |
| --- | --- |
| Paired cluster bootstrap, 20,000 replicates, seed 42 (140 per-backbone and 20 pooled contrasts) | Every numeric column equal to the released CSV within 1e-12 |
| Clustered DeLong (Table 4 p-values) | Numeric columns equal within 1e-12. The regenerated `Table_4_delta_auroc_vs_raw.md` is byte-identical |
| Marginal cluster bootstrap, 2,000 replicates (`aggregate_metrics.csv`, `bootstrap_summary.csv`) | Equal within 1e-9 |
| Table S5, operating points | 34 numeric columns equal within 1e-9 |
| Table S1 | Means and population SDs equal within 1e-15 |
| Table S4 | Table rows identical |
| Table S2 Panels A and B and Table 3, PA 200 images | Rebuilt with `cxr-segmentation-validation/pa200_reference.py` from the public reference contours, with the primary-pipeline masks for the 46 cohort images as in the manuscript. Every Dice, IoU, and CheXMask value and every 46-image MedSAM3 value equals the manuscript. The MedSAM3 masks of the other 154 images were rebuilt, and MedSAM3 GPU output changes by a few dozen pixels between runs, so two HD95 cells differ by 0.1 px: 68.2 (manuscript 68.3) for the 154-image mean and 120.6 (manuscript 120.7) for the 200-image SD |
| Patient partition | PA and AP CSVs regenerated with the builder match the SHA256 values in `splits/README.md` (Section 5) |

Table 6, Table 7, and Table S8 were checked for agreement with their CSV sources
and with Table 4, not recomputed from predictions.

The released tables were also compared cell by cell with the manuscript and its
supplementary Excel files: Tables 3 to 7, Table S2 to S5, Table S7, and every
numeric cell of Tables S1, S5, S8, S9, S10, and S11 agree, apart from the two
PA HD95 cells above. Table 5 has no separate source file; it is formatted from
`aggregate_metrics.csv` with the rounding stated in
`evaluation/format_table5_preprocessing_means.py`. The AP part of Table S2 needs
the expert masks, which are not in this release (credentialed PhysioNet users can
request them, `DATA_GOVERNANCE.md`), and Table S7 and Figure 6 need the
fold checkpoints, so those are not reproducible from this release alone. Table S3, the controls (Table S9), and the seed
runs (Table S10) were not rerun.

### Which script builds which table or figure

| Reported item | Where the manuscript numbers are |
| --- | --- |
| Table 3, expert-reference segmentation | `results/supplement/Table_3_expert_reference_lung_segmentation.md`, from the two Table S2 summary JSON files |
| Table 4 and Figure 5, paired ΔAUROC | `results/primary/Table_4_delta_auroc_vs_raw.md`. Intervals in `paired_delta_by_backbone.csv` are 20,000 replicates. Draw the figure with `figures/plot_figure5_delta_auroc.py`. The p-values printed in Table 4 are `table4_unadjusted_p` and `table4_holm_p` on the 56 versus-raw rows, from `evaluation/clustered_delong_delta_auroc.py`. The four `table4_*` columns (`table4_unadjusted_p`, `table4_holm_p`, `table4_se`, `table4_z`) are copies of `delta_auc_p`, `delta_auc_p_holm`, `delta_auc_se`, and `delta_auc_z` from `Table_4_clustered_delong_source.csv` for those 56 rows and are empty for the other rows. No script writes them. They are the same in `paired_delta_by_backbone.csv` and in the delta rows of `configuration_and_metrics.csv` |
| Table 5, backbone means by preprocessing | `results/primary/Table_5_mean_by_preprocessing.md`. Format with `evaluation/format_table5_preprocessing_means.py` |
| Table S1, fold-level AUROC/AUPRC | `results/primary/Table_S1_fold_level_validation_performance.csv`. Rebuild with `evaluation/extract_fold_vs_raw_table.py`, which writes `paper_tables/Table_S1_fold_level_validation_performance.csv` in this layout. Values agree with the released file to 1e-15 (last-digit floating-point differences in the standard deviations) |
| Table S4, input geometry | `results/primary/Table_S4_input_geometry_characteristics.csv`. Rebuild with `mask_analysis/analyze_input_geometry.py` |
| Table S5, operating point and calibration | `results/primary/Table_S5_operating_point_and_calibration.csv` (2,000 replicates). Rebuild with `evaluation/analyze_crossfit_operating_point.py` |
| Table S9 and Table 6, preprocessing controls | `results/sensitivity/preprocessing_controls/` (2,000 replicates). Format Table 6 with `evaluation/format_table6_preprocessing_controls.py` and Table S9 with `evaluation/format_table_s9_preprocessing_controls.py` |
| Table S10 and Table 7, training seeds | `results/sensitivity/seed_robustness/`. Format Table 7 with `evaluation/format_table7_training_seed.py` and Table S10 with `evaluation/format_table_s10_training_seed.py` |
| Table S2, expert-reference summaries | PA, 200 public reference contours: `results/supplement/Table_S2_pa200_expert_reference_summary.json`, recomputed with `cxr-segmentation-validation/pa200_reference.py` (Table 3 reports the same 200-image numbers). AP, 100 contours: `results/supplement/Table_S2_ap_expert_reference_summary.json`. Metric code is `cxr-segmentation-validation/evaluate_segmentation.py` |
| Table S3, mask failure and agreement | `results/supplement/Table_S3_segmentation_agreement_fallback_classification_error.csv`. Rebuild with `mask_analysis/summarize_mask_source_failure_table.py` and `mask_analysis/analyze_segmentation_quality_proxies.py` |
| Table S6, model and software record | `results/supplement/Table_S6_model_checkpoint_software.md`, rebuilt by `tools/export_table_s6.py` |
| Supplementary Figure S1, calibration curves | `figures/plot_figure_s1_calibration.py` using `results/figures/Supplementary_Figure_S1_calibration_bins.csv` |
| Table 4 p-values, clustered DeLong source | `results/primary/Table_4_clustered_delong_source.csv`, the numeric source of the Table 4 p-values. Rebuild with `evaluation/clustered_delong_delta_auroc.py` |
| Table S7, lung-field attribution | `results/supplement/Table_S7_lung_field_attribution_concentration.csv` (2,000 replicates). Rebuild with `interpretability/analyze_lung_attribution_concentration.py` |
| Table S8, uncertain-label endpoint | `results/supplement/Table_S8_uncertainty_endpoint_sensitivity.md`. Numeric files are `results/sensitivity/uncertainty_endpoint/`. Rebuild with `evaluation/evaluate_uncertainty_endpoint_sensitivity.py` |
| Table S11, training-policy sensitivity | `results/sensitivity/training_policy/Table_S11_training_policy_sensitivity.md`. Rebuild the CSV with `evaluation/evaluate_training_policy_sensitivity.py` and format it with `evaluation/format_table_s11_training_policy.py` |
| Grad-CAM figures | `interpretability/generate_gradcam_pneumonia.py` and `interpretability/generate_figure6_gradcam.py`. Write the output outside this repository. The published heatmaps need the unreleased fold checkpoints |
| Expert-reference Dice, IoU, and HD95 | `cxr-segmentation-validation/evaluate_segmentation.py` |

`results/` holds the aggregate numbers and the identifier-free manuscript
tables listed above. Per-image expert tables and Grad-CAM figures identify
individual radiographs and are not included there.

## 8. Checkpoints and predictions

**Trained model checkpoints.** The fold-specific trained checkpoints used in
this study are not redistributed through the public repository, because
PhysioNet asks that models derived from MIMIC data be shared on PhysioNet
under the same agreement as the source data (see `DATA_GOVERNANCE.md`). All model
architectures, pretrained initialization sources, training configurations,
patient-partition procedures, random seeds, checkpoint-selection criteria,
and evaluation code required to reproduce the trained models are provided in
the repository. The model identifiers, software versions, and checkpoint
checksums are in `configs/models.json`, `configs/reference_environment.json`,
and `checkpoints/trained_checkpoint_manifest.csv`.

**Evaluation outputs.** Aggregate evaluation results are in `results/`,
including the 20,000-replicate paired intervals used in Table 4 and Figure 5,
the clustered DeLong p-values of Table 4, Tables 3, 5, 6, and 7, fold-level Table S1,
expert-reference Table S2, Table S3, input-geometry Table S4, operating-point
Table S5, Table S6, lung-attribution Table S7, uncertain-label Table S8, and
Tables S9, S10, and S11.
Patient-level or image-level prediction files containing MIMIC-CXR identifiers
are not redistributed. Credentialed users of MIMIC-CXR-JPG can regenerate
these outputs using the released evaluation scripts and fixed patient-partition
files.

A SHA256 list of the 350 fold checkpoints (70 runs × 5 folds), without the
weight files, is in `checkpoints/trained_checkpoint_manifest.csv`.
`CHECKSUMS.sha256` lists the SHA256 of every file in this release except itself.
Verify with `sha256sum -c CHECKSUMS.sha256`. Regenerate it after any edit with
`bash tools/update_checksums.sh`.
`configs/training_config.json` is the shared hyperparameter record from the
previous public snapshot. The per-run files under `configs/ap/` and `configs/pa/` are
the executed configurations.

## 9. Retraining: time, resources, and reproducibility limits

Retraining is the route to the trained models, since the fold checkpoints are
not distributed (Section 8). The MedSAM3 LoRA needs no retraining; download it
as in Section 2.

**Hardware.** One NVIDIA H200 (143,771 MiB), batch size 64, bfloat16 (RAD-DINO
float32), 512 x 512 input. Peak GPU memory was not logged. Smaller GPUs may
need a smaller batch size, which changes the training recipe.

**Wall-clock time per run** (5 folds plus outer-test evaluation, one H200,
measured from the run folder timestamp to `summary.json` for the 70 primary
runs; approximate, and includes data loading):

| Backbone | Median per run | Range |
| --- | --- | --- |
| DINOv3-Base | 1.3 h | 1.0-1.5 h |
| EVA-X-Base | 1.3 h | 1.1-1.4 h |
| DenseNet-201 | 1.3 h | 1.2-1.4 h |
| ResNet-152 | 2.0 h | 1.9-2.5 h |
| ConvNeXt V2-Base | 2.6 h | 2.3-2.9 h |
| RAD-DINO | 3.4 h | 2.8-5.2 h |
| Swin V2-Base | 4.1 h | 3.8-4.7 h |

The 70 primary runs took about 162 GPU-hours in total (roughly 7 days on one
GPU). The control, seed, and training-policy sweeps are additional. One
experiment (`--split-mode cv5`) is the cheapest unit to try, and `MAX_FOLDS=1`
trains a single fold.

**Disk.** One run folder is about 4.5 GB (best and last checkpoints for five
folds). The AP MedSAM3 mask dataset on our machine is about 174 GB and the
cropped AP set about 23 GB. Mask generation time is not reported here.

**What will not be identical.** Retraining from the released configs, seeds,
and splits repeats the experiment, but not bit for bit:

- The new weight files will not match the SHA256 values in
  `checkpoints/trained_checkpoint_manifest.csv`.
- GPU kernels (cuDNN, bfloat16 accumulation), PyTorch and CUDA builds, and the
  GPU model can change results even with fixed seeds. The reported runs used
  `repro_mode=false` and `deterministic_algorithms=false` (recorded in each config).
  `--repro` and `--deterministic-algorithms` exist, but a run with them is not the
  configuration that produced the tables and is slower.
- Use partition CSVs whose SHA256 values match `splits/README.md`. Re-running the
  split under another scikit-learn build, or building AP after the exclusion
  instead of before it, assigns subjects differently (Section 5).
- Expect AUROC, AUPRC and thresholds to differ slightly from the tables, and
  small preprocessing contrasts to be within that variation (see the training-seed
  analysis, Table 7 and Table S10, for the size of seed-to-seed variation we
  measured). We did not run a repeat on the same seed and hardware.
- Grad-CAM figures drawn from retrained weights will not match Figure 6.

