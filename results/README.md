# Released evaluation outputs

These files are aggregate and fold-level results. They do not contain
MIMIC-CXR `subject_id` or `dicom_id` values, and they do not contain
image-level predictions.

Patient-level and image-level prediction files are not in this directory.
Credentialed users of MIMIC-CXR-JPG can regenerate them with the evaluation
scripts and the fixed patient-partition files described in `splits/README.md`.

Trained checkpoints are not in this directory. See `checkpoints/README.md`.

## Layout

| Path | Contents |
| --- | --- |
| `primary/aggregate_metrics.csv` | Outer-test AUROC/AUPRC and cluster-bootstrap intervals for the 70 configurations |
| `primary/aggregate_metrics_full.csv` | Same runs, with the executed hyperparameters |
| `primary/configuration_and_metrics.csv` | Configuration record plus metrics for every canonical run |
| `primary/bootstrap_summary.csv` | Subject-level cluster bootstrap versus image bootstrap |
| `primary/paired_delta_by_backbone.csv` | Paired differences by backbone. Table 4 and Figure 5 intervals use 20,000 subject-level replicates |
| `primary/paired_delta_pooled.csv` | Same contrasts pooled across backbones, also 20,000 replicates |
| `primary/Table_4_delta_auroc_vs_raw.md` | Manuscript Table 4. Intervals are the 20,000-replicate percentile intervals; p-values are clustered DeLong |
| `primary/Table_4_clustered_delong_source.csv` | Numeric source of the Table 4 p-values (clustered DeLong) |
| `primary/Table_5_mean_by_preprocessing.md` | Manuscript Table 5, formatted from `aggregate_metrics.csv` |
| `primary/canonical_70runs.txt` | Run-folder names of the 70 configurations |
| `primary/Table_S1_fold_level_validation_performance.csv` | Supplementary Table S1, five validation folds |
| `primary/Table_S4_input_geometry_characteristics.csv` | Supplementary Table S4 |
| `primary/Table_S5_operating_point_and_calibration.csv` | Supplementary Table S5, 2,000-replicate operating-point intervals |
| `supplement/Table_3_expert_reference_lung_segmentation.md` | Manuscript Table 3 |
| `supplement/Table_S2_expert_reference_lung_segmentation.md` | Supplementary Table S2, Panels A to C |
| `supplement/Table_S2_pa200_expert_reference_summary.json` | Table S2 Panels A and B and Table 3, PA expert-reference scores for the 200 public contours (all, overlap with the study cohort, outside it). Counts and summary statistics only |
| `supplement/Table_S7_lung_field_attribution_concentration.csv` | Table S7, lung-field attribution, 2,000-replicate intervals |
| `supplement/Table_S8_uncertainty_endpoint_sensitivity.md` | Table S8, CheXpert uncertain-label endpoint. Source CSVs are in `sensitivity/uncertainty_endpoint/` |
| `figures/` | Figure 5 and Supplementary Figure S1, drawn from the released tables |
| `supplement/` | Identifier-free Table 3 and Supplementary Tables S2, S3, S6, S7, and S8 |
| `sensitivity/preprocessing_controls/Table_6_preprocessing_controls.md` | Table 6, formatted from the 2,000-replicate control file |
| `sensitivity/preprocessing_controls/Table_S9_preprocessing_controls_by_backbone.md` | Supplementary Table S9, from the same file |
| `sensitivity/seed_robustness/Table_7_training_seed_robustness.md` | Table 7, formatted from the two-backbone seed summary |
| `sensitivity/seed_robustness/Table_S10_training_seed_robustness.md` | Supplementary Table S10, Panels A to D |
| `sensitivity/training_policy/Table_S11_training_policy_sensitivity.md` | Supplementary Table S11, training-policy sensitivity |
| `sensitivity/` | Preprocessing-control, training-seed, uncertainty-endpoint, and training-policy summaries. Controls remain the manuscript 2,000-replicate estimates |

Marginal AUROC and AUPRC intervals in `aggregate_metrics.csv` and
`bootstrap_summary.csv` remain the 2,000-replicate subject-level intervals.
The delta rows in `configuration_and_metrics.csv` use the same 20,000-replicate
intervals as `paired_delta_by_backbone.csv`. On the 56 preprocessing-versus-raw
rows, `table4_unadjusted_p` and `table4_holm_p` are the clustered DeLong
p-values printed in `Table_4_delta_auroc_vs_raw.md`. `delta_auc_p` remains the
paired cluster-bootstrap p-value and is not the p-value in Table 4.

Tables that list individual radiographs (expert-reference rows, Grad-CAM case
identifiers, and the split assignment files) are not redistributed here.
