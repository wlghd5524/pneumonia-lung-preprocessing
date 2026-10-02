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
| `primary/paired_delta_by_backbone.csv` | Paired preprocessing-minus-raw differences by backbone |
| `primary/paired_delta_pooled.csv` | Same contrasts pooled across backbones |
| `primary/canonical_70runs.txt` | Run-folder names of the 70 configurations |
| `sensitivity/` | Preprocessing-control, training-seed, uncertainty-endpoint, and training-policy summaries |

Numbered supplementary tables and figures are not stored here. The scripts
that build them are listed in the repository README.

Tables that list individual radiographs (expert-reference rows, Grad-CAM case
identifiers, and the split assignment files) are not redistributed here.
