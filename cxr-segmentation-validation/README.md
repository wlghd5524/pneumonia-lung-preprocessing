# Expert-reference lung segmentation

This directory records how the 100 AP expert lung contours were drawn and how Dice, IoU, and HD95 were calculated against them. It contains the procedure and the metric code. Chest radiographs, expert masks, polygon files, and tables that name individual radiographs stay with credentialed users of MIMIC-CXR-JPG.

| File | Contents |
| --- | --- |
| `annotation_guidelines.md` | Annotator, software, boundaries, mask format, and review steps |
| `evaluation_protocol.md` | Sample rules and the metric definitions used for the reported scores |
| `evaluate_segmentation.py` | Those metric definitions, plus a synthetic self-check |
| `sample_expert_reference.py` | Redraw of the 100-image sample from a local MIMIC-CXR-JPG tree |
| `pa200_reference.py` | Scores MedSAM3 and CheXMask on the 200 public PA expert contours (Table 3 and Table S2, Panels A and B) |
| `requirements.txt` | Packages used by the two scripts |

The metric packages match the pins in the repository `requirements.txt`.

## Check the metric without radiographs

```bash
python cxr-segmentation-validation/evaluate_segmentation.py --self-check
```

The check builds small synthetic masks in memory and in a temporary directory. It confirms identical overlap, partial overlap, empty masks, a one-pixel boundary, nearest-neighbor resizing, PNG and `.npy` binarization, and file pairing. It also locks the manuscript HD95 on a pair of disks: boundary-to-foreground distance returns 2.8284271247461903 px, while boundary-to-boundary distance on the same pair returns 3 px.

## Score local masks

Place the reference masks and one prediction set in two directories outside this repository. Pair files by stem. A stem may be a `.png` in one directory and a `.npy` in the other.

```bash
python cxr-segmentation-validation/evaluate_segmentation.py \
  --reference-dir /path/to/expert_masks \
  --prediction-dir /path/to/predicted_masks \
  --summary-json /path/outside/this/repo/summary.json
```

The summary counts pairs, unmatched files, resized pairs, and empty masks. It then reports Dice, IoU, and HD95 as count, mean, sample standard deviation, median, and quartiles. Missing metric values are omitted from those summaries. The summary does not list file names.

`--per-image-csv` writes one row per paired stem. When the stem is a DICOM identifier, keep that CSV outside this repository and do not commit it.

Run the command once for MedSAM3 and once for CheXMask. Each comparison uses the expert mask as the reference.

## Score the 200 public PA reference contours

The dataset *Heart and lung segmentations for MIMIC-CXR, MIMIC-CXR-JPG, and Montgomery County TB databases* on PhysioNet has expert lung contours for 200 PA radiographs. Part of the set is in the study cohort and the rest is not, so the masks of the rest are built here with the primary pipeline. Everything below runs outside this repository, and the dataset (its license file is ODC-By), MIMIC-CXR-JPG, and the CheXMask table must be downloaded under their own terms.

```bash
python cxr-segmentation-validation/pa200_reference.py prepare \
  --gold-root /path/to/heart-and-lung-segmentations.../mimic_masks \
  --mimic-root /path/to/mimic-cxr-jpg/2.1.0 \
  --out-dir /path/outside/this/repo/pa200
```

`prepare` copies the expert masks to `<out-dir>/reference/` named by DICOM identifier and writes `pa200_records.json`. It prints the two `preprocessing/build_cxr_lung_seg_dataset.py` commands to run next: one with `--seg-mode medsam3` and one with `--seg-mode chexmask`, both with `--labels-json <out-dir>/pa200_records.json`. These are the settings of the primary analysis. Then:

```bash
python cxr-segmentation-validation/pa200_reference.py summarize \
  --reference-dir /path/outside/this/repo/pa200/reference \
  --medsam3-root /path/outside/this/repo/pa200/medsam3 \
  --chexmask-root /path/outside/this/repo/pa200/chexmask \
  --cohort-json /path/to/pneumonia_labels.json \
  --summary-json /path/to/summary.json
```

The summary has counts, mean, SD, median, and quartiles for all 200 images, for the images that are in the study cohort, for those outside it, and for the cohort images by pneumonia label. It has no identifiers. It uses `evaluate_segmentation.py` for Dice, IoU, and HD95, so the definitions match the AP scores. The result is `results/supplement/Table_S2_pa200_expert_reference_summary.json`. MedSAM3 HD95 can differ by about 0.1 px between runs on a GPU, while Dice and IoU do not change at three decimals.

## Redraw the 100-image sample

```bash
python cxr-segmentation-validation/sample_expert_reference.py \
  --outer-csv /path/to/AP_outer_test.csv \
  --mimic-root /path/to/mimic-cxr-jpg/2.1.0 \
  --out-dir /path/outside/this/repo/expert_sample
```

The script refuses to write inside this repository. Its stdout reports stratum counts and two path-independent checksums. It does not print patient or image identifiers. The sample file is sorted by pneumonia label, stratum, subject identifier, and DICOM identifier. The contouring list is a separate permutation of that sorted file, from a new random generator with the same seed. `identity_matches_locked_sample=true` means the sorted sample matches the locked draw. `annotation_order_matches_locked_sample=true` means the contouring permutation matches the locked review order. The expert masks are still required before the manuscript Dice, IoU, and HD95 values can be recomputed.

## Data availability and restrictions

The source radiographs are MIMIC-CXR-JPG v2.1.0:

https://physionet.org/content/mimic-cxr-jpg/2.1.0/

PhysioNet asks that new annotations and other data derived from MIMIC be shared under the same credentialed access as the source data, through PhysioNet, rather than in a public repository:

https://www.physionet.org/news/post/mimic-derived-datasets-models/

| Material | This GitHub repository |
| --- | --- |
| MIMIC-CXR-JPG radiographs | Not included. Obtain them from PhysioNet under its data use agreement. |
| 100 new expert lung masks, polygon files, and per-image annotation tables | Not included. They are derived annotations and name individual radiographs. |
| Annotation guidelines, evaluation protocol, and evaluation code | Included in this directory. |
| An approved release path for the expert masks | None at present. |
| Manuscript Dice, IoU, and HD95 | Cannot be recomputed from this repository alone, because the reference masks are not here. |

`DATA_GOVERNANCE.md` states the same restriction for the rest of the repository.
