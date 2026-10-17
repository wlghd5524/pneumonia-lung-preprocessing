# Expert-reference lung segmentation

This directory records how the 100 AP expert lung contours were drawn and how Dice, IoU, and HD95 were calculated against them. It contains the procedure and the metric code. Chest radiographs, expert masks, polygon files, and tables that name individual radiographs stay with credentialed users of MIMIC-CXR-JPG.

| File | Contents |
| --- | --- |
| `annotation_guidelines.md` | Annotator, software, boundaries, mask format, and review steps |
| `evaluation_protocol.md` | Sample rules and the metric definitions used for the reported scores |
| `evaluate_segmentation.py` | Those metric definitions, plus a synthetic self-check |
| `sample_expert_reference.py` | Redraw of the 100-image sample from a local MIMIC-CXR-JPG tree |
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
