# Evaluation protocol

Dice, IoU, and HD95 compare each automated lung mask with the expert contour. Agreement between MedSAM3 and CheXMask is not the reference. `evaluate_segmentation.py` implements the definitions below.

## Sample

The reference set is 100 AP radiographs from the patient-level outer test of the pneumonia cohort. One radiograph was kept per patient. Among 312 patients who had both a pneumonia-positive and a pneumonia-negative eligible AP examination, one label was assigned at random with seed 42, so each patient entered one stratum. The draw then contained 50 report-derived pneumonia-positive and 50 pneumonia-negative radiographs. Random draws used one NumPy Generator initialized with seed 42.

Within each label, the challenging quota was the patient-level prevalence of challenging examinations multiplied by 50, with halves rounded upward. That produced 26 challenging and 24 routine radiographs in the positive stratum, and 12 challenging and 38 routine radiographs in the negative stratum. The prevalence was 391 of 766 positive-stratum patients and 232 of 997 negative-stratum patients.

An examination was challenging when any of the following was true: CheXpert Consolidation = 1, Pleural Effusion = 1, or Pleural Other = 1, or the DICOM orientation meaning was Recumbent. CheXpert values of −1 and missing values did not qualify. Portable acquisition, support devices, automated lung masks, and classifier scores were recorded outside the sampler and were not used to select the sample.

The saved sample file is sorted after those draws. The sort keys are `pneumonia_label` descending, then `stratum` ascending, then `subject_id`, then `dicom_id`. Text order puts challenging before routine, so the file runs through positive challenging, positive routine, negative challenging, and negative routine. `sample_id` is the row number in that sorted file. This sort is not a random shuffle.

The contouring list is a second object. A new NumPy Generator, again initialized with seed 42, permutes the sorted sample. That generator does not continue the sampling stream. The permutation drops the label and stratum, and `review_id` runs from 1 to 100 in the permuted order. The contouring screen followed `review_id`. Subgroup summaries of this set are exploratory because it contains 100 images.

`sample_expert_reference.py` repeats both steps. It reads a local copy of `AP_outer_test.csv` and a local MIMIC-CXR-JPG tree, and it writes the sorted sample outside this repository. The sampling stream is fixed: one Generator from seed 42 assigns labels for dual-label subjects in `subject_id` order; sampling then takes positive challenging subjects, one sorted DICOM identifier from each selected subject, then positive routine, negative challenging, and negative routine in the same way. Subject identifiers are sorted before each `choice`. When a patient has more than one candidate image, those DICOM identifiers are sorted and one index is drawn. A challenging patient contributes an image that meets the rule. A routine patient contributes an image that does not. The locked draw excluded no image for a missing file, so a local tree must contain every outer-test JPEG.

The published file checksum `b88e2f4c17dbf2d33082ea9dde12d95c912ace0ff7dbc641131adeb7289123ef` includes the image paths from the analysis machine. A redraw with a different local path will not reproduce that file byte for byte. The path-independent identity checksum of the sorted sample, taken over every sample column except `image_path`, is `2762de922d87c9961c7454d75ef9dd5ad5019a96f8b4f47ec3110b6f5f0be301`. The annotation-order checksum, taken over `review_id` and `dicom_id` in the permuted contouring list, is `0e53756acab9416ebcb80e6b7f4c1638ebf3e91b44b9e3d0321f8e193ed99389`. The script prints both digests and stops if either differs from the locked files. On the analysis machine, with the study outer-test CSV and MIMIC-CXR-JPG tree, that run printed `identity_matches_locked_sample=true` and `annotation_order_matches_locked_sample=true`. Redrawing the list still leaves the expert masks unavailable, so the manuscript scores cannot be recomputed from this repository alone.

## Mask alignment and binarization

Scores were computed on the native radiograph pixel grid. The expert lung is every PNG pixel whose gray value is greater than 0. A `.npy` mask is converted with `astype(bool)`.

For the reported 100 comparisons, the stored MedSAM3 mask, the stored CheXMask mask, and the expert mask already had the same height and width, so no resampling was applied. The code still defines the fallback: when the shapes differ, the prediction is resized to the reference height and width with Pillow nearest-neighbor interpolation, then binarized again with gray value greater than 0. The reference mask is not resized.

## Dice and IoU

Both scores use the whole-lung masks. Dice is `2|A ∩ B| / (|A| + |B|)`. IoU is `|A ∩ B| / |A ∪ B|`.

If both masks are empty, Dice is missing. If the union is empty, IoU is missing. If one mask is empty and the other is not, Dice and IoU are 0.

## HD95

The reported HD95 is the definition used for the manuscript tables. It stays in this code so a local rescoring of the same masks reproduces those tables.

The boundary of a mask is the set of foreground pixels removed by one binary erosion (`scipy.ndimage.binary_erosion`, one iteration, `border_value=0`). If that erosion removes every foreground pixel, the foreground itself is used as the boundary.

For each boundary pixel, the distance is the Euclidean distance to the nearest foreground pixel of the other mask. The other mask's interior counts. Distances from the reference boundary to the prediction and from the prediction boundary to the reference are pooled, and `numpy.percentile` takes one 95th percentile of that pool. The unit is pixels.

This is not boundary-to-boundary surface distance. On a 64×64 pair of filled disks, radii 10 and 12 pixels with centers one column apart, the manuscript definition returns 2.8284271247461903 px. The 95th percentile of the pooled distances between the two boundaries on that same pair is 3 px. `--self-check` locks the manuscript value.

This is also not the MONAI Hausdorff distance. MONAI summarizes each direction at the 95th percentile and then takes the larger of those two numbers. This study pools the directed distances first and then takes one percentile.

If either mask is empty, HD95 is missing and that pair is omitted from the HD95 summary. In the reported 100 comparisons, neither method had an empty mask.

## Summaries

For each metric, the summary reports the number of finite values, the mean, the sample standard deviation with divisor `n - 1`, the median, and the first and third quartiles. A single finite value has standard deviation 0. A metric with no finite values reports a count of 0.

The public script prints those aggregates and the counts of paired files, unmatched files, resized pairs, and empty masks. It does not print file names. A per-image table is optional and belongs outside this repository when the file stem identifies a radiograph.

The sheet for an explicit reader judgment that MedSAM3 or CheXMask was wrong was left blank. This protocol therefore scores the contours. It does not apply a separate expert-error flag.
