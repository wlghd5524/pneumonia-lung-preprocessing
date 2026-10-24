# Table S2. Detailed Expert-Reference Lung Segmentation Performance in PA and AP Radiographs

Clinician-reviewed lung contours from the public PhysioNet dataset Heart and lung segmentations for MIMIC-CXR, MIMIC-CXR-JPG, and Montgomery County TB databases were available for 200 PA radiographs. Overall segmentation performance for the complete 200-image PA reference set is reported in Table 3 of the main manuscript. Of these radiographs, 46 belonged to the report-derived pneumonia-analysis cohort and reused masks generated during the primary preprocessing pipeline, whereas the remaining 154 were outside the pneumonia-analysis cohort. For the latter images, MedSAM3 was applied using the same inference settings as in the primary analysis, and CheXMask masks were reconstructed from the publicly available RLE annotations. Segmentation metrics were computed at native image resolution against the same clinician-reviewed reference contours, with HD95 reported in pixels. PA subgroup results according to pneumonia-analysis cohort overlap and report-derived Pneumonia status are presented below. Expert-reference performance in AP radiographs sampled from the study outer-test cohort is reported separately in Panel C.

## A. Descriptive subgroup results according to report-derived pneumonia-analysis cohort overlap

| Subgroup | Method | N | Dice, mean ± SD | IoU, mean ± SD | HD95 (px), mean ± SD |
| --- | --- | ---: | ---: | ---: | ---: |
| Overlap with pneumonia-analysis cohort | MedSAM3 | 46 | 0.944 ± 0.051 | 0.897 ± 0.079 | 125.4 ± 177.2 |
| Overlap with pneumonia-analysis cohort | CheXMask | 46 | 0.940 ± 0.028 | 0.888 ± 0.048 | 81.8 ± 72.5 |
| Outside pneumonia-analysis cohort | MedSAM3 | 154 | 0.950 ± 0.029 | 0.906 ± 0.050 | 95.7 ± 128.4 |
| Outside pneumonia-analysis cohort | CheXMask | 154 | 0.938 ± 0.035 | 0.885 ± 0.058 | 93.2 ± 95.1 |

## B. Descriptive results by report-derived pneumonia label within the 46-image cohort-overlap subset

| Method | Pneumonia | N | Dice, mean ± SD | IoU, mean ± SD | HD95 (px), mean ± SD |
| --- | --- | ---: | ---: | ---: | ---: |
| MedSAM3 | Negative | 19 | 0.930 ± 0.072 | 0.875 ± 0.109 | 156.9 ± 222.4 |
| CheXMask | Negative | 19 | 0.940 ± 0.030 | 0.889 ± 0.051 | 73.1 ± 58.4 |
| MedSAM3 | Positive | 27 | 0.954 ± 0.027 | 0.912 ± 0.046 | 103.2 ± 137.4 |
| CheXMask | Positive | 27 | 0.940 ± 0.027 | 0.888 ± 0.047 | 88.0 ± 81.4 |

## C. AP expert-reference performance in the study outer-test sample

| Group | Method | N | Dice, mean ± SD | IoU, mean ± SD | HD95 (px), mean ± SD [median] |
| --- | --- | ---: | ---: | ---: | ---: |
| Overall | MedSAM3 | 100 | 0.930 ± 0.054 | 0.873 ± 0.083 | 153.8 ± 227.3 [65.5] |
| Overall | CheXMask | 100 | 0.915 ± 0.060 | 0.849 ± 0.091 | 119.8 ± 92.5 [88.1] |
| Pneumonia negative | MedSAM3 | 50 | 0.938 ± 0.054 | 0.887 ± 0.079 | 129.4 ± 230.1 [57.1] |
| Pneumonia negative | CheXMask | 50 | 0.926 ± 0.052 | 0.866 ± 0.079 | 90.9 ± 55.7 [70.4] |
| Pneumonia positive | MedSAM3 | 50 | 0.922 ± 0.054 | 0.859 ± 0.085 | 178.3 ± 223.9 [111.8] |
| Pneumonia positive | CheXMask | 50 | 0.905 ± 0.065 | 0.832 ± 0.098 | 148.7 ± 111.8 [113.9] |
| Challenging | MedSAM3 | 38 | 0.902 ± 0.075 | 0.828 ± 0.110 | 265.3 ± 324.3 [136.3] |
| Challenging | CheXMask | 38 | 0.883 ± 0.076 | 0.798 ± 0.110 | 174.3 ± 114.6 [143.0] |
| Routine | MedSAM3 | 62 | 0.947 ± 0.024 | 0.900 ± 0.043 | 85.6 ± 86.8 [52.5] |
| Routine | CheXMask | 62 | 0.935 ± 0.034 | 0.880 ± 0.057 | 86.4 ± 54.3 [66.9] |
| Portable | MedSAM3 | 85 | 0.930 ± 0.046 | 0.873 ± 0.074 | 151.9 ± 200.1 [66.4] |
| Portable | CheXMask | 85 | 0.913 ± 0.057 | 0.844 ± 0.088 | 124.8 ± 95.0 [96.3] |
| Not portable | MedSAM3 | 15 | 0.927 ± 0.090 | 0.874 ± 0.126 | 164.8 ± 353.2 [54.9] |
| Not portable | CheXMask | 15 | 0.928 ± 0.072 | 0.873 ± 0.106 | 91.4 ± 73.0 [68.9] |
| Difficult lung boundary | MedSAM3 | 22 | 0.889 ± 0.087 | 0.809 ± 0.125 | 281.7 ± 335.1 [150.5] |
| Difficult lung boundary | CheXMask | 22 | 0.873 ± 0.091 | 0.785 ± 0.129 | 187.9 ± 128.5 [145.8] |
| Boundary not marked difficult | MedSAM3 | 78 | 0.941 ± 0.033 | 0.891 ± 0.055 | 117.8 ± 172.6 [61.2] |
| Boundary not marked difficult | CheXMask | 78 | 0.927 ± 0.041 | 0.867 ± 0.067 | 100.6 ± 69.3 [74.3] |
| Any technical mark | MedSAM3 | 32 | 0.902 ± 0.076 | 0.828 ± 0.111 | 232.0 ± 291.9 [120.7] |
| Any technical mark | CheXMask | 32 | 0.878 ± 0.078 | 0.790 ± 0.111 | 179.3 ± 112.1 [145.8] |
| No technical mark | MedSAM3 | 68 | 0.943 ± 0.034 | 0.894 ± 0.055 | 117.1 ± 180.5 [60.4] |
| No technical mark | CheXMask | 68 | 0.933 ± 0.039 | 0.876 ± 0.063 | 91.8 ± 65.9 [68.0] |

## Notes for this repository

Panels A and B and the PA rows of Table 3 are summarized in `Table_S2_pa200_expert_reference_summary.json`, computed with `cxr-segmentation-validation/pa200_reference.py` from the public reference contours. As in the manuscript, the 46 cohort images are scored with the primary-pipeline masks and the 154 other images with the same MedSAM3 settings and with CheXMask masks rebuilt from the public RLE. MedSAM3 output on the GPU changes by a few dozen pixels between runs, so a new rebuild of the 154 MedSAM3 masks can move their HD95 by about 0.1 px; Dice and IoU do not change at three decimals.

HD95 in every panel is the symmetric boundary-to-boundary 95th-percentile Hausdorff distance defined in `cxr-segmentation-validation/evaluation_protocol.md`.

Panel C is summarized in `Table_S2_ap_expert_reference_summary.json`. The AP contours were drawn by one study author who is an infectious disease specialist; sampling and contouring rules are in `cxr-segmentation-validation/`. Technical difficulty means the reader marked rotation, low inspiration, or a difficult lung boundary. Association with classification error is Table S3, Panel E.

Per-image metrics are not distributed because they identify radiographs. The 100 AP expert masks are not distributed (`DATA_GOVERNANCE.md`).
