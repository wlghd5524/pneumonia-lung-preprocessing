# Table S2. Detailed Expert-Reference Lung Segmentation Performance in PA and AP Radiographs

Clinician-reviewed lung contours from the public PhysioNet dataset Heart and lung segmentations for MIMIC-CXR, MIMIC-CXR-JPG, and Montgomery County TB databases were available for 200 PA radiographs. Overall segmentation performance for the complete 200-image PA reference set is reported in Table 3 of the main manuscript. Of these radiographs, 46 belonged to the report-derived pneumonia-analysis cohort and reused masks generated during the primary preprocessing pipeline, whereas the remaining 154 were outside the pneumonia-analysis cohort. For the latter images, MedSAM3 was applied using the same inference settings as in the primary analysis, and CheXMask masks were reconstructed from the publicly available RLE annotations. Segmentation metrics were computed at native image resolution against the same clinician-reviewed reference contours, with HD95 reported in pixels. PA subgroup results according to pneumonia-analysis cohort overlap and report-derived Pneumonia status are presented below. Expert-reference performance in AP radiographs sampled from the study outer-test cohort is reported separately in Panel C.

## A. Descriptive subgroup results according to report-derived pneumonia-analysis cohort overlap

| Subgroup | Method | N | Dice, mean ± SD | IoU, mean ± SD | HD95 (px), mean ± SD |
| --- | --- | ---: | ---: | ---: | ---: |
| Overlap with pneumonia-analysis cohort | MedSAM3 | 46 | 0.944 ± 0.051 | 0.897 ± 0.079 | 99.2 ± 161.1 |
| Overlap with pneumonia-analysis cohort | CheXMask | 46 | 0.940 ± 0.028 | 0.888 ± 0.048 | 59.1 ± 56.6 |
| Outside pneumonia-analysis cohort | MedSAM3 | 154 | 0.950 ± 0.029 | 0.906 ± 0.050 | 68.3 ± 105.3 |
| Outside pneumonia-analysis cohort | CheXMask | 154 | 0.938 ± 0.035 | 0.885 ± 0.058 | 71.1 ± 80.9 |

## B. Descriptive results by report-derived pneumonia label within the 46-image cohort-overlap subset

| Method | Pneumonia | N | Dice, mean ± SD | IoU, mean ± SD | HD95 (px), mean ± SD |
| --- | --- | ---: | ---: | ---: | ---: |
| MedSAM3 | Negative | 19 | 0.930 ± 0.072 | 0.875 ± 0.109 | 136.9 ± 215.5 |
| CheXMask | Negative | 19 | 0.940 ± 0.030 | 0.889 ± 0.051 | 52.0 ± 50.2 |
| MedSAM3 | Positive | 27 | 0.954 ± 0.027 | 0.912 ± 0.046 | 72.7 ± 104.8 |
| CheXMask | Positive | 27 | 0.940 ± 0.027 | 0.888 ± 0.047 | 64.1 ± 61.0 |

## C. AP expert-reference performance in the study outer-test sample

| Group | Method | N | Dice, mean ± SD | IoU, mean ± SD | HD95 (px), mean ± SD [median] |
| --- | --- | ---: | ---: | ---: | ---: |
| Overall | MedSAM3 | 100 | 0.930 ± 0.054 | 0.873 ± 0.083 | 115.9 ± 203.3 [48.4] |
| Overall | CheXMask | 100 | 0.915 ± 0.060 | 0.849 ± 0.091 | 92.6 ± 75.9 [64.3] |
| Pneumonia negative | MedSAM3 | 50 | 0.938 ± 0.054 | 0.887 ± 0.079 | 98.1 ± 209.4 [41.6] |
| Pneumonia negative | CheXMask | 50 | 0.926 ± 0.052 | 0.866 ± 0.079 | 73.3 ± 50.1 [54.3] |
| Pneumonia positive | MedSAM3 | 50 | 0.922 ± 0.054 | 0.859 ± 0.085 | 133.7 ± 197.4 [73.7] |
| Pneumonia positive | CheXMask | 50 | 0.905 ± 0.065 | 0.832 ± 0.098 | 111.9 ± 91.5 [79.3] |
| Challenging | MedSAM3 | 38 | 0.902 ± 0.075 | 0.828 ± 0.110 | 206.0 ± 302.5 [88.7] |
| Challenging | CheXMask | 38 | 0.883 ± 0.076 | 0.798 ± 0.110 | 136.6 ± 98.1 [98.9] |
| Routine | MedSAM3 | 62 | 0.947 ± 0.024 | 0.900 ± 0.043 | 60.6 ± 58.1 [39.8] |
| Routine | CheXMask | 62 | 0.935 ± 0.034 | 0.880 ± 0.057 | 65.6 ± 39.5 [53.0] |
| Portable | MedSAM3 | 85 | 0.930 ± 0.046 | 0.873 ± 0.074 | 111.7 ± 171.8 [49.0] |
| Portable | CheXMask | 85 | 0.913 ± 0.057 | 0.844 ± 0.088 | 96.3 ± 76.8 [66.5] |
| Not portable | MedSAM3 | 15 | 0.927 ± 0.090 | 0.874 ± 0.126 | 139.6 ± 338.3 [37.7] |
| Not portable | CheXMask | 15 | 0.928 ± 0.072 | 0.873 ± 0.106 | 71.6 ± 69.3 [50.7] |
| Difficult lung boundary | MedSAM3 | 22 | 0.889 ± 0.087 | 0.809 ± 0.125 | 221.9 ± 314.4 [95.0] |
| Difficult lung boundary | CheXMask | 22 | 0.873 ± 0.091 | 0.785 ± 0.129 | 146.0 ± 115.6 [95.2] |
| Boundary not marked difficult | MedSAM3 | 78 | 0.941 ± 0.033 | 0.891 ± 0.055 | 85.9 ± 148.5 [42.4] |
| Boundary not marked difficult | CheXMask | 78 | 0.927 ± 0.041 | 0.867 ± 0.067 | 77.5 ± 52.1 [57.2] |
| Any technical mark | MedSAM3 | 32 | 0.902 ± 0.076 | 0.828 ± 0.111 | 179.1 ± 269.4 [87.2] |
| Any technical mark | CheXMask | 32 | 0.878 ± 0.078 | 0.790 ± 0.111 | 139.8 ± 98.6 [111.4] |
| No technical mark | MedSAM3 | 68 | 0.943 ± 0.034 | 0.894 ± 0.055 | 86.1 ± 157.0 [42.4] |
| No technical mark | CheXMask | 68 | 0.933 ± 0.039 | 0.876 ± 0.063 | 70.4 ± 49.5 [53.1] |

## Notes for this repository

Panels A and B and the PA rows of Table 3 are summarized in `Table_S2_pa200_expert_reference_summary.json`, recomputed with `cxr-segmentation-validation/pa200_reference.py` from the public reference contours. As in the manuscript, the 46 cohort images are scored with the primary-pipeline masks and the 154 other images with the same MedSAM3 settings and with CheXMask masks rebuilt from the public RLE. All Dice and IoU values, all CheXMask values, and the 46-image MedSAM3 values of the recomputation equal the table. The MedSAM3 masks of the 154 non-cohort images from the manuscript run were not kept, and MedSAM3 output on the GPU changes by a few dozen pixels between runs. The recomputed MedSAM3 HD95 is therefore 68.2 ± 105.3 px for the 154 images (table: 68.3 ± 105.3) and 75.4 ± 120.6 px for all 200 (Table 3: 75.4 ± 120.7).

Panel C is summarized in `Table_S2_ap_expert_reference_summary.json`. The AP contours were drawn by one study author who is an infectious disease specialist; sampling and contouring rules are in `cxr-segmentation-validation/`. Technical difficulty means the reader marked rotation, low inspiration, or a difficult lung boundary. Association with classification error is Table S3, Panel E.

Per-image metrics are not distributed because they identify radiographs.
