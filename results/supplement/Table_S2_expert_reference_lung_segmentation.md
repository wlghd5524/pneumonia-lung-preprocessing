# Table S2. Detailed Expert-Reference Lung Segmentation Performance

Dice, IoU, and HD95 compare each stored lung mask with an expert whole-lung contour at native radiograph resolution. HD95 is in pixels. Agreement between MedSAM3 and CheXMask is not the reference.

## A. Public PA expert contours overlapping the study cohort

Clinician-reviewed lung contours from the public PhysioNet dataset *Heart and lung segmentations for MIMIC-CXR, MIMIC-CXR-JPG, and Montgomery County TB databases* (200 PA images). Of these, **46** radiographs are also in the pneumonia study cohort and therefore have stored MedSAM3 and CheXMask masks. The remaining 154 gold-standard images have no stored MedSAM3 output because MedSAM3 was run only on pneumonia-labeled AP/PA images. All overlap images are PA. This panel does not include AP or portable AP examinations. Those are in Panel B.

| Method | N | Dice, mean ± SD | IoU, mean ± SD | HD95 (px), mean ± SD [median] |
| --- | ---: | ---: | ---: | ---: |
| MedSAM3 | 46 | 0.944 ± 0.051 | 0.897 ± 0.079 | 99.2 ± 161.1 [29.9] |
| CheXMask | 46 | 0.940 ± 0.028 | 0.888 ± 0.048 | 59.1 ± 56.6 [39.6] |

By pneumonia label in the overlap subset:

| Method | Pneumonia | N | Dice, mean ± SD | IoU, mean ± SD | HD95 (px), mean ± SD |
| --- | --- | ---: | ---: | ---: | ---: |
| MedSAM3 | Negative | 19 | 0.930 ± 0.072 | 0.875 ± 0.109 | 136.9 ± 215.5 |
| CheXMask | Negative | 19 | 0.940 ± 0.030 | 0.889 ± 0.051 | 52.0 ± 50.2 |
| MedSAM3 | Positive | 27 | 0.954 ± 0.027 | 0.912 ± 0.046 | 72.7 ± 104.8 |
| CheXMask | Positive | 27 | 0.940 ± 0.027 | 0.888 ± 0.047 | 64.1 ± 61.0 |

Per-image metrics: `Table_S2_expert_reference_lung_segmentation_per_image.csv`.

## B. AP expert-reference sample from the study outer test

One hundred AP outer-test radiographs, one image per patient, contoured by one study author who is an infectious disease specialist. The sample contains 50 report-derived pneumonia-positive and 50 pneumonia-negative radiographs, including 38 prespecified challenging and 62 routine examinations. Eighty-five examinations were portable and 15 were not. Subgroup rows are descriptive. Technical difficulty means the reader marked rotation, low inspiration, or a difficult lung boundary; 32 images had at least one of these marks, and 22 were marked as having a difficult lung boundary.

| Method | N | Dice, mean ± SD | IoU, mean ± SD | HD95 (px), mean ± SD [median] |
| --- | ---: | ---: | ---: | ---: |
| MedSAM3 | 100 | 0.930 ± 0.054 | 0.873 ± 0.083 | 115.9 ± 203.3 [48.4] |
| CheXMask | 100 | 0.915 ± 0.060 | 0.849 ± 0.091 | 92.6 ± 75.9 [64.3] |

| Group | N | Method | Dice, mean ± SD | IoU, mean ± SD | HD95 (px), mean ± SD [median] |
| --- | ---: | --- | ---: | ---: | ---: |
| Pneumonia negative | 50 | MedSAM3 | 0.938 ± 0.054 | 0.887 ± 0.079 | 98.1 ± 209.4 [41.6] |
| Pneumonia negative | 50 | CheXMask | 0.926 ± 0.052 | 0.866 ± 0.079 | 73.3 ± 50.1 [54.3] |
| Pneumonia positive | 50 | MedSAM3 | 0.922 ± 0.054 | 0.859 ± 0.085 | 133.7 ± 197.4 [73.7] |
| Pneumonia positive | 50 | CheXMask | 0.905 ± 0.065 | 0.832 ± 0.098 | 111.9 ± 91.5 [79.3] |
| Challenging | 38 | MedSAM3 | 0.902 ± 0.075 | 0.828 ± 0.110 | 206.0 ± 302.5 [88.7] |
| Challenging | 38 | CheXMask | 0.883 ± 0.076 | 0.798 ± 0.110 | 136.6 ± 98.1 [98.9] |
| Routine | 62 | MedSAM3 | 0.947 ± 0.024 | 0.900 ± 0.043 | 60.6 ± 58.1 [39.8] |
| Routine | 62 | CheXMask | 0.935 ± 0.034 | 0.880 ± 0.057 | 65.6 ± 39.5 [53.0] |
| Portable | 85 | MedSAM3 | 0.930 ± 0.046 | 0.873 ± 0.074 | 111.7 ± 171.8 [49.0] |
| Portable | 85 | CheXMask | 0.913 ± 0.057 | 0.844 ± 0.088 | 96.3 ± 76.8 [66.5] |
| Not portable | 15 | MedSAM3 | 0.927 ± 0.090 | 0.874 ± 0.126 | 139.6 ± 338.3 [37.7] |
| Not portable | 15 | CheXMask | 0.928 ± 0.072 | 0.873 ± 0.106 | 71.6 ± 69.3 [50.7] |
| Difficult lung boundary | 22 | MedSAM3 | 0.889 ± 0.087 | 0.809 ± 0.125 | 221.9 ± 314.4 [95.0] |
| Difficult lung boundary | 22 | CheXMask | 0.873 ± 0.091 | 0.785 ± 0.129 | 146.0 ± 115.6 [95.2] |
| Boundary not marked difficult | 78 | MedSAM3 | 0.941 ± 0.033 | 0.891 ± 0.055 | 85.9 ± 148.5 [42.4] |
| Boundary not marked difficult | 78 | CheXMask | 0.927 ± 0.041 | 0.867 ± 0.067 | 77.5 ± 52.1 [57.2] |
| Any technical mark | 32 | MedSAM3 | 0.902 ± 0.076 | 0.828 ± 0.111 | 179.1 ± 269.4 [87.2] |
| Any technical mark | 32 | CheXMask | 0.878 ± 0.078 | 0.790 ± 0.111 | 139.8 ± 98.6 [111.4] |
| No technical mark | 68 | MedSAM3 | 0.943 ± 0.034 | 0.894 ± 0.055 | 86.1 ± 157.0 [42.4] |
| No technical mark | 68 | CheXMask | 0.933 ± 0.039 | 0.876 ± 0.063 | 70.4 ± 49.5 [53.1] |

Per-image metrics: `Table_S2_ap_expert_reference_per_image.csv`. Sampling and contouring rules are in `paper_text/AP_expert_reference_integrated.md`. Association with classification error is Table S3, Panel E.
