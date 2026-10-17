# Table S3. Segmentation Agreement, Center-Crop Fallback Frequency, and Association with Classification Error

## A. Cross-source mask agreement vs classification error (outer-test set)

Cross-source Dice is **not** expert ground-truth accuracy. Error is EVA-X-Base ensemble probability vs the pneumonia label at threshold 0.5. N = 5801 outer-test images with both stored masks.

| View | MedSAM3–CheXMask Dice | N | Raw error | MedSAM3-mask model error | CheXMask-mask model error |
| --- | --- | ---: | ---: | ---: | ---: |
| AP | Dice ≥ 0.90 | 2463 | 608/2463 (24.7%) | 621/2463 (25.2%) | 625/2463 (25.4%) |
| AP | 0.80–0.90 | 630 | 165/630 (26.2%) | 169/630 (26.8%) | 173/630 (27.5%) |
| AP | Dice < 0.80 | 187 | 64/187 (34.2%) | 62/187 (33.2%) | 61/187 (32.6%) |
| PA | Dice ≥ 0.90 | 2439 | 484/2439 (19.8%) | 586/2439 (24.0%) | 583/2439 (23.9%) |
| PA | 0.80–0.90 | 58 | 13/58 (22.4%) | 16/58 (27.6%) | 17/58 (29.3%) |
| PA | Dice < 0.80 | 24 | 3/24 (12.5%) | 3/24 (12.5%) | 4/24 (16.7%) |
| Overall | Dice ≥ 0.90 | 4902 | 1092/4902 (22.3%) | 1207/4902 (24.6%) | 1208/4902 (24.6%) |
| Overall | 0.80–0.90 | 688 | 178/688 (25.9%) | 185/688 (26.9%) | 190/688 (27.6%) |
| Overall | Dice < 0.80 | 211 | 67/211 (31.8%) | 65/211 (30.8%) | 65/211 (30.8%) |

Fallback definition (Panels B and C). After keeping at most two 4-connected components and removing components smaller than 20 pixels, a mask triggers an 80% × 80% center crop if it is empty or covers <10% of the image. This is the crop-pipeline rule, scored separately for each mask source. Panel B reports how often that event occurs in the analysis cohort. Panel C uses the same per-source rule on the outer-test set and reports the corresponding crop-model error, because that classifier is the one that received the center-crop input when Fallback = Yes.

## B. Center-crop fallback frequency by view and pneumonia class (analysis cohort)

The table is restricted to the 41,503-image analysis cohort (22,894 AP + 18,609 PA). Mask failure is an empty lung mask, or, for MedSAM3, zero detections after the 0.5 confidence filter and NMS. Center-crop fallback uses the shared definition above. The two MedSAM3 empty masks are a subset of the fallback counts.

CheXMask mask failure is 0 **in this analysis cohort** because five AP radiographs with missing CheXMask lung RLE were excluded from the entire study before the cohort was fixed (2 pneumonia-negative, 3 pneumonia-positive). Those five images are not in *N* and are therefore not counted as CheXMask failures in the table. This is not evidence that CheXMask succeeded on every original radiograph. If the five missing-RLE images are counted as CheXMask failures before cohort restriction, CheXMask AP-negative failure is 2 / 11,517 and AP-positive failure is 3 / 11,382.

| Mask source | View | Pneumonia | N images | Mask failure, n (%) | Center-crop fallback, n (%) |
| --- | --- | --- | ---: | ---: | ---: |
| MedSAM3 | AP | Negative | 11,515 | 0 (0.00%) | 43 (0.37%) |
| MedSAM3 | AP | Positive | 11,379 | 2 (0.02%) | 67 (0.59%) |
| MedSAM3 | PA | Negative | 12,781 | 0 (0.00%) | 2 (0.02%) |
| MedSAM3 | PA | Positive | 5,828 | 0 (0.00%) | 3 (0.05%) |
| CheXMask | AP | Negative | 11,515 | 0 (0.00%) | 5 (0.04%) |
| CheXMask | AP | Positive | 11,379 | 0 (0.00%) | 3 (0.03%) |
| CheXMask | PA | Negative | 12,781 | 0 (0.00%) | 0 (0.00%) |
| CheXMask | PA | Positive | 5,828 | 0 (0.00%) | 1 (0.02%) |

## C. Center-crop fallback vs corresponding crop-model error (outer-test set)

Error is EVA-X-Base ensemble probability vs the pneumonia label at threshold 0.5. Fallback uses the same per-source crop-pipeline rule as Panel B. The crop-model column is the classifier trained on that source’s lung crop; when Fallback = Yes, that crop was the 80% × 80% center crop. Raw error is shown as a same-image reference that does not use the crop.

| Mask source | View | Pneumonia | Fallback | N | Raw error | Corresponding crop-model error |
| --- | --- | --- | --- | ---: | ---: | ---: |
| MedSAM3 | AP | Negative | No | 1693 | 457/1693 (27.0%) | 420/1693 (24.8%) |
| MedSAM3 | AP | Negative | Yes | 5 | 2/5 (40.0%) | 2/5 (40.0%) |
| MedSAM3 | AP | Positive | No | 1576 | 375/1576 (23.8%) | 401/1576 (25.4%) |
| MedSAM3 | AP | Positive | Yes | 6 | 3/6 (50.0%) | 2/6 (33.3%) |
| MedSAM3 | AP | All | No | 3269 | 832/3269 (25.5%) | 821/3269 (25.1%) |
| MedSAM3 | AP | All | Yes | 11 | 5/11 (45.5%) | 4/11 (36.4%) |
| MedSAM3 | PA | Negative | No | 1771 | 190/1771 (10.7%) | 291/1771 (16.4%) |
| MedSAM3 | PA | Negative | Yes | 0 | n/a | n/a |
| MedSAM3 | PA | Positive | No | 750 | 310/750 (41.3%) | 253/750 (33.7%) |
| MedSAM3 | PA | Positive | Yes | 0 | n/a | n/a |
| MedSAM3 | PA | All | No | 2521 | 500/2521 (19.8%) | 544/2521 (21.6%) |
| MedSAM3 | PA | All | Yes | 0 | n/a | n/a |
| MedSAM3 | Overall | Negative | No | 3464 | 647/3464 (18.7%) | 711/3464 (20.5%) |
| MedSAM3 | Overall | Negative | Yes | 5 | 2/5 (40.0%) | 2/5 (40.0%) |
| MedSAM3 | Overall | Positive | No | 2326 | 685/2326 (29.4%) | 654/2326 (28.1%) |
| MedSAM3 | Overall | Positive | Yes | 6 | 3/6 (50.0%) | 2/6 (33.3%) |
| MedSAM3 | Overall | All | No | 5790 | 1332/5790 (23.0%) | 1365/5790 (23.6%) |
| MedSAM3 | Overall | All | Yes | 11 | 5/11 (45.5%) | 4/11 (36.4%) |
| CheXMask | AP | Negative | No | 1698 | 459/1698 (27.0%) | 455/1698 (26.8%) |
| CheXMask | AP | Negative | Yes | 0 | n/a | n/a |
| CheXMask | AP | Positive | No | 1582 | 378/1582 (23.9%) | 376/1582 (23.8%) |
| CheXMask | AP | Positive | Yes | 0 | n/a | n/a |
| CheXMask | AP | All | No | 3280 | 837/3280 (25.5%) | 831/3280 (25.3%) |
| CheXMask | AP | All | Yes | 0 | n/a | n/a |
| CheXMask | PA | Negative | No | 1771 | 190/1771 (10.7%) | 341/1771 (19.3%) |
| CheXMask | PA | Negative | Yes | 0 | n/a | n/a |
| CheXMask | PA | Positive | No | 750 | 310/750 (41.3%) | 235/750 (31.3%) |
| CheXMask | PA | Positive | Yes | 0 | n/a | n/a |
| CheXMask | PA | All | No | 2521 | 500/2521 (19.8%) | 576/2521 (22.8%) |
| CheXMask | PA | All | Yes | 0 | n/a | n/a |
| CheXMask | Overall | Negative | No | 3469 | 649/3469 (18.7%) | 796/3469 (22.9%) |
| CheXMask | Overall | Negative | Yes | 0 | n/a | n/a |
| CheXMask | Overall | Positive | No | 2332 | 688/2332 (29.5%) | 611/2332 (26.2%) |
| CheXMask | Overall | Positive | Yes | 0 | n/a | n/a |
| CheXMask | Overall | All | No | 5801 | 1337/5801 (23.0%) | 1407/5801 (24.3%) |
| CheXMask | Overall | All | Yes | 0 | n/a | n/a |

## D. Lung-area ratio vs classification error (outer-test set)

Bins use the stored lung-area ratio of that mask source. The <10% bin is the fallback threshold used in the crop pipeline. On this outer-test set the stored-ratio <10% rule and the connected-component-filtered pipeline rule identified the same images. Error is EVA-X-Base ensemble probability vs the pneumonia label at threshold 0.5. The crop-model and mask-model columns are the classifiers trained on that source’s lung crop and hard mask, respectively.

| Mask source | View | Lung-area ratio | N | Raw error | Corresponding crop-model error | Corresponding mask-model error |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| MedSAM3 | AP | <10% | 11 | 5/11 (45.5%) | 4/11 (36.4%) | 5/11 (45.5%) |
| MedSAM3 | AP | 10–20% | 1514 | 438/1514 (28.9%) | 434/1514 (28.7%) | 443/1514 (29.3%) |
| MedSAM3 | AP | 20–30% | 1591 | 362/1591 (22.8%) | 352/1591 (22.1%) | 366/1591 (23.0%) |
| MedSAM3 | AP | 30–40% | 163 | 32/163 (19.6%) | 35/163 (21.5%) | 38/163 (23.3%) |
| MedSAM3 | AP | ≥40% | 1 | 0/1 (0.0%) | 0/1 (0.0%) | 0/1 (0.0%) |
| MedSAM3 | PA | <10% | 0 | n/a | n/a | n/a |
| MedSAM3 | PA | 10–20% | 431 | 105/431 (24.4%) | 112/431 (26.0%) | 134/431 (31.1%) |
| MedSAM3 | PA | 20–30% | 1570 | 305/1570 (19.4%) | 335/1570 (21.3%) | 366/1570 (23.3%) |
| MedSAM3 | PA | 30–40% | 499 | 83/499 (16.6%) | 90/499 (18.0%) | 97/499 (19.4%) |
| MedSAM3 | PA | ≥40% | 21 | 7/21 (33.3%) | 7/21 (33.3%) | 8/21 (38.1%) |
| MedSAM3 | Overall | <10% | 11 | 5/11 (45.5%) | 4/11 (36.4%) | 5/11 (45.5%) |
| MedSAM3 | Overall | 10–20% | 1945 | 543/1945 (27.9%) | 546/1945 (28.1%) | 577/1945 (29.7%) |
| MedSAM3 | Overall | 20–30% | 3161 | 667/3161 (21.1%) | 687/3161 (21.7%) | 732/3161 (23.2%) |
| MedSAM3 | Overall | 30–40% | 662 | 115/662 (17.4%) | 125/662 (18.9%) | 135/662 (20.4%) |
| MedSAM3 | Overall | ≥40% | 22 | 7/22 (31.8%) | 7/22 (31.8%) | 8/22 (36.4%) |
| CheXMask | AP | <10% | 0 | n/a | n/a | n/a |
| CheXMask | AP | 10–20% | 1150 | 332/1150 (28.9%) | 328/1150 (28.5%) | 348/1150 (30.3%) |
| CheXMask | AP | 20–30% | 1834 | 437/1834 (23.8%) | 433/1834 (23.6%) | 440/1834 (24.0%) |
| CheXMask | AP | 30–40% | 288 | 66/288 (22.9%) | 68/288 (23.6%) | 70/288 (24.3%) |
| CheXMask | AP | ≥40% | 8 | 2/8 (25.0%) | 2/8 (25.0%) | 1/8 (12.5%) |
| CheXMask | PA | <10% | 0 | n/a | n/a | n/a |
| CheXMask | PA | 10–20% | 370 | 87/370 (23.5%) | 101/370 (27.3%) | 116/370 (31.4%) |
| CheXMask | PA | 20–30% | 1610 | 314/1610 (19.5%) | 365/1610 (22.7%) | 375/1610 (23.3%) |
| CheXMask | PA | 30–40% | 514 | 91/514 (17.7%) | 100/514 (19.5%) | 104/514 (20.2%) |
| CheXMask | PA | ≥40% | 27 | 8/27 (29.6%) | 10/27 (37.0%) | 9/27 (33.3%) |
| CheXMask | Overall | <10% | 0 | n/a | n/a | n/a |
| CheXMask | Overall | 10–20% | 1520 | 419/1520 (27.6%) | 429/1520 (28.2%) | 464/1520 (30.5%) |
| CheXMask | Overall | 20–30% | 3444 | 751/3444 (21.8%) | 798/3444 (23.2%) | 815/3444 (23.7%) |
| CheXMask | Overall | 30–40% | 802 | 157/802 (19.6%) | 168/802 (20.9%) | 174/802 (21.7%) |
| CheXMask | Overall | ≥40% | 35 | 10/35 (28.6%) | 12/35 (34.3%) | 10/35 (28.6%) |

## E. AP expert-reference segmentation versus classification error

This panel uses the 100 AP expert contours in Table S2, Panel C. It is not the cross-source Dice in Panel A. Error is the EVA-X-Base five-fold ensemble probability versus the pneumonia label at threshold 0.5. MedSAM3 expert scores are paired with the MedSAM3 hard-mask and lung-crop models, and CheXMask scores with the corresponding CheXMask models. The raw-image model is the same-image comparator and does not use a lung mask. Comparisons are descriptive on 100 radiographs.

### Error rate by expert Dice

| Mask compared with the expert | Expert Dice | N | Raw-image error | Hard-mask model error | Lung-crop model error |
| --- | --- | ---: | ---: | ---: | ---: |
| MedSAM3 | Dice ≥ 0.90 | 85 | 27/85 (31.8%) | 28/85 (32.9%) | 28/85 (32.9%) |
| MedSAM3 | 0.80–0.90 | 11 | 6/11 (54.5%) | 5/11 (45.5%) | 4/11 (36.4%) |
| MedSAM3 | Dice < 0.80 | 4 | 0/4 (0.0%) | 0/4 (0.0%) | 0/4 (0.0%) |
| CheXMask | Dice ≥ 0.90 | 75 | 24/75 (32.0%) | 26/75 (34.7%) | 25/75 (33.3%) |
| CheXMask | 0.80–0.90 | 21 | 8/21 (38.1%) | 5/21 (23.8%) | 6/21 (28.6%) |
| CheXMask | Dice < 0.80 | 4 | 1/4 (25.0%) | 1/4 (25.0%) | 1/4 (25.0%) |

Fisher exact tests compare Dice < 0.90 with Dice ≥ 0.90. The 2×2 table is [high-Dice correct, high-Dice incorrect; low-Dice correct, low-Dice incorrect].

| Mask | Classifier | Odds ratio | Fisher p | 2×2 table |
| --- | --- | ---: | ---: | --- |
| MedSAM3 | MedSAM3 hard mask | 1.02 | 1.000 | 57, 28; 10, 5 |
| MedSAM3 | MedSAM3 lung crop | 0.74 | 0.769 | 57, 28; 11, 4 |
| MedSAM3 | Raw image | 1.43 | 0.560 | 58, 27; 9, 6 |
| CheXMask | CheXMask hard mask | 0.60 | 0.458 | 49, 26; 19, 6 |
| CheXMask | CheXMask lung crop | 0.78 | 0.805 | 50, 25; 18, 7 |
| CheXMask | Raw image | 1.20 | 0.807 | 51, 24; 16, 9 |

### Expert scores in correctly and incorrectly classified images

Values are median [first quartile, third quartile]. The Mann–Whitney test compares the expert-reference score between incorrect and correct images. A positive HD95 difference means the incorrect images had a larger boundary gap.

| Mask | Classifier | Outcome | N | Dice | IoU | HD95 (px) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| MedSAM3 | MedSAM3 hard mask | Correct | 67 | 0.949 [0.923, 0.959] | 0.903 [0.858, 0.920] | 47.7 [32.6, 89.0] |
| MedSAM3 | MedSAM3 hard mask | Incorrect | 33 | 0.944 [0.913, 0.959] | 0.894 [0.840, 0.921] | 49.0 [37.7, 99.3] |
| MedSAM3 | MedSAM3 lung crop | Correct | 68 | 0.950 [0.924, 0.959] | 0.904 [0.859, 0.921] | 46.0 [32.6, 90.2] |
| MedSAM3 | MedSAM3 lung crop | Incorrect | 32 | 0.942 [0.915, 0.958] | 0.891 [0.843, 0.919] | 60.0 [37.9, 98.5] |
| MedSAM3 | Raw image | Correct | 67 | 0.950 [0.928, 0.960] | 0.906 [0.866, 0.922] | 45.3 [32.6, 87.2] |
| MedSAM3 | Raw image | Incorrect | 33 | 0.942 [0.910, 0.958] | 0.891 [0.836, 0.919] | 66.0 [39.2, 99.7] |
| CheXMask | CheXMask hard mask | Correct | 68 | 0.938 [0.896, 0.950] | 0.883 [0.811, 0.906] | 63.9 [46.9, 113.6] |
| CheXMask | CheXMask hard mask | Incorrect | 32 | 0.932 [0.912, 0.954] | 0.873 [0.839, 0.911] | 64.6 [46.3, 99.5] |
| CheXMask | CheXMask lung crop | Correct | 68 | 0.937 [0.899, 0.949] | 0.881 [0.816, 0.904] | 64.8 [46.9, 110.4] |
| CheXMask | CheXMask lung crop | Incorrect | 32 | 0.937 [0.908, 0.954] | 0.882 [0.831, 0.911] | 62.9 [46.3, 112.4] |
| CheXMask | Raw image | Correct | 67 | 0.938 [0.909, 0.951] | 0.883 [0.832, 0.906] | 63.3 [46.8, 101.2] |
| CheXMask | Raw image | Incorrect | 33 | 0.930 [0.879, 0.953] | 0.868 [0.784, 0.911] | 65.0 [47.0, 133.2] |

| Mask | Classifier | Metric | Median, incorrect − correct | Mann–Whitney p |
| --- | --- | --- | ---: | ---: |
| MedSAM3 | MedSAM3 hard mask | Dice | -0.005 | 0.387 |
| MedSAM3 | MedSAM3 hard mask | IoU | -0.009 | 0.387 |
| MedSAM3 | MedSAM3 hard mask | HD95 (px) | 1.3 | 0.502 |
| MedSAM3 | MedSAM3 lung crop | Dice | -0.008 | 0.276 |
| MedSAM3 | MedSAM3 lung crop | IoU | -0.014 | 0.276 |
| MedSAM3 | MedSAM3 lung crop | HD95 (px) | 14.0 | 0.391 |
| MedSAM3 | Raw image | Dice | -0.008 | 0.108 |
| MedSAM3 | Raw image | IoU | -0.015 | 0.108 |
| MedSAM3 | Raw image | HD95 (px) | 20.7 | 0.163 |
| CheXMask | CheXMask hard mask | Dice | -0.006 | 0.804 |
| CheXMask | CheXMask hard mask | IoU | -0.010 | 0.804 |
| CheXMask | CheXMask hard mask | HD95 (px) | 0.7 | 0.807 |
| CheXMask | CheXMask lung crop | Dice | 0.001 | 0.731 |
| CheXMask | CheXMask lung crop | IoU | 0.001 | 0.731 |
| CheXMask | CheXMask lung crop | HD95 (px) | -1.9 | 0.807 |
| CheXMask | Raw image | Dice | -0.008 | 0.649 |
| CheXMask | Raw image | IoU | -0.014 | 0.649 |
| CheXMask | Raw image | HD95 (px) | 1.7 | 0.595 |

### Within pneumonia class

Median expert Dice, split by the pneumonia label. On the 50 pneumonia-negative images, the raw-image model and the MedSAM3 hard-mask model classified every image the same way (11 shared errors). The raw-image model and the CheXMask lung-crop model also matched on all 50 negative images. A Dice gap in that class is therefore shared with a model that does not use a lung mask.

| Mask | Classifier | Pneumonia | Incorrect / N | Dice, correct | Dice, incorrect | Mann–Whitney p |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| MedSAM3 | MedSAM3 hard mask | Negative | 11/50 | 0.957 [0.946, 0.962] | 0.913 [0.906, 0.952] | 0.022 |
| MedSAM3 | MedSAM3 hard mask | Positive | 22/50 | 0.929 [0.892, 0.952] | 0.946 [0.928, 0.961] | 0.077 |
| MedSAM3 | MedSAM3 lung crop | Negative | 11/50 | 0.957 [0.946, 0.962] | 0.931 [0.909, 0.952] | 0.035 |
| MedSAM3 | MedSAM3 lung crop | Positive | 21/50 | 0.934 [0.890, 0.952] | 0.945 [0.917, 0.960] | 0.231 |
| MedSAM3 | Raw image | Negative | 11/50 | 0.957 [0.946, 0.962] | 0.913 [0.906, 0.952] | 0.022 |
| MedSAM3 | Raw image | Positive | 22/50 | 0.938 [0.904, 0.954] | 0.944 [0.917, 0.958] | 0.488 |
| CheXMask | CheXMask hard mask | Negative | 12/50 | 0.948 [0.932, 0.954] | 0.918 [0.894, 0.930] | 0.005 |
| CheXMask | CheXMask hard mask | Positive | 20/50 | 0.910 [0.859, 0.938] | 0.943 [0.916, 0.957] | 0.003 |
| CheXMask | CheXMask lung crop | Negative | 11/50 | 0.948 [0.933, 0.954] | 0.914 [0.882, 0.929] | 0.005 |
| CheXMask | CheXMask lung crop | Positive | 21/50 | 0.912 [0.869, 0.937] | 0.945 [0.916, 0.956] | 0.002 |
| CheXMask | Raw image | Negative | 11/50 | 0.948 [0.933, 0.954] | 0.914 [0.882, 0.929] | 0.005 |
| CheXMask | Raw image | Positive | 22/50 | 0.914 [0.880, 0.938] | 0.940 [0.888, 0.955] | 0.045 |

### Same image, raw model versus mask-using model

Raw-correct and model-incorrect counts are images the raw model classified correctly and the mask-using model missed. These counts describe this 100-image sample.

| Mask | Model | Expert Dice | N | Both correct | Both incorrect | Raw correct, model incorrect | Raw incorrect, model correct |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| MedSAM3 | MedSAM3 hard mask | All | 100 | 64 | 30 | 3 | 3 |
| MedSAM3 | MedSAM3 hard mask | Dice ≥ 0.90 | 85 | 56 | 26 | 2 | 1 |
| MedSAM3 | MedSAM3 hard mask | Dice < 0.90 | 15 | 8 | 4 | 1 | 2 |
| MedSAM3 | MedSAM3 lung crop | All | 100 | 64 | 29 | 3 | 4 |
| MedSAM3 | MedSAM3 lung crop | Dice ≥ 0.90 | 85 | 56 | 26 | 2 | 1 |
| MedSAM3 | MedSAM3 lung crop | Dice < 0.90 | 15 | 8 | 3 | 1 | 3 |
| CheXMask | CheXMask hard mask | All | 100 | 65 | 30 | 2 | 3 |
| CheXMask | CheXMask hard mask | Dice ≥ 0.90 | 75 | 49 | 24 | 2 | 0 |
| CheXMask | CheXMask hard mask | Dice < 0.90 | 25 | 16 | 6 | 0 | 3 |
| CheXMask | CheXMask lung crop | All | 100 | 65 | 30 | 2 | 3 |
| CheXMask | CheXMask lung crop | Dice ≥ 0.90 | 75 | 49 | 23 | 2 | 1 |
| CheXMask | CheXMask lung crop | Dice < 0.90 | 25 | 16 | 7 | 0 | 2 |

Overall error on these 100 images:

- Raw image: 33/100 (33.0%)
- MedSAM3 hard mask: 33/100 (33.0%)
- MedSAM3 lung crop: 32/100 (32.0%)
- CheXMask hard mask: 32/100 (32.0%)
- CheXMask lung crop: 32/100 (32.0%)

Per-image link: `Table_S3_ap_expert_classification_link.csv`.
