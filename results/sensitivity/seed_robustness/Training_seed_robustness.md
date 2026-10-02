# Training-seed robustness of paired preprocessing effects

Manuscript versions: Table 7 is the eight-row two-backbone mean (`paper_tables/Table_7_training_seed_robustness.md`). Supplementary Table S10 holds panels A–D (`paper_tables/Supplementary_Table_S10_training_seed_robustness.md`).

Patient partitions were held fixed. Every run of a given view used the same outer-test patients and the same five development folds (split seed 42; subject lists hashed to one value per view across all 30 runs). Only the training seed changed: 42 (the canonical run), 123, and 2026. Each AUROC is the five-model ensemble on the untouched outer-test set.

Scope of the repeat: AP and PA × ResNet-152 and EVA-X-Base × raw, MedSAM3 mask, MedSAM3 crop, CheXMask mask, and CheXMask crop. That is 20 configurations × 3 seeds = 60 runs. The other five backbones were not retrained.

Δ is preprocessed minus raw, matched on view, backbone, and training seed. Mean and SD are across the three seeds. SD divides by n−1 = 2. Positive means the preprocessed model scored higher than raw.

The outer-test images and labels were the same across seeds and preprocessing conditions within a view (matched on DICOM file name and label): AP 3,280 images, 1,582 pneumonia-positive; PA 2,521 images, 750 pneumonia-positive.

## 1. Two-backbone mean ΔAUROC versus raw

This is the same summary used in the main preprocessing comparison: the average of ResNet-152 and EVA-X-Base.

| View | Preprocessing | Seed 42 | Seed 123 | Seed 2026 | Mean | SD | Direction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| AP | MedSAM3 crop | +0.0021 | +0.0015 | +0.0031 | +0.0022 | 0.0008 | 3/3 higher than raw |
| AP | CheXMask crop | +0.0026 | +0.0020 | +0.0027 | +0.0024 | 0.0004 | 3/3 higher than raw |
| AP | MedSAM3 mask | -0.0049 | -0.0064 | -0.0054 | -0.0056 | 0.0008 | 3/3 lower than raw |
| AP | CheXMask mask | -0.0026 | -0.0034 | -0.0021 | -0.0027 | 0.0006 | 3/3 lower than raw |
| PA | MedSAM3 crop | -0.0017 | +0.0013 | -0.0015 | -0.0006 | 0.0017 | 2/3 lower than raw |
| PA | CheXMask crop | +0.0003 | -0.0012 | +0.0002 | -0.0002 | 0.0008 | 2/3 higher than raw |
| PA | MedSAM3 mask | -0.0186 | -0.0187 | -0.0175 | -0.0183 | 0.0007 | 3/3 lower than raw |
| PA | CheXMask mask | -0.0222 | -0.0150 | -0.0157 | -0.0177 | 0.0040 | 3/3 lower than raw |

## 2. Backbone-specific ΔAUROC versus raw

| View | Backbone | Preprocessing | Seed 42 | Seed 123 | Seed 2026 | Mean | SD | Direction |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| AP | ResNet-152 | MedSAM3 crop | +0.0034 | +0.0033 | +0.0022 | +0.0030 | 0.0007 | 3/3 higher than raw |
| AP | ResNet-152 | CheXMask crop | +0.0049 | +0.0044 | +0.0040 | +0.0044 | 0.0005 | 3/3 higher than raw |
| AP | ResNet-152 | MedSAM3 mask | -0.0034 | -0.0040 | -0.0059 | -0.0044 | 0.0013 | 3/3 lower than raw |
| AP | ResNet-152 | CheXMask mask | -0.0001 | -0.0010 | -0.0017 | -0.0009 | 0.0008 | 3/3 lower than raw |
| AP | EVA-X-Base | MedSAM3 crop | +0.0009 | -0.0004 | +0.0040 | +0.0015 | 0.0022 | 2/3 higher than raw |
| AP | EVA-X-Base | CheXMask crop | +0.0003 | -0.0004 | +0.0014 | +0.0004 | 0.0009 | 2/3 higher than raw |
| AP | EVA-X-Base | MedSAM3 mask | -0.0064 | -0.0088 | -0.0050 | -0.0067 | 0.0019 | 3/3 lower than raw |
| AP | EVA-X-Base | CheXMask mask | -0.0052 | -0.0058 | -0.0026 | -0.0045 | 0.0017 | 3/3 lower than raw |
| PA | ResNet-152 | MedSAM3 crop | -0.0006 | -0.0006 | -0.0028 | -0.0014 | 0.0013 | 3/3 lower than raw |
| PA | ResNet-152 | CheXMask crop | +0.0030 | -0.0030 | -0.0040 | -0.0013 | 0.0038 | 2/3 lower than raw |
| PA | ResNet-152 | MedSAM3 mask | -0.0085 | -0.0142 | -0.0162 | -0.0130 | 0.0040 | 3/3 lower than raw |
| PA | ResNet-152 | CheXMask mask | -0.0190 | -0.0079 | -0.0101 | -0.0123 | 0.0059 | 3/3 lower than raw |
| PA | EVA-X-Base | MedSAM3 crop | -0.0028 | +0.0032 | -0.0001 | +0.0001 | 0.0030 | 2/3 lower than raw |
| PA | EVA-X-Base | CheXMask crop | -0.0023 | +0.0006 | +0.0044 | +0.0009 | 0.0034 | 2/3 higher than raw |
| PA | EVA-X-Base | MedSAM3 mask | -0.0286 | -0.0232 | -0.0188 | -0.0236 | 0.0049 | 3/3 lower than raw |
| PA | EVA-X-Base | CheXMask mask | -0.0255 | -0.0222 | -0.0214 | -0.0230 | 0.0022 | 3/3 lower than raw |

## 3. Crop minus hard mask, ΔAUROC

| View | Backbone | Contrast | Seed 42 | Seed 123 | Seed 2026 | Mean | SD | Direction |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| AP | ResNet-152 | MedSAM3 crop − mask | +0.0067 | +0.0074 | +0.0081 | +0.0074 | 0.0007 | 3/3 crop higher |
| AP | EVA-X-Base | MedSAM3 crop − mask | +0.0074 | +0.0084 | +0.0089 | +0.0082 | 0.0008 | 3/3 crop higher |
| AP | Two-backbone mean | MedSAM3 crop − mask | +0.0070 | +0.0079 | +0.0085 | +0.0078 | 0.0007 | 3/3 crop higher |
| PA | ResNet-152 | MedSAM3 crop − mask | +0.0079 | +0.0135 | +0.0134 | +0.0116 | 0.0032 | 3/3 crop higher |
| PA | EVA-X-Base | MedSAM3 crop − mask | +0.0259 | +0.0265 | +0.0187 | +0.0237 | 0.0043 | 3/3 crop higher |
| PA | Two-backbone mean | MedSAM3 crop − mask | +0.0169 | +0.0200 | +0.0160 | +0.0176 | 0.0021 | 3/3 crop higher |
| AP | ResNet-152 | CheXMask crop − mask | +0.0049 | +0.0054 | +0.0057 | +0.0053 | 0.0004 | 3/3 crop higher |
| AP | EVA-X-Base | CheXMask crop − mask | +0.0055 | +0.0054 | +0.0040 | +0.0050 | 0.0009 | 3/3 crop higher |
| AP | Two-backbone mean | CheXMask crop − mask | +0.0052 | +0.0054 | +0.0048 | +0.0052 | 0.0003 | 3/3 crop higher |
| PA | ResNet-152 | CheXMask crop − mask | +0.0220 | +0.0049 | +0.0061 | +0.0110 | 0.0096 | 3/3 crop higher |
| PA | EVA-X-Base | CheXMask crop − mask | +0.0232 | +0.0228 | +0.0258 | +0.0239 | 0.0016 | 3/3 crop higher |
| PA | Two-backbone mean | CheXMask crop − mask | +0.0226 | +0.0139 | +0.0159 | +0.0175 | 0.0046 | 3/3 crop higher |

## 4. Backbone-specific ΔAUPRC versus raw

| View | Backbone | Preprocessing | Seed 42 | Seed 123 | Seed 2026 | Mean | SD | Direction |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| AP | ResNet-152 | MedSAM3 crop | +0.0001 | +0.0058 | +0.0009 | +0.0022 | 0.0031 | 3/3 higher than raw |
| AP | ResNet-152 | CheXMask crop | +0.0026 | +0.0079 | +0.0025 | +0.0044 | 0.0031 | 3/3 higher than raw |
| AP | ResNet-152 | MedSAM3 mask | -0.0045 | -0.0002 | -0.0068 | -0.0038 | 0.0033 | 3/3 lower than raw |
| AP | ResNet-152 | CheXMask mask | -0.0014 | +0.0011 | -0.0031 | -0.0011 | 0.0021 | 2/3 lower than raw |
| AP | EVA-X-Base | MedSAM3 crop | +0.0010 | -0.0008 | +0.0023 | +0.0008 | 0.0016 | 2/3 higher than raw |
| AP | EVA-X-Base | CheXMask crop | +0.0015 | -0.0020 | +0.0014 | +0.0003 | 0.0020 | 2/3 higher than raw |
| AP | EVA-X-Base | MedSAM3 mask | -0.0072 | -0.0091 | -0.0053 | -0.0072 | 0.0019 | 3/3 lower than raw |
| AP | EVA-X-Base | CheXMask mask | -0.0067 | -0.0071 | -0.0029 | -0.0056 | 0.0023 | 3/3 lower than raw |
| PA | ResNet-152 | MedSAM3 crop | +0.0091 | +0.0079 | -0.0018 | +0.0051 | 0.0060 | 2/3 higher than raw |
| PA | ResNet-152 | CheXMask crop | +0.0070 | -0.0005 | -0.0047 | +0.0006 | 0.0059 | 2/3 lower than raw |
| PA | ResNet-152 | MedSAM3 mask | -0.0058 | -0.0180 | -0.0179 | -0.0139 | 0.0070 | 3/3 lower than raw |
| PA | ResNet-152 | CheXMask mask | -0.0160 | -0.0029 | -0.0073 | -0.0087 | 0.0067 | 3/3 lower than raw |
| PA | EVA-X-Base | MedSAM3 crop | -0.0054 | +0.0011 | -0.0046 | -0.0029 | 0.0035 | 2/3 lower than raw |
| PA | EVA-X-Base | CheXMask crop | -0.0052 | +0.0011 | +0.0033 | -0.0003 | 0.0044 | 2/3 higher than raw |
| PA | EVA-X-Base | MedSAM3 mask | -0.0429 | -0.0341 | -0.0288 | -0.0353 | 0.0071 | 3/3 lower than raw |
| PA | EVA-X-Base | CheXMask mask | -0.0388 | -0.0329 | -0.0311 | -0.0343 | 0.0040 | 3/3 lower than raw |

## 5. Absolute outer-test AUROC

| View | Backbone | Input | Seed 42 | Seed 123 | Seed 2026 | Mean | SD |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| AP | ResNet-152 | Raw | 0.7860 | 0.7847 | 0.7851 | 0.7853 | 0.0006 |
| AP | ResNet-152 | MedSAM3 mask | 0.7826 | 0.7807 | 0.7792 | 0.7808 | 0.0017 |
| AP | ResNet-152 | MedSAM3 crop | 0.7893 | 0.7881 | 0.7873 | 0.7882 | 0.0010 |
| AP | ResNet-152 | CheXMask mask | 0.7859 | 0.7838 | 0.7834 | 0.7843 | 0.0014 |
| AP | ResNet-152 | CheXMask crop | 0.7908 | 0.7892 | 0.7890 | 0.7897 | 0.0010 |
| AP | EVA-X-Base | Raw | 0.8128 | 0.8134 | 0.8105 | 0.8122 | 0.0015 |
| AP | EVA-X-Base | MedSAM3 mask | 0.8064 | 0.8045 | 0.8056 | 0.8055 | 0.0009 |
| AP | EVA-X-Base | MedSAM3 crop | 0.8137 | 0.8129 | 0.8145 | 0.8137 | 0.0008 |
| AP | EVA-X-Base | CheXMask mask | 0.8076 | 0.8075 | 0.8080 | 0.8077 | 0.0002 |
| AP | EVA-X-Base | CheXMask crop | 0.8131 | 0.8129 | 0.8119 | 0.8127 | 0.0006 |
| PA | ResNet-152 | Raw | 0.7797 | 0.7807 | 0.7811 | 0.7805 | 0.0007 |
| PA | ResNet-152 | MedSAM3 mask | 0.7712 | 0.7666 | 0.7649 | 0.7675 | 0.0032 |
| PA | ResNet-152 | MedSAM3 crop | 0.7791 | 0.7801 | 0.7783 | 0.7792 | 0.0009 |
| PA | ResNet-152 | CheXMask mask | 0.7607 | 0.7729 | 0.7710 | 0.7682 | 0.0066 |
| PA | ResNet-152 | CheXMask crop | 0.7827 | 0.7778 | 0.7771 | 0.7792 | 0.0031 |
| PA | EVA-X-Base | Raw | 0.8261 | 0.8235 | 0.8229 | 0.8242 | 0.0017 |
| PA | EVA-X-Base | MedSAM3 mask | 0.7975 | 0.8003 | 0.8041 | 0.8006 | 0.0033 |
| PA | EVA-X-Base | MedSAM3 crop | 0.8233 | 0.8268 | 0.8228 | 0.8243 | 0.0022 |
| PA | EVA-X-Base | CheXMask mask | 0.8006 | 0.8013 | 0.8015 | 0.8011 | 0.0004 |
| PA | EVA-X-Base | CheXMask crop | 0.8238 | 0.8242 | 0.8273 | 0.8251 | 0.0019 |

## Reading

Hard-mask versus raw was lower in all 8 backbone-specific contrasts and in all 3 seeds (24 of 24 paired comparisons). The two-backbone mean drop was about 0.003 to 0.006 on AP and about 0.015 to 0.022 on PA. Seed-to-seed SD of those mean drops was 0.0006 to 0.0040, smaller than the drop itself on PA and usually smaller on AP as well.

Crop versus raw stayed small. Across the 8 backbone-specific crop contrasts, only 3 kept the same sign in all three seeds (AP ResNet-152, both crops, slightly higher; PA ResNet-152 MedSAM3 crop, slightly lower). The other five changed sign. Two-backbone mean crop effects on AP stayed slightly positive (about +0.002) in all three seeds. On PA they sat within about ±0.002 of zero and did not keep one direction. For crops, the seed SD was often as large as the mean difference.

Crop minus the matching hard mask stayed positive in all 8 backbone-specific contrasts and all 3 seeds. That direction is the stable paired effect: cropping scored higher than hard masking, while cropping did not reliably score higher than raw.

EVA-X-Base outer-test AUROC was higher than ResNet-152 in all 30 view × input × seed comparisons (gap +0.0217 to +0.0502). This repeat does not change the caution on architectural superiority. Only two backbones were retrained, and a stable ordering of these two models is not a claim that one architecture class is superior.
