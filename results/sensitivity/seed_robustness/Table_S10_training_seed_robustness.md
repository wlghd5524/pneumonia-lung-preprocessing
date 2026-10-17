# Table S10. Independent training-seed robustness of paired preprocessing effects

Main-text Table 7 reports the two-backbone mean ΔAUROC. This table provides the backbone-specific paired differences and the absolute outer-test AUROC at each training seed. Patient partitions were fixed (split seed 42). Training seeds were 42, 123, and 2026. Each score is the five-model ensemble on the same outer-test radiographs. Δ is preprocessed minus raw, or crop minus the matching hard mask, within the same view, backbone, and training seed. Mean and SD are across the three seeds; SD divides by n−1. Positive values indicate a higher score for the first condition. Only ResNet-152 and EVA-X-Base were retrained.

## A. Backbone-specific ΔAUROC versus raw

| View | Backbone | Preprocessing | Seed 42 | Seed 123 | Seed 2026 | Mean ΔAUROC | SD | Direction |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| AP | ResNet-152 | MedSAM3 crop | 0.0034 | 0.0033 | 0.0022 | 0.0030 | 0.0007 | 3/3 higher than raw |
| AP | ResNet-152 | CheXMask crop | 0.0049 | 0.0044 | 0.0040 | 0.0044 | 0.0005 | 3/3 higher than raw |
| AP | ResNet-152 | MedSAM3 mask | -0.0034 | -0.0040 | -0.0059 | -0.0044 | 0.0013 | 3/3 lower than raw |
| AP | ResNet-152 | CheXMask mask | -0.0001 | -0.0010 | -0.0017 | -0.0009 | 0.0008 | 3/3 lower than raw |
| AP | EVA-X-Base | MedSAM3 crop | 0.0009 | -0.0004 | 0.0040 | 0.0015 | 0.0022 | 2/3 higher than raw |
| AP | EVA-X-Base | CheXMask crop | 0.0003 | -0.0004 | 0.0014 | 0.0004 | 0.0009 | 2/3 higher than raw |
| AP | EVA-X-Base | MedSAM3 mask | -0.0064 | -0.0088 | -0.0050 | -0.0067 | 0.0019 | 3/3 lower than raw |
| AP | EVA-X-Base | CheXMask mask | -0.0052 | -0.0058 | -0.0026 | -0.0045 | 0.0017 | 3/3 lower than raw |
| PA | ResNet-152 | MedSAM3 crop | -0.0006 | -0.0006 | -0.0028 | -0.0014 | 0.0013 | 3/3 lower than raw |
| PA | ResNet-152 | CheXMask crop | 0.0030 | -0.0030 | -0.0040 | -0.0013 | 0.0038 | 2/3 lower than raw |
| PA | ResNet-152 | MedSAM3 mask | -0.0085 | -0.0142 | -0.0162 | -0.0130 | 0.0040 | 3/3 lower than raw |
| PA | ResNet-152 | CheXMask mask | -0.0190 | -0.0079 | -0.0101 | -0.0123 | 0.0059 | 3/3 lower than raw |
| PA | EVA-X-Base | MedSAM3 crop | -0.0028 | 0.0032 | -0.0001 | 0.0001 | 0.0030 | 2/3 lower than raw |
| PA | EVA-X-Base | CheXMask crop | -0.0023 | 0.0006 | 0.0044 | 0.0009 | 0.0034 | 2/3 higher than raw |
| PA | EVA-X-Base | MedSAM3 mask | -0.0286 | -0.0232 | -0.0188 | -0.0236 | 0.0049 | 3/3 lower than raw |
| PA | EVA-X-Base | CheXMask mask | -0.0255 | -0.0222 | -0.0214 | -0.0230 | 0.0022 | 3/3 lower than raw |

## B. Crop minus hard mask, ΔAUROC

| View | Backbone | Contrast | Seed 42 | Seed 123 | Seed 2026 | Mean ΔAUROC | SD | Direction |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| AP | ResNet-152 | MedSAM3 crop - mask | 0.0067 | 0.0074 | 0.0081 | 0.0074 | 0.0007 | 3/3 crop higher |
| AP | EVA-X-Base | MedSAM3 crop - mask | 0.0074 | 0.0084 | 0.0089 | 0.0082 | 0.0008 | 3/3 crop higher |
| AP | Two-backbone mean | MedSAM3 crop - mask | 0.0070 | 0.0079 | 0.0085 | 0.0078 | 0.0007 | 3/3 crop higher |
| PA | ResNet-152 | MedSAM3 crop - mask | 0.0079 | 0.0135 | 0.0134 | 0.0116 | 0.0032 | 3/3 crop higher |
| PA | EVA-X-Base | MedSAM3 crop - mask | 0.0259 | 0.0265 | 0.0187 | 0.0237 | 0.0043 | 3/3 crop higher |
| PA | Two-backbone mean | MedSAM3 crop - mask | 0.0169 | 0.0200 | 0.0160 | 0.0176 | 0.0021 | 3/3 crop higher |
| AP | ResNet-152 | CheXMask crop - mask | 0.0049 | 0.0054 | 0.0057 | 0.0053 | 0.0004 | 3/3 crop higher |
| AP | EVA-X-Base | CheXMask crop - mask | 0.0055 | 0.0054 | 0.0040 | 0.0050 | 0.0009 | 3/3 crop higher |
| AP | Two-backbone mean | CheXMask crop - mask | 0.0052 | 0.0054 | 0.0048 | 0.0052 | 0.0003 | 3/3 crop higher |
| PA | ResNet-152 | CheXMask crop - mask | 0.0220 | 0.0049 | 0.0061 | 0.0110 | 0.0096 | 3/3 crop higher |
| PA | EVA-X-Base | CheXMask crop - mask | 0.0232 | 0.0228 | 0.0258 | 0.0239 | 0.0016 | 3/3 crop higher |
| PA | Two-backbone mean | CheXMask crop - mask | 0.0226 | 0.0139 | 0.0159 | 0.0175 | 0.0046 | 3/3 crop higher |

## C. Backbone-specific ΔAUPRC versus raw

| View | Backbone | Preprocessing | Seed 42 | Seed 123 | Seed 2026 | Mean ΔAUPRC | SD | Direction |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| AP | ResNet-152 | MedSAM3 crop | 0.0001 | 0.0058 | 0.0009 | 0.0022 | 0.0031 | 3/3 higher than raw |
| AP | ResNet-152 | CheXMask crop | 0.0026 | 0.0079 | 0.0025 | 0.0044 | 0.0031 | 3/3 higher than raw |
| AP | ResNet-152 | MedSAM3 mask | -0.0045 | -0.0002 | -0.0068 | -0.0038 | 0.0033 | 3/3 lower than raw |
| AP | ResNet-152 | CheXMask mask | -0.0014 | 0.0011 | -0.0031 | -0.0011 | 0.0021 | 2/3 lower than raw |
| AP | EVA-X-Base | MedSAM3 crop | 0.0010 | -0.0008 | 0.0023 | 0.0008 | 0.0016 | 2/3 higher than raw |
| AP | EVA-X-Base | CheXMask crop | 0.0015 | -0.0020 | 0.0014 | 0.0003 | 0.0020 | 2/3 higher than raw |
| AP | EVA-X-Base | MedSAM3 mask | -0.0072 | -0.0091 | -0.0053 | -0.0072 | 0.0019 | 3/3 lower than raw |
| AP | EVA-X-Base | CheXMask mask | -0.0067 | -0.0071 | -0.0029 | -0.0056 | 0.0023 | 3/3 lower than raw |
| PA | ResNet-152 | MedSAM3 crop | 0.0091 | 0.0079 | -0.0018 | 0.0051 | 0.0060 | 2/3 higher than raw |
| PA | ResNet-152 | CheXMask crop | 0.0070 | -0.0005 | -0.0047 | 0.0006 | 0.0059 | 2/3 lower than raw |
| PA | ResNet-152 | MedSAM3 mask | -0.0058 | -0.0180 | -0.0179 | -0.0139 | 0.0070 | 3/3 lower than raw |
| PA | ResNet-152 | CheXMask mask | -0.0160 | -0.0029 | -0.0073 | -0.0087 | 0.0067 | 3/3 lower than raw |
| PA | EVA-X-Base | MedSAM3 crop | -0.0054 | 0.0011 | -0.0046 | -0.0029 | 0.0035 | 2/3 lower than raw |
| PA | EVA-X-Base | CheXMask crop | -0.0052 | 0.0011 | 0.0033 | -0.0003 | 0.0044 | 2/3 higher than raw |
| PA | EVA-X-Base | MedSAM3 mask | -0.0429 | -0.0341 | -0.0288 | -0.0353 | 0.0071 | 3/3 lower than raw |
| PA | EVA-X-Base | CheXMask mask | -0.0388 | -0.0329 | -0.0311 | -0.0343 | 0.0040 | 3/3 lower than raw |

## D. Absolute outer-test AUROC by training seed

| View | Backbone | Input | Seed 42 | Seed 123 | Seed 2026 | Mean AUROC | SD |
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
