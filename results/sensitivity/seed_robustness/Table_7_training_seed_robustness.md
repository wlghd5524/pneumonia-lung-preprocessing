# Table 7. Independent training-seed robustness of paired preprocessing effects. Each entry is the mean paired outer-test ΔAUROC of ResNet-152 and EVA-X-Base. Patient partitions were fixed. Training seeds were 42, 123, and 2026. The Seed 42, Seed 123, and Seed 2026 columns are within-seed differences (preprocessed minus raw). Mean ΔAUROC and SD summarize those three seeds; SD divides by n−1. Positive values indicate higher AUROC than raw. Direction counts how many of the three seeds have the stated sign. Backbone-specific AUROC, crop-minus-mask, AUPRC, and absolute AUROC results are in Supplementary Table S10.

| View | Preprocessing | Seed 42 | Seed 123 | Seed 2026 | Mean ΔAUROC | SD | Direction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| AP | MedSAM3 crop | +0.0021 | +0.0015 | +0.0031 | +0.0022 | 0.0008 | 3/3 higher than raw |
| AP | CheXMask crop | +0.0026 | +0.0020 | +0.0027 | +0.0024 | 0.0004 | 3/3 higher than raw |
| AP | MedSAM3 mask | -0.0049 | -0.0064 | -0.0054 | -0.0056 | 0.0008 | 3/3 lower than raw |
| AP | CheXMask mask | -0.0026 | -0.0034 | -0.0021 | -0.0027 | 0.0006 | 3/3 lower than raw |
| PA | MedSAM3 crop | -0.0017 | +0.0013 | -0.0015 | -0.0006 | 0.0017 | 2/3 lower than raw |
| PA | CheXMask crop | +0.0003 | -0.0012 | +0.0002 | -0.0002 | 0.0008 | 2/3 higher than raw |
| PA | MedSAM3 mask | -0.0186 | -0.0187 | -0.0175 | -0.0183 | 0.0007 | 3/3 lower than raw |
| PA | CheXMask mask | -0.0222 | -0.0150 | -0.0157 | -0.0177 | 0.0040 | 3/3 lower than raw |

AP uses 3,280 images from 1,763 patients; PA uses 2,521 images from 1,743 patients. Only ResNet-152 and EVA-X-Base were retrained. The caution regarding architectural superiority is unchanged.
