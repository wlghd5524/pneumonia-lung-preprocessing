# Table 7. Independent training-seed robustness of paired preprocessing effects

Each entry represents the mean paired outer-test ΔAUROC across ResNet-152 and EVA-X-Base. Patient partitions were held fixed, and training seeds 42, 123, and 2026 were evaluated. Seed-specific values represent preprocessing minus raw within the same acquisition view, backbone, and training seed. Mean ΔAUROC and SD summarize the three training seeds, with SD calculated using the sample convention (n − 1). Positive values indicate higher AUROC than raw input. Detailed backbone-specific results are provided in Supplementary Table S10.

| View | Preprocessing | Seed 42 | Seed 123 | Seed 2026 | Mean ΔAUROC | SD | Direction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| AP | MedSAM3 crop | +0.0021 | +0.0015 | +0.0031 | +0.0022 | 0.0008 | 3/3 higher than raw |
| AP | CheXMask crop | +0.0026 | +0.0020 | +0.0027 | +0.0024 | 0.0004 | 3/3 higher than raw |
| AP | MedSAM3 mask | −0.0049 | −0.0064 | −0.0054 | −0.0056 | 0.0008 | 3/3 lower than raw |
| AP | CheXMask mask | −0.0026 | −0.0034 | −0.0021 | −0.0027 | 0.0006 | 3/3 lower than raw |
| PA | MedSAM3 crop | −0.0017 | +0.0013 | −0.0015 | −0.0006 | 0.0017 | 2/3 lower than raw |
| PA | CheXMask crop | +0.0003 | −0.0012 | +0.0002 | −0.0002 | 0.0008 | 2/3 higher than raw |
| PA | MedSAM3 mask | −0.0186 | −0.0187 | −0.0175 | −0.0183 | 0.0007 | 3/3 lower than raw |
| PA | CheXMask mask | −0.0222 | −0.0150 | −0.0157 | −0.0177 | 0.0040 | 3/3 lower than raw |
