# Supplementary Table S11. Clustered DeLong test statistics for ΔAUROC versus raw

SE is the clustered DeLong standard error of ΔAUROC (Obuchowski, 1997) and z = ΔAUROC / SE. Unclustered SE is the paired DeLong standard error with every radiograph treated as independent. Bootstrap SE is the standard deviation of the 20,000 paired subject-level cluster bootstrap replicates used for the 95% CIs in Table 4. P values match Table 4.

| View | Backbone | Preprocessing | ΔAUROC | SE | z | Unadjusted p | Holm-adjusted p | Unclustered SE | Bootstrap SE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AP | ResNet-152 | MedSAM3 mask | -0.0034 | 0.0039 | -0.85 | 0.393 | 1.000 | 0.0037 | 0.0039 |
| AP | ResNet-152 | MedSAM3 crop | +0.0034 | 0.0028 | +1.18 | 0.238 | 1.000 | 0.0027 | 0.0029 |
| AP | ResNet-152 | CheXMask mask | -0.0001 | 0.0040 | -0.01 | 0.989 | 1.000 | 0.0037 | 0.0040 |
| AP | ResNet-152 | CheXMask crop | +0.0049 | 0.0026 | +1.89 | 0.059 | 1.000 | 0.0025 | 0.0026 |
| AP | DenseNet-201 | MedSAM3 mask | -0.0060 | 0.0038 | -1.58 | 0.114 | 1.000 | 0.0036 | 0.0038 |
| AP | DenseNet-201 | MedSAM3 crop | -0.0006 | 0.0026 | -0.22 | 0.828 | 1.000 | 0.0024 | 0.0025 |
| AP | DenseNet-201 | CheXMask mask | -0.0059 | 0.0037 | -1.58 | 0.114 | 1.000 | 0.0035 | 0.0037 |
| AP | DenseNet-201 | CheXMask crop | -0.0007 | 0.0027 | -0.24 | 0.807 | 1.000 | 0.0026 | 0.0027 |
| AP | ConvNeXt V2-Base | MedSAM3 mask | -0.0054 | 0.0030 | -1.81 | 0.070 | 1.000 | 0.0029 | 0.0030 |
| AP | ConvNeXt V2-Base | MedSAM3 crop | +0.0042 | 0.0018 | +2.30 | 0.021 | 0.988 | 0.0018 | 0.0019 |
| AP | ConvNeXt V2-Base | CheXMask mask | -0.0071 | 0.0032 | -2.19 | 0.028 | 1.000 | 0.0031 | 0.0033 |
| AP | ConvNeXt V2-Base | CheXMask crop | -0.0016 | 0.0020 | -0.80 | 0.423 | 1.000 | 0.0020 | 0.0020 |
| AP | Swin V2-Base | MedSAM3 mask | -0.0043 | 0.0031 | -1.39 | 0.166 | 1.000 | 0.0031 | 0.0031 |
| AP | Swin V2-Base | MedSAM3 crop | +0.0013 | 0.0031 | +0.43 | 0.665 | 1.000 | 0.0030 | 0.0031 |
| AP | Swin V2-Base | CheXMask mask | -0.0011 | 0.0032 | -0.34 | 0.732 | 1.000 | 0.0032 | 0.0032 |
| AP | Swin V2-Base | CheXMask crop | +0.0029 | 0.0031 | +0.93 | 0.352 | 1.000 | 0.0030 | 0.0031 |
| AP | DINOv3-Base | MedSAM3 mask | -0.0055 | 0.0026 | -2.09 | 0.036 | 1.000 | 0.0025 | 0.0026 |
| AP | DINOv3-Base | MedSAM3 crop | +0.0036 | 0.0022 | +1.65 | 0.098 | 1.000 | 0.0020 | 0.0022 |
| AP | DINOv3-Base | CheXMask mask | -0.0063 | 0.0026 | -2.39 | 0.017 | 0.806 | 0.0026 | 0.0026 |
| AP | DINOv3-Base | CheXMask crop | +0.0017 | 0.0022 | +0.79 | 0.430 | 1.000 | 0.0021 | 0.0022 |
| AP | EVA-X-Base | MedSAM3 mask | -0.0064 | 0.0027 | -2.35 | 0.019 | 0.882 | 0.0026 | 0.0027 |
| AP | EVA-X-Base | MedSAM3 crop | +0.0009 | 0.0018 | +0.53 | 0.596 | 1.000 | 0.0017 | 0.0018 |
| AP | EVA-X-Base | CheXMask mask | -0.0052 | 0.0026 | -2.04 | 0.041 | 1.000 | 0.0024 | 0.0026 |
| AP | EVA-X-Base | CheXMask crop | +0.0003 | 0.0018 | +0.16 | 0.875 | 1.000 | 0.0017 | 0.0018 |
| AP | RAD-DINO | MedSAM3 mask | -0.0087 | 0.0031 | -2.81 | 0.005 | 0.258 | 0.0029 | 0.0031 |
| AP | RAD-DINO | MedSAM3 crop | +0.0021 | 0.0018 | +1.21 | 0.225 | 1.000 | 0.0017 | 0.0018 |
| AP | RAD-DINO | CheXMask mask | -0.0068 | 0.0030 | -2.29 | 0.022 | 0.998 | 0.0028 | 0.0030 |
| AP | RAD-DINO | CheXMask crop | -0.0012 | 0.0016 | -0.76 | 0.448 | 1.000 | 0.0016 | 0.0016 |
| PA | ResNet-152 | MedSAM3 mask | -0.0085 | 0.0065 | -1.32 | 0.187 | 1.000 | 0.0062 | 0.0065 |
| PA | ResNet-152 | MedSAM3 crop | -0.0006 | 0.0045 | -0.13 | 0.893 | 1.000 | 0.0045 | 0.0045 |
| PA | ResNet-152 | CheXMask mask | -0.0190 | 0.0071 | -2.68 | 0.007 | 0.364 | 0.0065 | 0.0071 |
| PA | ResNet-152 | CheXMask crop | +0.0030 | 0.0045 | +0.66 | 0.509 | 1.000 | 0.0044 | 0.0046 |
| PA | DenseNet-201 | MedSAM3 mask | -0.0061 | 0.0065 | -0.94 | 0.348 | 1.000 | 0.0062 | 0.0065 |
| PA | DenseNet-201 | MedSAM3 crop | -0.0050 | 0.0041 | -1.22 | 0.224 | 1.000 | 0.0041 | 0.0041 |
| PA | DenseNet-201 | CheXMask mask | -0.0039 | 0.0063 | -0.62 | 0.539 | 1.000 | 0.0061 | 0.0063 |
| PA | DenseNet-201 | CheXMask crop | -0.0067 | 0.0041 | -1.61 | 0.107 | 1.000 | 0.0040 | 0.0041 |
| PA | ConvNeXt V2-Base | MedSAM3 mask | -0.0106 | 0.0056 | -1.90 | 0.057 | 1.000 | 0.0052 | 0.0056 |
| PA | ConvNeXt V2-Base | MedSAM3 crop | -0.0017 | 0.0034 | -0.49 | 0.627 | 1.000 | 0.0033 | 0.0034 |
| PA | ConvNeXt V2-Base | CheXMask mask | -0.0113 | 0.0055 | -2.04 | 0.041 | 1.000 | 0.0052 | 0.0056 |
| PA | ConvNeXt V2-Base | CheXMask crop | -0.0055 | 0.0031 | -1.79 | 0.074 | 1.000 | 0.0030 | 0.0031 |
| PA | Swin V2-Base | MedSAM3 mask | -0.0070 | 0.0059 | -1.20 | 0.231 | 1.000 | 0.0058 | 0.0059 |
| PA | Swin V2-Base | MedSAM3 crop | +0.0059 | 0.0050 | +1.18 | 0.239 | 1.000 | 0.0048 | 0.0050 |
| PA | Swin V2-Base | CheXMask mask | -0.0085 | 0.0057 | -1.49 | 0.136 | 1.000 | 0.0055 | 0.0058 |
| PA | Swin V2-Base | CheXMask crop | -0.0023 | 0.0046 | -0.51 | 0.612 | 1.000 | 0.0046 | 0.0046 |
| PA | DINOv3-Base | MedSAM3 mask | -0.0232 | 0.0054 | -4.32 | 1.57 × 10⁻⁵ | 8.50 × 10⁻⁴ | 0.0051 | 0.0054 |
| PA | DINOv3-Base | MedSAM3 crop | +0.0020 | 0.0034 | +0.58 | 0.561 | 1.000 | 0.0032 | 0.0034 |
| PA | DINOv3-Base | CheXMask mask | -0.0189 | 0.0052 | -3.62 | 2.94 × 10⁻⁴ | 0.016 | 0.0050 | 0.0052 |
| PA | DINOv3-Base | CheXMask crop | -0.0018 | 0.0038 | -0.46 | 0.646 | 1.000 | 0.0037 | 0.0038 |
| PA | EVA-X-Base | MedSAM3 mask | -0.0286 | 0.0053 | -5.43 | 5.54 × 10⁻⁸ | 3.10 × 10⁻⁶ | 0.0051 | 0.0052 |
| PA | EVA-X-Base | MedSAM3 crop | -0.0028 | 0.0030 | -0.92 | 0.358 | 1.000 | 0.0030 | 0.0030 |
| PA | EVA-X-Base | CheXMask mask | -0.0255 | 0.0052 | -4.91 | 9.10 × 10⁻⁷ | 5.01 × 10⁻⁵ | 0.0050 | 0.0052 |
| PA | EVA-X-Base | CheXMask crop | -0.0023 | 0.0029 | -0.80 | 0.426 | 1.000 | 0.0029 | 0.0029 |
| PA | RAD-DINO | MedSAM3 mask | -0.0144 | 0.0052 | -2.77 | 0.006 | 0.290 | 0.0050 | 0.0051 |
| PA | RAD-DINO | MedSAM3 crop | +0.0018 | 0.0023 | +0.75 | 0.454 | 1.000 | 0.0022 | 0.0024 |
| PA | RAD-DINO | CheXMask mask | -0.0126 | 0.0051 | -2.47 | 0.014 | 0.663 | 0.0049 | 0.0051 |
| PA | RAD-DINO | CheXMask crop | +0.0029 | 0.0023 | +1.25 | 0.211 | 1.000 | 0.0021 | 0.0023 |
