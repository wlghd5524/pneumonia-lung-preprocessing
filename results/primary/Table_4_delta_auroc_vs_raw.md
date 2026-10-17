# Table 4. Per-backbone ΔAUROC versus raw input

Paired comparisons of each lung preprocessing condition with raw input on the held-out internal test set for each backbone and acquisition view. ΔAUROC is the image-level AUROC of the preprocessed model minus that of the corresponding raw-input model, calculated from unrounded estimates. Pointwise 95% CIs are percentile intervals obtained from 20,000 paired subject-level cluster bootstrap replicates and are not adjusted for multiple comparisons. Unadjusted two-sided p-values were calculated using a subject-clustered DeLong-type test that accounts for paired predictions on the same radiographs and correlations among radiographs from the same patient. The Holm procedure was applied jointly to the 56 preprocessing-versus-raw comparisons using unrounded p-values. Both unadjusted and Holm-adjusted p-values are reported.

| View | Backbone | Preprocessing | Raw AUROC | Preprocessed AUROC | ΔAUROC vs. Raw [95% CI] | Unadjusted p | Holm-adjusted p |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| AP | ResNet-152 | MedSAM3 mask | 0.7860 | 0.7826 | -0.0034 [-0.0110, +0.0044] | 0.393 | 1.000 |
| AP | ResNet-152 | MedSAM3 crop | 0.7860 | 0.7893 | +0.0034 [-0.0022, +0.0090] | 0.238 | 1.000 |
| AP | ResNet-152 | CheXMask mask | 0.7860 | 0.7859 | -0.0001 [-0.0078, +0.0078] | 0.989 | 1.000 |
| AP | ResNet-152 | CheXMask crop | 0.7860 | 0.7908 | +0.0049 [-0.0003, +0.0099] | 0.059 | 1.000 |
| AP | DenseNet-201 | MedSAM3 mask | 0.7960 | 0.7900 | -0.0060 [-0.0134, +0.0014] | 0.114 | 1.000 |
| AP | DenseNet-201 | MedSAM3 crop | 0.7960 | 0.7955 | -0.0006 [-0.0055, +0.0044] | 0.828 | 1.000 |
| AP | DenseNet-201 | CheXMask mask | 0.7960 | 0.7902 | -0.0059 [-0.0131, +0.0014] | 0.114 | 1.000 |
| AP | DenseNet-201 | CheXMask crop | 0.7960 | 0.7954 | -0.0007 [-0.0060, +0.0048] | 0.807 | 1.000 |
| AP | ConvNeXt V2-Base | MedSAM3 mask | 0.8029 | 0.7975 | -0.0054 [-0.0114, +0.0004] | 0.070 | 1.000 |
| AP | ConvNeXt V2-Base | MedSAM3 crop | 0.8029 | 0.8071 | +0.0042 [+0.0006, +0.0079] | 0.021 | 0.988 |
| AP | ConvNeXt V2-Base | CheXMask mask | 0.8029 | 0.7958 | -0.0071 [-0.0136, -0.0007] | 0.028 | 1.000 |
| AP | ConvNeXt V2-Base | CheXMask crop | 0.8029 | 0.8013 | -0.0016 [-0.0055, +0.0023] | 0.423 | 1.000 |
| AP | Swin V2-Base | MedSAM3 mask | 0.7996 | 0.7953 | -0.0043 [-0.0105, +0.0018] | 0.166 | 1.000 |
| AP | Swin V2-Base | MedSAM3 crop | 0.7996 | 0.8010 | +0.0013 [-0.0047, +0.0075] | 0.665 | 1.000 |
| AP | Swin V2-Base | CheXMask mask | 0.7996 | 0.7985 | -0.0011 [-0.0074, +0.0052] | 0.732 | 1.000 |
| AP | Swin V2-Base | CheXMask crop | 0.7996 | 0.8025 | +0.0029 [-0.0032, +0.0090] | 0.352 | 1.000 |
| AP | DINOv3-Base | MedSAM3 mask | 0.8054 | 0.8000 | -0.0055 [-0.0106, -0.0004] | 0.036 | 1.000 |
| AP | DINOv3-Base | MedSAM3 crop | 0.8054 | 0.8090 | +0.0036 [-0.0006, +0.0079] | 0.098 | 1.000 |
| AP | DINOv3-Base | CheXMask mask | 0.8054 | 0.7992 | -0.0063 [-0.0115, -0.0013] | 0.017 | 0.806 |
| AP | DINOv3-Base | CheXMask crop | 0.8054 | 0.8072 | +0.0017 [-0.0025, +0.0061] | 0.430 | 1.000 |
| AP | EVA-X-Base | MedSAM3 mask | 0.8128 | 0.8064 | -0.0064 [-0.0118, -0.0012] | 0.019 | 0.882 |
| AP | EVA-X-Base | MedSAM3 crop | 0.8128 | 0.8137 | +0.0009 [-0.0025, +0.0044] | 0.596 | 1.000 |
| AP | EVA-X-Base | CheXMask mask | 0.8128 | 0.8076 | -0.0052 [-0.0103, -0.0002] | 0.041 | 1.000 |
| AP | EVA-X-Base | CheXMask crop | 0.8128 | 0.8131 | +0.0003 [-0.0031, +0.0037] | 0.875 | 1.000 |
| AP | RAD-DINO | MedSAM3 mask | 0.8131 | 0.8045 | -0.0087 [-0.0147, -0.0026] | 0.005 | 0.258 |
| AP | RAD-DINO | MedSAM3 crop | 0.8131 | 0.8153 | +0.0021 [-0.0013, +0.0057] | 0.225 | 1.000 |
| AP | RAD-DINO | CheXMask mask | 0.8131 | 0.8064 | -0.0068 [-0.0126, -0.0010] | 0.022 | 0.998 |
| AP | RAD-DINO | CheXMask crop | 0.8131 | 0.8119 | -0.0012 [-0.0044, +0.0020] | 0.448 | 1.000 |
| PA | ResNet-152 | MedSAM3 mask | 0.7797 | 0.7712 | -0.0085 [-0.0213, +0.0042] | 0.187 | 1.000 |
| PA | ResNet-152 | MedSAM3 crop | 0.7797 | 0.7791 | -0.0006 [-0.0095, +0.0082] | 0.893 | 1.000 |
| PA | ResNet-152 | CheXMask mask | 0.7797 | 0.7607 | -0.0190 [-0.0329, -0.0054] | 0.007 | 0.364 |
| PA | ResNet-152 | CheXMask crop | 0.7797 | 0.7827 | +0.0030 [-0.0058, +0.0120] | 0.509 | 1.000 |
| PA | DenseNet-201 | MedSAM3 mask | 0.7940 | 0.7879 | -0.0061 [-0.0190, +0.0066] | 0.348 | 1.000 |
| PA | DenseNet-201 | MedSAM3 crop | 0.7940 | 0.7890 | -0.0050 [-0.0131, +0.0031] | 0.224 | 1.000 |
| PA | DenseNet-201 | CheXMask mask | 0.7940 | 0.7902 | -0.0039 [-0.0161, +0.0084] | 0.539 | 1.000 |
| PA | DenseNet-201 | CheXMask crop | 0.7940 | 0.7874 | -0.0067 [-0.0147, +0.0015] | 0.107 | 1.000 |
| PA | ConvNeXt V2-Base | MedSAM3 mask | 0.8109 | 0.8003 | -0.0106 [-0.0216, +0.0002] | 0.057 | 1.000 |
| PA | ConvNeXt V2-Base | MedSAM3 crop | 0.8109 | 0.8092 | -0.0017 [-0.0084, +0.0051] | 0.627 | 1.000 |
| PA | ConvNeXt V2-Base | CheXMask mask | 0.8109 | 0.7996 | -0.0113 [-0.0222, -0.0003] | 0.041 | 1.000 |
| PA | ConvNeXt V2-Base | CheXMask crop | 0.8109 | 0.8054 | -0.0055 [-0.0116, +0.0004] | 0.074 | 1.000 |
| PA | Swin V2-Base | MedSAM3 mask | 0.7966 | 0.7896 | -0.0070 [-0.0186, +0.0045] | 0.231 | 1.000 |
| PA | Swin V2-Base | MedSAM3 crop | 0.7966 | 0.8025 | +0.0059 [-0.0040, +0.0155] | 0.239 | 1.000 |
| PA | Swin V2-Base | CheXMask mask | 0.7966 | 0.7881 | -0.0085 [-0.0200, +0.0028] | 0.136 | 1.000 |
| PA | Swin V2-Base | CheXMask crop | 0.7966 | 0.7943 | -0.0023 [-0.0114, +0.0067] | 0.612 | 1.000 |
| PA | DINOv3-Base | MedSAM3 mask | 0.8100 | 0.7868 | -0.0232 [-0.0338, -0.0128] | 1.57 × 10⁻⁵ | 8.50 × 10⁻⁴ |
| PA | DINOv3-Base | MedSAM3 crop | 0.8100 | 0.8119 | +0.0020 [-0.0046, +0.0087] | 0.561 | 1.000 |
| PA | DINOv3-Base | CheXMask mask | 0.8100 | 0.7910 | -0.0189 [-0.0294, -0.0087] | 2.94 × 10⁻⁴ | 0.016 |
| PA | DINOv3-Base | CheXMask crop | 0.8100 | 0.8082 | -0.0018 [-0.0092, +0.0058] | 0.646 | 1.000 |
| PA | EVA-X-Base | MedSAM3 mask | 0.8261 | 0.7975 | -0.0286 [-0.0391, -0.0185] | 5.54 × 10⁻⁸ | 3.10 × 10⁻⁶ |
| PA | EVA-X-Base | MedSAM3 crop | 0.8261 | 0.8233 | -0.0028 [-0.0086, +0.0032] | 0.358 | 1.000 |
| PA | EVA-X-Base | CheXMask mask | 0.8261 | 0.8006 | -0.0255 [-0.0359, -0.0154] | 9.10 × 10⁻⁷ | 5.01 × 10⁻⁵ |
| PA | EVA-X-Base | CheXMask crop | 0.8261 | 0.8238 | -0.0023 [-0.0080, +0.0035] | 0.426 | 1.000 |
| PA | RAD-DINO | MedSAM3 mask | 0.8141 | 0.7997 | -0.0144 [-0.0245, -0.0042] | 0.006 | 0.290 |
| PA | RAD-DINO | MedSAM3 crop | 0.8141 | 0.8158 | +0.0018 [-0.0028, +0.0064] | 0.454 | 1.000 |
| PA | RAD-DINO | CheXMask mask | 0.8141 | 0.8015 | -0.0126 [-0.0226, -0.0027] | 0.014 | 0.663 |
| PA | RAD-DINO | CheXMask crop | 0.8141 | 0.8169 | +0.0029 [-0.0016, +0.0074] | 0.211 | 1.000 |
