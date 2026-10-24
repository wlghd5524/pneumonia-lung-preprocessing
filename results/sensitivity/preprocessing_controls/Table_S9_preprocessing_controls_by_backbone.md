# Table S9. Backbone-specific paired differences for the preprocessing controls

Each row represents one outer-test contrast for ResNet-152 or EVA-X-Base. Δ denotes the first-named condition minus the second-named condition. Ninety-five percent confidence intervals and unadjusted two-sided p values were estimated using 2000 subject-level paired cluster-bootstrap replicates. All comparisons were fully paired on the same outer-test radiographs and labels. The two-backbone mean estimates for the principal contrasts are summarized in Table 6.

| View | Backbone | Contrast | AUROC, first | AUROC, second | ΔAUROC (95% CI) | ΔAUROC p | ΔAUPRC (95% CI) | ΔAUPRC p |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AP | ResNet-152 | Study crop − Centered crop | 0.7893 | 0.7897 | -0.0004 (-0.0053, +0.0046) | 0.870 | -0.0015 (-0.0078, +0.0047) | 0.659 |
| AP | ResNet-152 | Study crop − Hard mask | 0.7893 | 0.7826 | +0.0067 (-0.0010, +0.0146) | 0.106 | +0.0046 (-0.0053, +0.0149) | 0.353 |
| AP | ResNet-152 | Study crop − Soft mask | 0.7893 | 0.7826 | +0.0067 (-0.0011, +0.0142) | 0.096 | +0.0104 (-0.0004, +0.0215) | 0.063 |
| AP | ResNet-152 | Hard mask − Soft mask | 0.7826 | 0.7826 | 0.0000 (-0.0060, +0.0059) | 0.993 | +0.0059 (-0.0019, +0.0137) | 0.136 |
| AP | ResNet-152 | Hard mask − Raw | 0.7826 | 0.7860 | -0.0034 (-0.0111, +0.0042) | 0.392 | -0.0045 (-0.0139, +0.0044) | 0.324 |
| AP | ResNet-152 | Soft mask − Raw | 0.7826 | 0.7860 | -0.0034 (-0.0104, +0.0041) | 0.374 | -0.0104 (-0.0205, -0.0008) | 0.033 |
| AP | ResNet-152 | Study crop − Raw | 0.7893 | 0.7860 | +0.0033 (-0.0019, +0.0088) | 0.228 | +0.0001 (-0.0083, +0.0079) | 0.984 |
| AP | ResNet-152 | Study crop − 0% margin crop | 0.7893 | 0.7897 | -0.0004 (-0.0054, +0.0046) | 0.877 | -0.0032 (-0.0095, +0.0033) | 0.344 |
| AP | ResNet-152 | Study crop − 10% margin crop | 0.7893 | 0.7917 | -0.0024 (-0.0070, +0.0022) | 0.296 | -0.0067 (-0.0125, -0.0011) | 0.020 |
| AP | ResNet-152 | Study crop − 20% margin crop | 0.7893 | 0.7872 | +0.0021 (-0.0029, +0.0075) | 0.422 | +0.0006 (-0.0058, +0.0074) | 0.839 |
| AP | ResNet-152 | Centered crop − Raw | 0.7897 | 0.7860 | +0.0037 (-0.0021, +0.0094) | 0.197 | +0.0015 (-0.0059, +0.0088) | 0.668 |
| AP | ResNet-152 | Centered crop − Hard mask | 0.7897 | 0.7826 | +0.0071 (-0.0009, +0.0148) | 0.074 | +0.0060 (-0.0034, +0.0154) | 0.217 |
| AP | ResNet-152 | 0% margin crop − Raw | 0.7897 | 0.7860 | +0.0037 (-0.0023, +0.0096) | 0.228 | +0.0033 (-0.0056, +0.0117) | 0.454 |
| AP | ResNet-152 | 10% margin crop − Raw | 0.7917 | 0.7860 | +0.0057 (+0.0007, +0.0108) | 0.026 | +0.0068 (-0.0004, +0.0136) | 0.066 |
| AP | ResNet-152 | 20% margin crop − Raw | 0.7872 | 0.7860 | +0.0012 (-0.0033, +0.0060) | 0.653 | -0.0005 (-0.0069, +0.0060) | 0.876 |
| AP | ResNet-152 | 0% margin crop − 20% margin crop | 0.7897 | 0.7872 | +0.0025 (-0.0032, +0.0086) | 0.393 | +0.0038 (-0.0040, +0.0115) | 0.340 |
| AP | EVA-X-Base | Study crop − Centered crop | 0.8137 | 0.8137 | 0.0000 (-0.0031, +0.0033) | 0.991 | -0.0010 (-0.0051, +0.0030) | 0.607 |
| AP | EVA-X-Base | Study crop − Hard mask | 0.8137 | 0.8064 | +0.0073 (+0.0014, +0.0135) | 0.023 | +0.0082 (+0.0011, +0.0156) | 0.021 |
| AP | EVA-X-Base | Study crop − Soft mask | 0.8137 | 0.8053 | +0.0084 (+0.0020, +0.0150) | 0.010 | +0.0077 (+0.0003, +0.0154) | 0.043 |
| AP | EVA-X-Base | Hard mask − Soft mask | 0.8064 | 0.8053 | +0.0011 (-0.0026, +0.0048) | 0.561 | -0.0005 (-0.0048, +0.0039) | 0.791 |
| AP | EVA-X-Base | Hard mask − Raw | 0.8064 | 0.8128 | -0.0064 (-0.0118, -0.0010) | 0.025 | -0.0072 (-0.0136, -0.0004) | 0.039 |
| AP | EVA-X-Base | Soft mask − Raw | 0.8053 | 0.8128 | -0.0075 (-0.0134, -0.0013) | 0.016 | -0.0067 (-0.0138, +0.0003) | 0.067 |
| AP | EVA-X-Base | Study crop − Raw | 0.8137 | 0.8128 | +0.0009 (-0.0027, +0.0043) | 0.601 | +0.0010 (-0.0037, +0.0061) | 0.701 |
| AP | EVA-X-Base | Study crop − 0% margin crop | 0.8137 | 0.8134 | +0.0003 (-0.0030, +0.0036) | 0.876 | -0.0023 (-0.0069, +0.0025) | 0.364 |
| AP | EVA-X-Base | Study crop − 10% margin crop | 0.8137 | 0.8147 | -0.0010 (-0.0037, +0.0017) | 0.492 | -0.0006 (-0.0048, +0.0035) | 0.751 |
| AP | EVA-X-Base | Study crop − 20% margin crop | 0.8137 | 0.8167 | -0.0030 (-0.0061, +0.0005) | 0.083 | -0.0026 (-0.0065, +0.0017) | 0.237 |
| AP | EVA-X-Base | Centered crop − Raw | 0.8137 | 0.8128 | +0.0009 (-0.0029, +0.0048) | 0.608 | +0.0020 (-0.0023, +0.0062) | 0.350 |
| AP | EVA-X-Base | Centered crop − Hard mask | 0.8137 | 0.8064 | +0.0073 (+0.0015, +0.0126) | 0.009 | +0.0092 (+0.0025, +0.0159) | 0.005 |
| AP | EVA-X-Base | 0% margin crop − Raw | 0.8134 | 0.8128 | +0.0006 (-0.0036, +0.0049) | 0.802 | +0.0033 (-0.0019, +0.0085) | 0.230 |
| AP | EVA-X-Base | 10% margin crop − Raw | 0.8147 | 0.8128 | +0.0019 (-0.0014, +0.0053) | 0.254 | +0.0016 (-0.0026, +0.0058) | 0.440 |
| AP | EVA-X-Base | 20% margin crop − Raw | 0.8167 | 0.8128 | +0.0039 (+0.0006, +0.0072) | 0.014 | +0.0036 (-0.0005, +0.0077) | 0.090 |
| AP | EVA-X-Base | 0% margin crop − 20% margin crop | 0.8134 | 0.8167 | -0.0033 (-0.0074, +0.0009) | 0.111 | -0.0003 (-0.0052, +0.0043) | 0.888 |
| PA | ResNet-152 | Study crop − Centered crop | 0.7791 | 0.7806 | -0.0015 (-0.0097, +0.0064) | 0.708 | +0.0057 (-0.0065, +0.0174) | 0.352 |
| PA | ResNet-152 | Study crop − Hard mask | 0.7791 | 0.7712 | +0.0079 (-0.0056, +0.0215) | 0.282 | +0.0149 (-0.0064, +0.0342) | 0.175 |
| PA | ResNet-152 | Study crop − Soft mask | 0.7791 | 0.7620 | +0.0171 (+0.0033, +0.0307) | 0.017 | +0.0259 (+0.0035, +0.0457) | 0.019 |
| PA | ResNet-152 | Hard mask − Soft mask | 0.7712 | 0.7620 | +0.0092 (-0.0011, +0.0198) | 0.082 | +0.0110 (-0.0030, +0.0246) | 0.133 |
| PA | ResNet-152 | Hard mask − Raw | 0.7712 | 0.7797 | -0.0085 (-0.0206, +0.0043) | 0.183 | -0.0058 (-0.0261, +0.0169) | 0.685 |
| PA | ResNet-152 | Soft mask − Raw | 0.7620 | 0.7797 | -0.0177 (-0.0316, -0.0044) | 0.010 | -0.0168 (-0.0374, +0.0071) | 0.181 |
| PA | ResNet-152 | Study crop − Raw | 0.7791 | 0.7797 | -0.0006 (-0.0097, +0.0082) | 0.890 | +0.0091 (-0.0045, +0.0218) | 0.187 |
| PA | ResNet-152 | Study crop − 0% margin crop | 0.7791 | 0.7805 | -0.0014 (-0.0096, +0.0073) | 0.775 | +0.0023 (-0.0098, +0.0154) | 0.713 |
| PA | ResNet-152 | Study crop − 10% margin crop | 0.7791 | 0.7723 | +0.0068 (-0.0016, +0.0151) | 0.116 | +0.0125 (+0.0008, +0.0245) | 0.038 |
| PA | ResNet-152 | Study crop − 20% margin crop | 0.7791 | 0.7865 | -0.0074 (-0.0159, +0.0012) | 0.089 | -0.0010 (-0.0154, +0.0138) | 0.889 |
| PA | ResNet-152 | Centered crop − Raw | 0.7806 | 0.7797 | +0.0009 (-0.0072, +0.0091) | 0.835 | +0.0034 (-0.0091, +0.0159) | 0.614 |
| PA | ResNet-152 | Centered crop − Hard mask | 0.7806 | 0.7712 | +0.0094 (-0.0030, +0.0224) | 0.142 | +0.0092 (-0.0099, +0.0270) | 0.372 |
| PA | ResNet-152 | 0% margin crop − Raw | 0.7805 | 0.7797 | +0.0008 (-0.0088, +0.0108) | 0.876 | +0.0067 (-0.0081, +0.0218) | 0.376 |
| PA | ResNet-152 | 10% margin crop − Raw | 0.7723 | 0.7797 | -0.0074 (-0.0155, +0.0005) | 0.067 | -0.0035 (-0.0153, +0.0092) | 0.582 |
| PA | ResNet-152 | 20% margin crop − Raw | 0.7865 | 0.7797 | +0.0068 (-0.0010, +0.0151) | 0.083 | +0.0101 (-0.0027, +0.0222) | 0.120 |
| PA | ResNet-152 | 0% margin crop − 20% margin crop | 0.7805 | 0.7865 | -0.0060 (-0.0158, +0.0037) | 0.227 | -0.0034 (-0.0198, +0.0125) | 0.699 |
| PA | EVA-X-Base | Study crop − Centered crop | 0.8233 | 0.8226 | +0.0007 (-0.0042, +0.0062) | 0.756 | +0.0004 (-0.0072, +0.0080) | 0.896 |
| PA | EVA-X-Base | Study crop − Hard mask | 0.8233 | 0.7975 | +0.0258 (+0.0159, +0.0362) | <0.001 | +0.0375 (+0.0217, +0.0535) | <0.001 |
| PA | EVA-X-Base | Study crop − Soft mask | 0.8233 | 0.8077 | +0.0156 (+0.0053, +0.0254) | 0.002 | +0.0269 (+0.0104, +0.0428) | 0.001 |
| PA | EVA-X-Base | Hard mask − Soft mask | 0.7975 | 0.8077 | -0.0102 (-0.0168, -0.0037) | <0.001 | -0.0106 (-0.0192, -0.0019) | 0.024 |
| PA | EVA-X-Base | Hard mask − Raw | 0.7975 | 0.8261 | -0.0286 (-0.0395, -0.0187) | <0.001 | -0.0429 (-0.0593, -0.0272) | <0.001 |
| PA | EVA-X-Base | Soft mask − Raw | 0.8077 | 0.8261 | -0.0184 (-0.0288, -0.0083) | <0.001 | -0.0323 (-0.0472, -0.0174) | <0.001 |
| PA | EVA-X-Base | Study crop − Raw | 0.8233 | 0.8261 | -0.0028 (-0.0084, +0.0034) | 0.350 | -0.0054 (-0.0138, +0.0028) | 0.188 |
| PA | EVA-X-Base | Study crop − 0% margin crop | 0.8233 | 0.8242 | -0.0009 (-0.0068, +0.0049) | 0.770 | +0.0010 (-0.0072, +0.0087) | 0.814 |
| PA | EVA-X-Base | Study crop − 10% margin crop | 0.8233 | 0.8222 | +0.0011 (-0.0039, +0.0064) | 0.658 | +0.0007 (-0.0071, +0.0084) | 0.827 |
| PA | EVA-X-Base | Study crop − 20% margin crop | 0.8233 | 0.8268 | -0.0035 (-0.0087, +0.0019) | 0.194 | -0.0065 (-0.0138, +0.0008) | 0.087 |
| PA | EVA-X-Base | Centered crop − Raw | 0.8226 | 0.8261 | -0.0035 (-0.0090, +0.0022) | 0.203 | -0.0058 (-0.0140, +0.0027) | 0.174 |
| PA | EVA-X-Base | Centered crop − Hard mask | 0.8226 | 0.7975 | +0.0251 (+0.0148, +0.0352) | <0.001 | +0.0371 (+0.0214, +0.0534) | <0.001 |
| PA | EVA-X-Base | 0% margin crop − Raw | 0.8242 | 0.8261 | -0.0019 (-0.0090, +0.0052) | 0.583 | -0.0064 (-0.0146, +0.0026) | 0.165 |
| PA | EVA-X-Base | 10% margin crop − Raw | 0.8222 | 0.8261 | -0.0039 (-0.0093, +0.0013) | 0.136 | -0.0061 (-0.0140, +0.0013) | 0.121 |
| PA | EVA-X-Base | 20% margin crop − Raw | 0.8268 | 0.8261 | +0.0007 (-0.0039, +0.0055) | 0.790 | +0.0012 (-0.0055, +0.0077) | 0.735 |
| PA | EVA-X-Base | 0% margin crop − 20% margin crop | 0.8242 | 0.8268 | -0.0026 (-0.0089, +0.0040) | 0.436 | -0.0075 (-0.0161, +0.0013) | 0.097 |

AP: 3280 radiographs from 1763 patients; PA: 2521 radiographs from 1743 patients. All comparisons were fully paired on the same outer-test radiographs and labels. Δ denotes the first-named condition minus the second-named condition. AUROC values are reported to four decimal places, and the displayed ΔAUROC point estimates were calculated from these four-decimal values for numerical consistency; confidence intervals were estimated from the original unrounded predictions using 2000 subject-level paired cluster-bootstrap replicates. p values were not adjusted for multiple comparisons and are reported to three decimal places; values below 0.001 are reported as p < 0.001.
