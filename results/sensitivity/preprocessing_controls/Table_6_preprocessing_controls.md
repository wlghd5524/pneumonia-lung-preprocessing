# Table 6. Controlled classification experiments for separating geometric and masking-related preprocessing effects. Values represent mean paired differences across ResNet-152 and EVA-X-Base on the matched outer-test cohorts, with 95% confidence intervals estimated using 2,000 subject-level paired cluster-bootstrap replicates. Positive values indicate higher performance for the first condition in each contrast. Detailed backbone-specific AUROC and AUPRC results are provided in Supplementary Table S9.

| Contrast | AP ΔAUROC (95% CI) | PA ΔAUROC (95% CI) | Interpretation |
| --- | ---: | ---: | --- |
| Study crop − Matched center crop | -0.0002 (-0.0031, +0.0027) | -0.0004 (-0.0053, +0.0047) | Little effect of crop location |
| Study crop − Hard mask | +0.0070 (+0.0018, +0.0124) | +0.0169 (+0.0077, +0.0263) | Crop higher than hard masking |
| Study crop − Soft mask | +0.0076 (+0.0021, +0.0130) | +0.0164 (+0.0070, +0.0265) | Boundary smoothing does not close the gap |
| Hard mask − Soft mask | +0.0006 (-0.0029, +0.0041) | -0.0005 (-0.0068, +0.0058) | Limited effect of the sharp boundary alone |
| Hard mask − Raw | -0.0049 (-0.0097, +0.0002) | -0.0186 (-0.0279, -0.0091) | Masking lower than raw, particularly on PA |
| Soft mask − Raw | -0.0055 (-0.0107, +0.0000) | -0.0181 (-0.0280, -0.0090) | Decrease remains after boundary softening |

AP uses 3,280 images from 1,763 patients; PA uses 2,521 images from 1,743 patients. The matched center crop uses the width and height of the corresponding MedSAM3 study-crop window after its standard padding and before black-border trimming and square padding. Intervals are not adjusted for multiple comparisons.
