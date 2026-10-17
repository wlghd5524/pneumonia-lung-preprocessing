# Table 6. Controlled classification experiments for separating geometric and masking-related preprocessing effects

Values represent mean paired AUROC differences across ResNet-152 and EVA-X-Base evaluated on the matched outer-test cohorts. Ninety-five percent confidence intervals were estimated using 2,000 subject-level paired cluster-bootstrap replicates. Positive values indicate higher AUROC for the first condition named in each contrast. AP comprised 3,280 radiographs from 1,763 patients and PA comprised 2,521 radiographs from 1,743 patients. Confidence intervals were not adjusted for multiple comparisons. Detailed backbone-specific AUROC and AUPRC results are provided in Supplementary Table S9.

| Contrast | AP ΔAUROC (95% CI) | PA ΔAUROC (95% CI) |
| --- | ---: | ---: |
| Study crop − Centered crop | −0.0002 (−0.0031, +0.0027) | −0.0004 (−0.0053, +0.0047) |
| Study crop − Hard mask | +0.0070 (+0.0018, +0.0124) | +0.0169 (+0.0077, +0.0263) |
| Study crop − Soft mask | +0.0076 (+0.0021, +0.0130) | +0.0164 (+0.0070, +0.0265) |
| Hard mask − Soft mask | +0.0006 (−0.0029, +0.0041) | −0.0005 (−0.0068, +0.0058) |
| Hard mask − Raw | −0.0049 (−0.0097, +0.0002) | −0.0186 (−0.0279, −0.0091) |
| Soft mask − Raw | −0.0055 (−0.0107, +0.0000) | −0.0181 (−0.0280, −0.0090) |

Centered crop refers to the centered crop with segmentation-matched dimensions. Its width and height were obtained from the corresponding MedSAM3-derived study-crop window, whereas the crop window was positioned at the image center.
