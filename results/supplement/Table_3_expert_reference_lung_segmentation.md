# Table 3. Independent expert-reference evaluation of lung segmentation in the PA and AP reference sets

| View | Method | N | Dice, mean ± SD | IoU, mean ± SD | HD95, px, mean ± SD |
| --- | --- | ---: | ---: | ---: | ---: |
| PA | MedSAM3 | 200 | 0.949 ± 0.035 | 0.904 ± 0.058 | 102.5 ± 141.2 |
| PA | CheXMask | 200 | 0.938 ± 0.033 | 0.886 ± 0.056 | 90.6 ± 90.4 |
| AP | MedSAM3 | 100 | 0.930 ± 0.054 | 0.873 ± 0.083 | 153.8 ± 227.3 |
| AP | CheXMask | 100 | 0.915 ± 0.060 | 0.849 ± 0.091 | 119.8 ± 92.5 |

PA rows: `Table_S2_pa200_expert_reference_summary.json` (`all`). AP rows: `Table_S2_ap_expert_reference_summary.json`. HD95 is the symmetric boundary-to-boundary 95th-percentile Hausdorff distance (`cxr-segmentation-validation/evaluation_protocol.md`).
