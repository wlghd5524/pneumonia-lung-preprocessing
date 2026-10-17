# Table 3. Independent expert-reference evaluation of lung segmentation in the PA and AP reference sets

| View | Method | N | Dice, mean ± SD | IoU, mean ± SD | HD95, px, mean ± SD |
| --- | --- | ---: | ---: | ---: | ---: |
| PA | MedSAM3 | 200 | 0.949 ± 0.035 | 0.904 ± 0.058 | 75.4 ± 120.7 |
| PA | CheXMask | 200 | 0.938 ± 0.033 | 0.886 ± 0.056 | 68.4 ± 76.0 |
| AP | MedSAM3 | 100 | 0.930 ± 0.054 | 0.873 ± 0.083 | 115.9 ± 203.3 |
| AP | CheXMask | 100 | 0.915 ± 0.060 | 0.849 ± 0.091 | 92.6 ± 75.9 |

PA rows: `Table_S2_pa200_expert_reference_summary.json` (`all`). AP rows: `Table_S2_ap_expert_reference_summary.json`. The recomputed PA MedSAM3 HD95 SD is 120.6 px; see the notes under Table S2.
