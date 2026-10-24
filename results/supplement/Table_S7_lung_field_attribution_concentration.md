# Table S7. Lung-Field Attribution Concentration on the Outer-Test Set

Metrics are computed on the five-fold mean Grad-CAM after each fold map is reprojected to original radiograph coordinates. Positive attribution is retained (`ReLU`); visualization min-max scaling is not applied. Hard-mask inputs are excluded. Results are reported independently for MedSAM3- and CheXMask-derived lung regions. Paired differences are crop minus raw on the same image. 95% CIs use a 2000-replicate subject-level cluster bootstrap (seed 42).

| View | Backbone | Input | Reference mask | N | Lung attribution mass, median [IQR] | Δ mass vs raw [95% CI] | Top-10% inside lung, median [IQR] | Δ top-10% vs raw [95% CI] |
|---|---|---|---|---:|---|---|---|---|
| AP | ResNet-152 | Raw | MedSAM3 | 3280 | 0.49 [0.38–0.59] | Ref. | 0.61 [0.49–0.73] | Ref. |
| AP | ResNet-152 | MedSAM3 crop | MedSAM3 | 3280 | 0.62 [0.54–0.68] | +0.11 [+0.11 to +0.12] | 0.77 [0.66–0.86] | +0.14 [+0.13 to +0.14] |
| AP | ResNet-152 | CheXMask crop | MedSAM3 | 3280 | 0.60 [0.51–0.68] | +0.10 [+0.09 to +0.10] | 0.74 [0.62–0.83] | +0.10 [+0.10 to +0.11] |
| AP | ResNet-152 | Raw | CheXMask | 3280 | 0.51 [0.41–0.60] | Ref. | 0.63 [0.52–0.74] | Ref. |
| AP | ResNet-152 | MedSAM3 crop | CheXMask | 3280 | 0.63 [0.56–0.69] | +0.12 [+0.11 to +0.12] | 0.78 [0.68–0.87] | +0.13 [+0.12 to +0.14] |
| AP | ResNet-152 | CheXMask crop | CheXMask | 3280 | 0.62 [0.54–0.68] | +0.10 [+0.10 to +0.11] | 0.75 [0.65–0.84] | +0.10 [+0.10 to +0.11] |
| AP | RAD-DINO | Raw | MedSAM3 | 3280 | 0.22 [0.13–0.32] | Ref. | 0.24 [0.07–0.46] | Ref. |
| AP | RAD-DINO | MedSAM3 crop | MedSAM3 | 3280 | 0.34 [0.25–0.43] | +0.11 [+0.10 to +0.11] | 0.37 [0.21–0.55] | +0.10 [+0.09 to +0.10] |
| AP | RAD-DINO | CheXMask crop | MedSAM3 | 3280 | 0.33 [0.24–0.42] | +0.10 [+0.10 to +0.10] | 0.38 [0.21–0.57] | +0.10 [+0.10 to +0.11] |
| AP | RAD-DINO | Raw | CheXMask | 3280 | 0.24 [0.14–0.33] | Ref. | 0.26 [0.08–0.47] | Ref. |
| AP | RAD-DINO | MedSAM3 crop | CheXMask | 3280 | 0.36 [0.27–0.44] | +0.11 [+0.11 to +0.11] | 0.39 [0.22–0.56] | +0.10 [+0.10 to +0.11] |
| AP | RAD-DINO | CheXMask crop | CheXMask | 3280 | 0.35 [0.26–0.43] | +0.10 [+0.10 to +0.10] | 0.40 [0.22–0.57] | +0.10 [+0.10 to +0.11] |
| PA | ResNet-152 | Raw | MedSAM3 | 2521 | 0.46 [0.38–0.55] | Ref. | 0.60 [0.47–0.71] | Ref. |
| PA | ResNet-152 | MedSAM3 crop | MedSAM3 | 2521 | 0.58 [0.50–0.65] | +0.11 [+0.10 to +0.11] | 0.73 [0.62–0.83] | +0.12 [+0.11 to +0.13] |
| PA | ResNet-152 | CheXMask crop | MedSAM3 | 2521 | 0.56 [0.48–0.63] | +0.08 [+0.08 to +0.09] | 0.68 [0.57–0.78] | +0.07 [+0.07 to +0.08] |
| PA | ResNet-152 | Raw | CheXMask | 2521 | 0.47 [0.39–0.56] | Ref. | 0.60 [0.48–0.72] | Ref. |
| PA | ResNet-152 | MedSAM3 crop | CheXMask | 2521 | 0.59 [0.51–0.66] | +0.11 [+0.10 to +0.11] | 0.74 [0.63–0.84] | +0.12 [+0.11 to +0.13] |
| PA | ResNet-152 | CheXMask crop | CheXMask | 2521 | 0.56 [0.49–0.64] | +0.08 [+0.08 to +0.09] | 0.69 [0.58–0.79] | +0.07 [+0.07 to +0.08] |
| PA | RAD-DINO | Raw | MedSAM3 | 2521 | 0.27 [0.21–0.33] | Ref. | 0.29 [0.18–0.43] | Ref. |
| PA | RAD-DINO | MedSAM3 crop | MedSAM3 | 2521 | 0.41 [0.33–0.50] | +0.12 [+0.12 to +0.13] | 0.44 [0.30–0.60] | +0.10 [+0.10 to +0.11] |
| PA | RAD-DINO | CheXMask crop | MedSAM3 | 2521 | 0.39 [0.29–0.49] | +0.10 [+0.09 to +0.11] | 0.41 [0.23–0.60] | +0.06 [+0.05 to +0.07] |
| PA | RAD-DINO | Raw | CheXMask | 2521 | 0.27 [0.21–0.34] | Ref. | 0.29 [0.18–0.43] | Ref. |
| PA | RAD-DINO | MedSAM3 crop | CheXMask | 2521 | 0.41 [0.33–0.50] | +0.12 [+0.12 to +0.13] | 0.44 [0.29–0.60] | +0.10 [+0.09 to +0.11] |
| PA | RAD-DINO | CheXMask crop | CheXMask | 2521 | 0.39 [0.29–0.49] | +0.10 [+0.09 to +0.11] | 0.41 [0.23–0.60] | +0.06 [+0.05 to +0.07] |

