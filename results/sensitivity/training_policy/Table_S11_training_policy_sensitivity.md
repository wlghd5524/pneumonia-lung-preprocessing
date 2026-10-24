# Table S11. Training-Policy Sensitivity to CheXpert Uncertain Pneumonia Labels in Representative Model Configurations

ResNet-152 and EVA-X-Base models were retrained for raw and MedSAM3-cropped inputs in the AP and PA cohorts under three training-label policies: Explicit-only, U-Zero, and U-One. Patient-level data partitions were preserved across policies, and validation and outer-test evaluation used the same explicit 0/1 Pneumonia endpoint. AUROC and AUPRC are reported with 95% confidence intervals estimated using 2000 subject-level cluster bootstrap replicates. ΔAUROC and ΔAUPRC represent paired differences relative to the corresponding Explicit-only model, estimated by resampling the same patients for both models.

| View | Backbone | Input | Training Policy | AUROC (95% CI) | ΔAUROC vs. Explicit-only (95% CI) | AUPRC (95% CI) | ΔAUPRC vs. Explicit-only (95% CI) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| AP | ResNet-152 | Raw | Explicit-only | 0.786 (0.768, 0.804) | Ref. | 0.772 (0.745, 0.800) | Ref. |
| AP | ResNet-152 | Raw | U-Zero | 0.767 (0.748, 0.785) | -0.019 (-0.025, -0.013) | 0.752 (0.724, 0.779) | -0.020 (-0.030, -0.011) |
| AP | ResNet-152 | Raw | U-One | 0.790 (0.773, 0.807) | +0.004 (-0.001, +0.008) | 0.778 (0.752, 0.804) | +0.006 (-0.000, +0.012) |
| AP | ResNet-152 | MedSAM3 crop | Explicit-only | 0.789 (0.772, 0.807) | Ref. | 0.772 (0.746, 0.798) | Ref. |
| AP | ResNet-152 | MedSAM3 crop | U-Zero | 0.770 (0.751, 0.787) | -0.020 (-0.027, -0.013) | 0.750 (0.723, 0.778) | -0.022 (-0.032, -0.014) |
| AP | ResNet-152 | MedSAM3 crop | U-One | 0.792 (0.774, 0.809) | +0.002 (-0.002, +0.007) | 0.778 (0.751, 0.804) | +0.005 (-0.000, +0.011) |
| AP | EVA-X-Base | Raw | Explicit-only | 0.813 (0.797, 0.830) | Ref. | 0.802 (0.777, 0.826) | Ref. |
| AP | EVA-X-Base | Raw | U-Zero | 0.803 (0.786, 0.820) | -0.010 (-0.014, -0.006) | 0.790 (0.764, 0.814) | -0.012 (-0.018, -0.006) |
| AP | EVA-X-Base | Raw | U-One | 0.816 (0.800, 0.832) | +0.003 (-0.000, +0.006) | 0.804 (0.780, 0.828) | +0.003 (-0.002, +0.007) |
| AP | EVA-X-Base | MedSAM3 crop | Explicit-only | 0.814 (0.798, 0.830) | Ref. | 0.803 (0.779, 0.826) | Ref. |
| AP | EVA-X-Base | MedSAM3 crop | U-Zero | 0.806 (0.790, 0.823) | -0.008 (-0.012, -0.003) | 0.795 (0.770, 0.818) | -0.008 (-0.014, -0.002) |
| AP | EVA-X-Base | MedSAM3 crop | U-One | 0.816 (0.800, 0.833) | +0.002 (-0.001, +0.005) | 0.805 (0.781, 0.828) | +0.002 (-0.002, +0.006) |
| PA | ResNet-152 | Raw | Explicit-only | 0.780 (0.757, 0.800) | Ref. | 0.640 (0.598, 0.678) | Ref. |
| PA | ResNet-152 | Raw | U-Zero | 0.772 (0.749, 0.792) | -0.008 (-0.016, +0.000) | 0.621 (0.579, 0.659) | -0.018 (-0.032, -0.005) |
| PA | ResNet-152 | Raw | U-One | 0.777 (0.754, 0.798) | -0.003 (-0.009, +0.004) | 0.636 (0.596, 0.674) | -0.004 (-0.016, +0.007) |
| PA | ResNet-152 | MedSAM3 crop | Explicit-only | 0.779 (0.756, 0.800) | Ref. | 0.649 (0.610, 0.686) | Ref. |
| PA | ResNet-152 | MedSAM3 crop | U-Zero | 0.769 (0.747, 0.790) | -0.010 (-0.019, -0.001) | 0.617 (0.576, 0.658) | -0.032 (-0.046, -0.017) |
| PA | ResNet-152 | MedSAM3 crop | U-One | 0.787 (0.765, 0.808) | +0.008 (+0.000, +0.015) | 0.648 (0.609, 0.686) | -0.001 (-0.012, +0.011) |
| PA | EVA-X-Base | Raw | Explicit-only | 0.826 (0.806, 0.846) | Ref. | 0.728 (0.694, 0.761) | Ref. |
| PA | EVA-X-Base | Raw | U-Zero | 0.819 (0.798, 0.839) | -0.007 (-0.011, -0.003) | 0.719 (0.685, 0.754) | -0.009 (-0.016, -0.001) |
| PA | EVA-X-Base | Raw | U-One | 0.822 (0.802, 0.842) | -0.004 (-0.008, +0.001) | 0.721 (0.687, 0.756) | -0.007 (-0.014, -0.000) |
| PA | EVA-X-Base | MedSAM3 crop | Explicit-only | 0.823 (0.803, 0.843) | Ref. | 0.723 (0.690, 0.756) | Ref. |
| PA | EVA-X-Base | MedSAM3 crop | U-Zero | 0.822 (0.801, 0.843) | -0.001 (-0.006, +0.003) | 0.721 (0.687, 0.754) | -0.002 (-0.009, +0.006) |
| PA | EVA-X-Base | MedSAM3 crop | U-One | 0.823 (0.802, 0.843) | -0.000 (-0.005, +0.005) | 0.721 (0.686, 0.754) | -0.002 (-0.009, +0.005) |
