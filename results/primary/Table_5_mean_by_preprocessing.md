# Table 5. Mean held-out test performance across the seven evaluated backbone architectures according to acquisition view and lung preprocessing method

AUROC was the primary discrimination metric, and AUPRC was evaluated as a secondary metric. Differences relative to raw input are shown descriptively using the corresponding backbone-averaged values within each acquisition view.

| View | Preprocessing | Mean AUROC | ΔAUROC vs. Raw | Mean AUPRC | ΔAUPRC vs. Raw |
| --- | --- | ---: | ---: | ---: | ---: |
| AP | Raw | 0.8023 | Reference | 0.7886 | Reference |
| AP | CheXMask mask | 0.7977 | −0.0046 | 0.7841 | −0.0045 |
| AP | CheXMask crop | 0.8032 | +0.0009 | 0.7891 | +0.0005 |
| AP | MedSAM3 mask | 0.7966 | −0.0057 | 0.7818 | −0.0068 |
| AP | MedSAM3 crop | 0.8044 | +0.0021 | 0.7904 | +0.0018 |
| PA | Raw | 0.8045 | Reference | 0.6868 | Reference |
| PA | CheXMask mask | 0.7902 | −0.0142 | 0.6676 | −0.0192 |
| PA | CheXMask crop | 0.8027 | −0.0018 | 0.6828 | −0.0040 |
| PA | MedSAM3 mask | 0.7904 | −0.0141 | 0.6683 | −0.0185 |
| PA | MedSAM3 crop | 0.8044 | −0.0001 | 0.6880 | +0.0012 |

Mean AUROC and mean AUPRC average the seven per-backbone values rounded to four decimals. ΔAUROC is the mean of the seven per-backbone ΔAUROC values in Table 4. ΔAUPRC is the displayed mean AUPRC minus the displayed raw mean AUPRC.
