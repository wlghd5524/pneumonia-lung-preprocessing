# What stays out of the public repository

Do not commit these files:

- patient or image split CSVs, and the list of excluded DICOM identifiers
- MIMIC-derived label JSON files
- chest radiographs, lung masks, or cropped images
- image-level or patient-level prediction files that include `subject_id` or `dicom_id`
- trained model binaries (`.pt`, `.pth`, `.safetensors`)

MIMIC-CXR-JPG is distributed only to credentialed PhysioNet users, and the
data use agreement limits redistribution of the restricted data:

https://physionet.org/content/mimic-cxr-jpg/2.1.0/

PhysioNet also asks that datasets and models derived from MIMIC be shared
under the same credentialed access as the source data:

https://www.physionet.org/news/post/mimic-derived-datasets-models/

This repository follows that constraint for identifiers and images. It does
not claim that the agreement, by itself, forbids releasing trained network
weights. The fold checkpoints are simply not part of this public release.
Everything required to train them again is included: model definitions,
pretrained initialization sources, JSON configs, seeds, the checkpoint rule
(best validation AUROC), and the software environment (Supplementary Table S6).

The public `splits/` directory contains counts and SHA256 checksums only.
`.gitignore` blocks `splits/*.csv`. A credentialed user puts the four CSV
files there locally; they are not pushed.

Aggregate metrics in `results/` have no MIMIC identifiers and no local
filesystem paths. Confirm any future file the same way before committing it.

Before a public release, confirm this arrangement with the institution's data
governance contact and the current PhysioNet terms.
