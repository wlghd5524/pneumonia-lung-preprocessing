# What stays out of the public repository

Do not commit these files:

- patient or image split CSVs, and the list of excluded DICOM identifiers
- MIMIC-derived label JSON files
- chest radiographs, lung masks, or cropped images
- image-level or patient-level prediction files that include `subject_id` or `dicom_id`
- expert lung masks, polygon files, and annotation sheets that name radiographs
- trained model binaries (`.pt`, `.pth`, `.safetensors`)

MIMIC-CXR-JPG is distributed only to credentialed PhysioNet users, and the
data use agreement limits redistribution of the restricted data:

https://physionet.org/content/mimic-cxr-jpg/2.1.0/

## Why the fold checkpoints are withheld

PhysioNet's policy on MIMIC-derived resources (news item of 24 April 2024)
states that "any derived datasets or models should be treated as containing
sensitive information" and that, to share them, "they should be shared on
PhysioNet under the same agreement as the source data":

https://www.physionet.org/news/post/mimic-derived-datasets-models/

The 350 fold-specific classifiers (70 runs x 5 folds) were trained on
MIMIC-CXR-JPG and are models derived from it in this sense. A public GitHub
repository is not credentialed access, so we do not place them here. This is
our reading of PhysioNet's policy, not a claim that the data use agreement
text by itself names network weights. The policy page does not say that
sharing elsewhere is prohibited, but it names PhysioNet as the place to share.

Nothing else in this repository is a MIMIC-derived model. The MedSAM3 LoRA file
(`checkpoints/best_lora_weights.pt`) is the authors' public MedSAM3-v1 release,
used unmodified and not trained on MIMIC data by us. See `checkpoints/README.md`.

### How to obtain or regenerate the checkpoints

- Regenerate: everything required to train them again is included: model
  definitions, pretrained initialization sources, JSON configs, seeds, the
  checkpoint rule (best validation AUROC), and the software environment
  (Supplementary Table S6). `README.md` section 9 gives run time, resources,
  and the limits on reproducing identical numbers.
- Verify: `checkpoints/trained_checkpoint_manifest.csv` lists size and SHA256 of
  each of the 350 original files, so anyone who obtains a copy through an
  authorized channel can check it.
- Authorized copy: none. The authors do not distribute the fold checkpoints,
  neither on PhysioNet nor on request. Retraining is the supported route.

`cxr-segmentation-validation/` documents the expert-contour procedure and the
Dice, IoU, and HD95 calculation. It does not contain masks or radiograph
identifiers. A credentialed user supplies local mask directories when running
that script.

The public `splits/` directory contains counts and SHA256 checksums only.
`.gitignore` blocks `splits/*.csv`. The four CSV files are not distributed. A
credentialed user generates them locally with the released builder and checks
the SHA256 values (`README.md`, Section 5); they are not pushed.

Aggregate metrics in `results/` have no MIMIC identifiers and no local
filesystem paths. Confirm any future file the same way before committing it.

Before a public release, confirm this arrangement with the institution's data
governance contact and the current PhysioNet terms.
