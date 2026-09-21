# MIMIC-derived release policy

Do not commit the patient/image split CSVs, excluded DICOM identifiers,
MIMIC-derived labels, images, or trained model binaries to a public GitHub
repository.

PhysioNet states that datasets and models derived from MIMIC must be treated as
sensitive and shared under the same credentialed agreement as the source data:

https://www.physionet.org/news/post/mimic-derived-datasets-models/

Use a two-part release:

1. **Public GitHub repository:** code, portable configs, environment files,
   non-sensitive metadata, checkpoint hashes, and documentation.
2. **Credentialed PhysioNet project:** exact split identifier files,
   MIMIC-derived labels, and trained checkpoint binaries. Select MIMIC-CXR as a
   parent project and use an `Ext` project name if the MIMIC name is included.

The public `splits/` directory contains only non-identifying metadata and
expected checksums. Users who receive the credentialed files should place them
under `splits/` before running training; `.gitignore` prevents accidental
re-commit.

Before release, confirm the final arrangement with the institution's data
governance or compliance office and the current PhysioNet terms.
