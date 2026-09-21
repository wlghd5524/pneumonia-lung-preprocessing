# Credentialed patient-level split identifiers

The exact CSV/JSON identifier files are intentionally absent from the public
GitHub release. Download them from the companion credentialed PhysioNet
project and place them in this directory. They contain de-identified MIMIC-CXR
`subject_id` and `dicom_id` values; one patient is assigned to exactly one
outer/CV partition.

File checksums:

```text
f6ff52686352f8d455a3bc176a82c9af9dc1cc38f55cf86d077e51f292b09c8f  AP_outer_test.csv
f573481e7bb09660068dd1ae6a2ac2cac3984493f743a593ae95a65f48fb1362  AP_fold_assignment.csv
70522edc345e1d6d51247a1d43a59e1ce5276f2da0bc0eb7b73566537f0de193  PA_outer_test.csv
ebcc61291b2bc851e1a6e3704c30d0d6ae6515224f3d2710274063d414a7c641  PA_fold_assignment.csv
```

Image counts:

- AP outer test: 3,280
- AP train/validation: 19,614
- PA outer test: 2,521
- PA train/validation: 16,088

Unique subject counts are 11,951 for the released AP split and 12,385 for PA.
The AP metadata's 11,953-subject reference count is the pre-exclusion count.

The credentialed `AP_excluded_dicom_ids.json` records the five AP images
excluded because no usable CheXMask lung RLE was available.
`split_summary.json` retains only non-identifying split methods and cohort
counts.
