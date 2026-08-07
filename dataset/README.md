# Dataset layout

MedARC uses LLaVA-style JSON manifests. Medical images and the large CT-RATE /
MR-RATE manifests are not distributed by this repository. Obtain each dataset
from its official source, comply with its data-use agreement, and keep all
patient data outside Git.

## Manifest schema

Use paths relative to the `IMAGE_FOLDER` supplied to a training/evaluation
script. Do not store machine-specific absolute paths in a manifest.

```json
[
  {
    "id": "sample-001",
    "video": "batch01/sample-001/volume.nii.gz",
    "conversations": [
      {"from": "human", "value": "<video>\nDescribe the findings."},
      {"from": "gpt", "value": "Findings: ..."}
    ]
  }
]
```

CT VQA manifests may contain multiple alternating human/GPT turn pairs and a
`type` field on human turns. A `video` value may also be a list when an example
contains more than one volume.

## Recommended local layout

```text
MedARC/
├── dataset/
│   ├── CT-RATE/
│   │   ├── validation_reports_cleaned_med_readable.json
│   │   └── valid_vqa_cleaned_med_readable.json
│   └── MR-RATE/
│       └── mr_rate_med_test_readable.json
└── data/
    ├── CT-RATE/
    │   └── valid_fixed/...
    └── MR-RATE/
        └── batch01/...
```

Both `dataset/**` and `data/` are ignored by Git; this README is the only file
under `dataset/` intended for publication.
