# Pre-processing Pipeline

This package groups every operation that happens **after** DICOM→NIfTI conversion and **before** the learning / analysis stages.

📂 Sub-modules

```
preprocessing/
│
├── co_registration/          # spatial alignment of all modalities (elastix based)
│   └── README.md
│
├── intensity_standardisation/ # Nyul / histogram methods (MRI) + SUV/CT rescaling
│   └── README.md
│
├── harmonization_combat/      # site / scanner batch-effect removal using ComBat
│   └── README.md
│
└── roi_segmentation/          # automated & semi-automated organ / lesion masks
    └── README.md
```

Each sub-folder will eventually contain:
1. A light-weight python API (`*.py`)
2. Example parameter files / model checkpoints
3. A CLI wrapper (`__main__.py`) so the whole stage can be executed via `python -m preprocessing.<module>`

> NOTE: **Do not add implementation yet.**  We will flesh out each module step-by-step following the detailed instructions that will reference elastix manual, nIHMS MSERg/SDM paper, and PllS intensity papers. 