# Step 1: Project Setup & Dataset Acquisition

## Overview
This document details the complete environment setup, repository structuring, and dataset acquisition for the **RoadSense: Robust Traffic Sign Recognition** capstone project (Group 4, Atria University).

---

## 1. Repository Architecture Initialized
Per Section 8 of the Capstone Project Brief, the repository structure was created:

```text
CNN-TRAFFIC/
├── .gitignore                      # Configured to ignore datasets, model weights, checkpoints
├── requirements.txt                # Python dependencies
├── data/
│   ├── README.md                   # Dataset documentation & source links
│   ├── Train.csv                   # 39,209 training records
│   ├── Test.csv                    # 12,630 test records with ground truth
│   ├── Meta.csv                    # Sign class visual metadata
│   ├── Train/                      # 43 class subfolders (00000 to 00042)
│   └── Test/                       # 12,630 test images
├── notebooks/                      # Jupyter notebooks for each project phase
├── src/                            # Modular Python source code
│   └── download_data.py            # Reproducible downloader script
├── results/
│   ├── figures/                    # Saved plots and visual outputs
│   └── confusion_matrices/         # Model confusion matrices
├── docs/                           # Step-by-step documentation
└── report/                         # Final project report PDF
```

---

## 2. Dataset Acquisition Details
- **Dataset Name:** German Traffic Sign Recognition Benchmark (GTSRB)
- **Primary Characteristics:**
  - Over 50,000 cropped RGB traffic sign images from real dashcam video footage.
  - 43 distinct classes (speed limits, warnings, prohibitions, mandatory directions).
  - Variable image sizes ranging from ~15×15 to ~250×250 pixels.
  - Video tracks where consecutive frames capture the same physical sign under changing distance and angle.

### Download Implementation:
- Implemented `src/download_data.py` to fetch from the verified GTSRB Kaggle/ERDA mirror.
- Extracted all training images into `data/Train/<ClassId>/` and test images into `data/Test/`.
- Temporary parquet download archives were safely cleaned up to conserve local disk space.

---

## 3. Dataset Verification Results

| Dataset Component | Target in Project Brief | Actual Verified on Disk | Status |
| :--- | :--- | :--- | :--- |
| **Total Images** | > 50,000 | 51,839 | PASSED |
| **Class Folders** | 43 classes | 43 directories (`0` to `42`) | PASSED |
| **Training Images** | ~39,209 | 39,209 images | PASSED |
| **Test Images** | ~12,630 | 12,630 images | PASSED |
| **Metadata CSVs** | Train, Test, Meta CSVs | `Train.csv`, `Test.csv`, `Meta.csv` | PASSED |
| **Track Structure** | Multi-frame tracks per sign | Preserved in file naming scheme | PASSED |

---

## 4. Next Step
Proceed to **Phase 1: Understand the Data (EDA & Leakage-Free Validation Split)**:
1. Class distribution analysis & visualization.
2. 43-class sample image grid with human-readable labels.
3. Image dimension & brightness distributions.
4. Leakage-free track-based train/validation split (preventing consecutive-frame data leakage).
5. Comprehensive discussion on label-safe data augmentations.
