# RoadSense: Robust Traffic Sign Recognition

Multi-class traffic sign classification with a focus on robustness to real-world driving corruptions and environmental degradation.

**Course:** Neural Network and Convolutional Neural Network Sprint  
**Institution:** Atria University  
**Group:** Group 4  

---

## Team Contributions

| Member | Role | Primary Responsibilities | Owned Components |
| :--- | :--- | :--- | :--- |
| **Aseem** | **Team Leader** | Project coordination, pipeline architecture, dataset acquisition & leak-free splitting | `src/download_data.py`, `src/eda_and_split.py`, `data/` |
| **Raju** | Member | Scratch CNN design, training loop, hyperparameter tuning, clean evaluation & confusion matrix | `src/models.py`, `src/train.py`, `notebooks/02_baseline_cnn.ipynb` |
| **Sohail** | Member | Robustness test bench implementation (5 corruptions × 3 severities), stress-test analysis | `src/corruptions.py`, `src/evaluate.py`, `results/figures/` |
| **Manoj** | Member | Label-safe data augmentation, transfer learning (ResNet-18 / MobileNetV2), before/after comparison | `notebooks/03_transfer_learning.ipynb`, fine-tuning experiments |

---

## Step-by-Step Documentation

Detailed step-by-step documentation is maintained for every phase of the project:
1. [Step 1: Dataset Acquisition & Verification](docs/STEP1_DATASET_ACQUISITION.md) — GTSRB mirror download, extraction, and file verification.
2. [Step 2: Phase 1 — EDA & Leakage-Free Validation Split](docs/STEP2_PHASE1_EDA_AND_SPLIT.md) — Class distribution, 43-class sample grid, size/brightness analysis, and disjoint track-based splitting.
3. [Step 3: Phase 2 — Baseline CNN from Scratch](docs/STEP3_PHASE2_BASELINE_CNN.md) — Architecture design, training convergence, clean test evaluation, and confusion matrix error analysis.

---

## Experimental Benchmark Results

### 1. Clean Test Set Performance (12,630 Images)

| Model Architecture | Owned By | Parameters | Clean Accuracy | Macro-F1 | Weighted-F1 | Checkpoint |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline CNN (Scratch)** | Raju | 823,307 | **97.98%** | **97.23%** | **97.98%** | `checkpoints/best_baseline_cnn.pth` |

---

## Project Repository Layout

```text
CNN-TRAFFIC/
├── README.md                               # Project overview, setup, and team table
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Excludes raw images, checkpoints, large weights
├── data/
│   ├── README.md                           # Dataset source and download notes
│   ├── Train.csv                           # Full 39,209 training metadata records
│   ├── Test.csv                            # 12,630 test metadata records with ground truth
│   ├── Meta.csv                            # Class metadata definitions
│   ├── train_split.csv                     # 31,379 training images (80% atomic tracks)
│   ├── val_split.csv                       # 7,830 validation images (20% atomic tracks)
│   ├── Train/                              # 43 class subfolders (00000 to 00042)
│   └── Test/                               # 12,630 test images
├── docs/
│   ├── STEP1_DATASET_ACQUISITION.md        # Step 1 documentation
│   └── STEP2_PHASE1_EDA_AND_SPLIT.md       # Step 2 documentation
├── notebooks/
│   ├── 01_eda.ipynb                        # Phase 1: Interactive EDA & Split Notebook
│   ├── 02_baseline_cnn.ipynb               # Phase 2: Scratch CNN design and training
│   ├── 03_transfer_learning.ipynb          # Phase 4: Transfer learning & augmentation
│   └── 04_evaluation_and_explainability.ipynb # Phase 3 & 5: Robustness bench & Grad-CAM
├── src/
│   ├── utils.py                            # Class names (43), categories, seed fixing
│   ├── download_data.py                    # Dataset downloader & verification
│   ├── eda_and_split.py                    # Headless EDA generator & track splitter
│   ├── dataset.py                          # PyTorch Dataset & DataLoader pipeline
│   ├── models.py                           # Custom Scratch CNN & Transfer architectures
│   ├── train.py                            # Training loop, validation, and checkpointing
│   ├── evaluate.py                         # Clean and corrupted evaluation suite
│   └── corruptions.py                      # 5 corruption types at 3 severity levels
├── results/
│   ├── figures/                            # Saved high-res EDA and evaluation plots
│   └── confusion_matrices/                 # Generated confusion matrices
└── report/                                 # Final technical project report (PDF)
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd CNN-TRAFFIC
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Acquire dataset:**
   ```bash
   python src/download_data.py
   ```

4. **Run Phase 1 EDA & Leak-Free Validation Split:**
   ```bash
   python src/eda_and_split.py
   ```
   Or open `notebooks/01_eda.ipynb` in your Jupyter environment.
