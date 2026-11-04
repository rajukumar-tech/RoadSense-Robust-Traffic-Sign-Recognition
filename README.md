# RoadSense: Robust Traffic Sign Recognition

Multi-class traffic sign classification with a focus on robustness to real-world driving corruptions and environmental degradation.

**Course:** Neural Network and Convolutional Neural Network Sprint  
**Institution:** Atria University  
**Group:** Group 4  

---

## Team Contributions

| Member | Role | Phase(s) | Primary Responsibilities | Owned Components |
| :--- | :--- | :---: | :--- | :--- |
| **Raju** | Member | **Phase 1, 2 & 3** | EDA, leakage-free splitting, scratch CNN design, training loop, clean evaluation, robustness test bench (code & testing) | `src/eda_and_split.py`, `src/models.py`, `src/train.py`, `src/dataset.py`, `src/corruptions.py`, `src/robustness_bench.py`, `notebooks/01_eda.ipynb`, `notebooks/02_baseline_cnn.ipynb` |
| **Manoj** | Member | **Phase 4** (code) | Designed and implemented the full Phase 4 codebase: `TransferModel` class (ResNet-18 & MobileNetV2) with two-stage fine-tuning strategy; rewrote unified training pipeline (`src/train.py`) supporting all 4 model types with differential LRs, class-weighted loss & label-safe augmentation; extended `src/evaluate.py` with checkpoint-aware auto model detection; extended `src/robustness_bench.py` for all model types; built end-to-end pipeline runners (`run_phase4.py`, `run_all_phases.py`); authored Phase 4 interactive notebook and Colab GPU training notebook | `src/models.py` (TransferModel + BaselineCNN extensions), `src/train.py` (full rewrite), `src/evaluate.py` (extended), `src/robustness_bench.py` (extended), `notebooks/03_transfer_learning.ipynb`, `RoadSense_Colab_Training.ipynb`, `run_phase4.py`, `run_all_phases.py`, `docs/STEP4_PHASE4_TRANSFER_LEARNING.md` |
| **Sohail** | Member | **Phase 4** (training & testing) + **Phase 5** | Phase 4 model training, evaluation & comparison; Grad-CAM explainability, interpretation of all models | `src/evaluate.py`, `src/gradcam.py`, `notebooks/04_evaluation_and_explainability.ipynb`, `results/figures/` |
| **Aseem** | **Team Leader** | **Phase 6** | Documentation, project coordination, pipeline architecture, dataset acquisition, final report | `src/download_data.py`, `docs/`, `report/`, `README.md` |

---

## Step-by-Step Documentation

Detailed step-by-step documentation is maintained for every phase of the project:
1. [Step 1: Dataset Acquisition & Verification](docs/STEP1_DATASET_ACQUISITION.md) — GTSRB mirror download, extraction, and file verification. *(Aseem)*
2. [Step 2: Phase 1 — EDA & Leakage-Free Validation Split](docs/STEP2_PHASE1_EDA_AND_SPLIT.md) — Class distribution, 43-class sample grid, size/brightness analysis, and disjoint track-based splitting. *(Raju)*
3. [Step 3: Phase 2 — Baseline CNN from Scratch](docs/STEP3_PHASE2_BASELINE_CNN.md) — Architecture design, training convergence, clean test evaluation, and confusion matrix error analysis. *(Raju)*
4. [Step 4: Phase 4 — Robustness Improvement & Transfer Learning](docs/STEP4_PHASE4_TRANSFER_LEARNING.md) — Two-stage fine-tuning, label-safe augmentation, unified pipeline, four-way model comparison. *(Manoj)*
5. [Step 5: Phase 5 — Explainability, Grad-CAM & Real-World Evaluation](docs/STEP5_PHASE5_EXPLAINABILITY.md) — Native Grad-CAM implementation, clean vs corrupted attention shift, dual-target error analysis, and Indian road sign evaluation. *(Sohail & Team)*

---

## Experimental Benchmark Results

### Clean Test Set Performance (12,630 Images)

| Model Architecture | Owned By | Parameters | Clean Accuracy | Macro-F1 | Weighted-F1 | Checkpoint |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline CNN (Scratch)** | Raju | 823,307 | **97.98%** | **97.23%** | **97.98%** | `checkpoints/best_baseline_cnn.pth` |
| **Improved CNN (+augmentation)** | Raju | 823,307 | **98.70%** | **97.98%** | **98.69%** | `checkpoints/best_improved_cnn.pth` |
| **ResNet-18 (fine-tuned)** | Manoj/Sohail | 11,319,147 | **96.27%** | **94.70%** | **96.25%** | `checkpoints/best_resnet18.pth` |
| **MobileNetV2 (fine-tuned)** | Manoj/Sohail | 2,563,115 | **93.54%** | **92.29%** | **93.56%** | `checkpoints/best_mobilenet_v2.pth` |

### Robustness Benchmark (5 Corruptions × 3 Severities = 15 Conditions)

| Model | Clean Acc | Mean Corruption Acc (mCA) | Robustness Gap |
| :--- | :---: | :---: | :---: |
| **Baseline CNN** | 97.98% | 81.46% | -16.52% |
| **Improved CNN (+aug)** | **98.70%** | **83.11%** | **-15.59%** |
| **ResNet-18** | 96.27% | 73.82% | -22.45% |
| **MobileNetV2** | 93.54% | 66.18% | -27.36% |

> Full comparison table saved to `results/model_comparison.csv` after running Phase 4.

---

## Project Repository Layout

```text
ROADSENSE/
├── README.md                                   # Project overview, team table, results
├── requirements.txt                            # Python dependencies
├── .gitignore                                  # Excludes raw images, checkpoints, large weights
├── data/
│   ├── README.md                               # Dataset source and download notes
│   ├── Train.csv                               # Full 39,209 training metadata records
│   ├── Test.csv                                # 12,630 test metadata records with ground truth
│   ├── Meta.csv                                # Class metadata definitions
│   ├── train_split.csv                         # 31,379 training images (80% atomic tracks)
│   ├── val_split.csv                           # 7,830 validation images (20% atomic tracks)
│   ├── Train/                                  # 43 class subfolders (00000 to 00042) — not committed
│   └── Test/                                   # 12,630 test images — not committed
├── docs/
│   ├── STEP1_DATASET_ACQUISITION.md
│   ├── STEP2_PHASE1_EDA_AND_SPLIT.md
│   ├── STEP3_PHASE2_BASELINE_CNN.md
│   └── STEP4_PHASE4_TRANSFER_LEARNING.md
├── notebooks/
│   ├── 01_eda.ipynb                            # Phase 1: Interactive EDA & Split
│   ├── 02_baseline_cnn.ipynb                   # Phase 2: Scratch CNN design & training
│   ├── 03_transfer_learning.ipynb              # Phase 4: Augmentation & transfer learning
│   └── 04_evaluation_and_explainability.ipynb  # Phase 3 & 5: Robustness bench & Grad-CAM
├── src/
│   ├── utils.py                                # Class names (43), categories, seed fixing
│   ├── download_data.py                        # Dataset downloader & verification
│   ├── eda_and_split.py                        # Headless EDA generator & track splitter
│   ├── dataset.py                              # PyTorch Dataset & DataLoader pipeline
│   ├── models.py                               # BaselineCNN + TransferModel (ResNet-18/MobileNetV2)
│   ├── train.py                                # Unified training pipeline (all 4 model types)
│   ├── evaluate.py                             # Clean evaluation suite (auto-detects model type)
│   ├── robustness_bench.py                     # 5×3 corruption robustness benchmark
│   ├── corruptions.py                          # 5 corruption types at 3 severity levels
│   └── explainability.py                       # Native Grad-CAM engine & real-world evaluation
├── run_phase4.py                               # Phase 4 automated runner
├── run_phase5.py                               # Phase 5 automated explainability runner
├── run_all_phases.py                           # Master end-to-end runner (Phases 1-5)
├── results/
│   ├── figures/                                # Saved high-res EDA and evaluation plots
│   ├── confusion_matrices/                     # Generated confusion matrices
│   ├── metrics.csv                             # Clean test set metrics (all models)
│   └── model_comparison.csv                    # Four-way model comparison (generated in Phase 4)
└── report/                                     # Final technical project report (PDF)
```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone <repo-url>
cd ROADSENSE
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Acquire the GTSRB dataset
```bash
python src/download_data.py
```
This downloads GTSRB from Kaggle/official mirror into `data/Train/` and `data/Test/`.

---

## Running the Full Pipeline

### Phase 1 — EDA & Leak-Free Split
```bash
python src/eda_and_split.py
# OR open: notebooks/01_eda.ipynb
```

### Phase 2 — Baseline CNN from Scratch
```bash
python src/train.py --model_type baseline_cnn --epochs 15 --name baseline_cnn
python src/evaluate.py --name baseline_cnn
```

### Phase 3 — Robustness Benchmark (Baseline)
```bash
python src/robustness_bench.py --name baseline_cnn
```

### Phase 4A — Improved CNN with Augmentation
```bash
python src/train.py --model_type improved_cnn --epochs 15 --use_aug --name improved_cnn
python src/evaluate.py --name improved_cnn
python src/robustness_bench.py --name improved_cnn
```

### Phase 4B — Transfer Learning
```bash
# ResNet-18 (two-stage fine-tuning: 5 frozen + 10 full)
python src/train.py --model_type resnet18 --epochs 15 --use_aug --stage1_epochs 5 --name resnet18
python src/evaluate.py --name resnet18
python src/robustness_bench.py --name resnet18

# MobileNetV2
python src/train.py --model_type mobilenet_v2 --epochs 15 --use_aug --stage1_epochs 5 --name mobilenet_v2
python src/evaluate.py --name mobilenet_v2
python src/robustness_bench.py --name mobilenet_v2
```

### Phase 5 — Explainability & Real-World Evaluation
`ash
# Run Grad-CAM on a specific model:
python run_phase5.py --name baseline_cnn
python run_phase5.py --name resnet18

# Run full Phase 5 across all 4 architectures:
python run_phase5.py --all_models
`

### Notebooks
Open the notebooks in Jupyter for interactive walkthroughs:
- `notebooks/03_transfer_learning.ipynb` — Phase 4: full augmentation + transfer learning comparison
- `notebooks/04_evaluation_and_explainability.ipynb` — Phase 5: Grad-CAM explainability

---

## Model Architecture Summary

### BaselineCNN (Phases 2 & 4A)
Custom 6-layer CNN designed from scratch for 32×32 GTSRB images.
- 3 ConvBlocks (32→64→128 channels), each with 2× Conv3×3 + BatchNorm + ReLU + MaxPool + Dropout
- Classifier head: Flatten → Linear(2048→256) → BN → ReLU → Dropout → Linear(256→43)
- **823,307 trainable parameters**

### TransferModel — ResNet-18 (Phase 4B)
ImageNet-pretrained ResNet-18 with custom GTSRB head. Input: 48×48.
- Backbone: 8 residual blocks (frozen Stage 1, fine-tuned Stage 2)
- New head: Dropout → Linear(512→256) → BN → ReLU → Dropout → Linear(256→43)
- **~11.7M total parameters**

### TransferModel — MobileNetV2 (Phase 4B)
ImageNet-pretrained MobileNetV2 with custom GTSRB head. Input: 48×48.
- Backbone: Inverted residual blocks with depthwise separable convolutions
- New head: Dropout → Linear(1280→256) → BN → ReLU → Dropout → Linear(256→43)
- **~3.4M total parameters**
