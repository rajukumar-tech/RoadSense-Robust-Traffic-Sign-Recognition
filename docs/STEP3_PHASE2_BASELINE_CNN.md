# Step 3: Phase 2 — Baseline CNN from Scratch

**Project:** RoadSense: Robust Traffic Sign Recognition  
**Group:** Group 4 | Atria University  
**Phase Lead:** Raju (Scratch CNN Design, Training, Evaluation & Error Analysis)  
**Team Members:** Aseem (Team Leader), Raju, Sohail, Manoj  

---

## 1. Overview & Objectives
Phase 2 fulfills the primary core modeling deliverable specified in the Capstone Brief:
> *"Design and train a CNN from scratch; report accuracy and macro-F1 on the clean test set. Plot learning curves and the confusion matrix; list the most confused sign pairs."*

Rather than relying on pre-trained backbones or transfer learning, this baseline model is designed entirely from scratch, tailored specifically for $32 \times 32$ RGB traffic sign images, and regularized to prevent overfitting on dominant classes.

---

## 2. Architecture Design & Technical Justification

The custom architecture (`BaselineCNN` in `src/models.py`) consists of 6 convolutional layers arranged into 3 hierarchical feature extraction blocks followed by a dense classification head:

```text
Input (3 x 32 x 32)
  │
  ├── [ConvBlock 1]
  │     Conv2D(3  → 32, 3x3) + BatchNorm2D + ReLU
  │     Conv2D(32 → 32, 3x3) + BatchNorm2D + ReLU
  │     MaxPool2D(2x2, stride=2)  ──>  (32 x 16 x 16)
  │     Dropout2D(p = 0.2)
  │
  ├── [ConvBlock 2]
  │     Conv2D(32 → 64, 3x3) + BatchNorm2D + ReLU
  │     Conv2D(64 → 64, 3x3) + BatchNorm2D + ReLU
  │     MaxPool2D(2x2, stride=2)  ──>  (64 x 8 x 8)
  │     Dropout2D(p = 0.3)
  │
  ├── [ConvBlock 3]
  │     Conv2D(64  → 128, 3x3) + BatchNorm2D + ReLU
  │     Conv2D(128 → 128, 3x3) + BatchNorm2D + ReLU
  │     MaxPool2D(2x2, stride=2)  ──>  (128 x 4 x 4)
  │     Dropout2D(p = 0.4)
  │
  └── [Classification Head]
        Flatten(2048)
        Linear(2048 → 256) + BatchNorm1D + ReLU
        Dropout(p = 0.5)
        Linear(256 → 43)  ──>  Logits for 43 classes
```

### Architectural Decisions & Justifications:
1. **$3 \times 3$ Small Kernel Stacking:** Stacking two $3 \times 3$ convolutions provides the same effective receptive field as a $5 \times 5$ convolution ($2 \times (3 - 1) + 1 = 5$), but introduces an additional non-linear activation (ReLU) and reduces total parameters by $28\%$ ($\frac{2 \times 3^2}{5^2} = \frac{18}{25} = 0.72$).
2. **Batch Normalization on Every Layer:** Stabilizes internal covariate shift, acts as mild regularization, and allows for higher learning rates without divergence.
3. **Progressive Spatial Downsampling:** Three $2 \times 2$ MaxPool operations reduce spatial dimensions ($32 \rightarrow 16 \rightarrow 8 \rightarrow 4$), allowing deep layers to capture global sign semantics while keeping feature map memory compact.
4. **Progressive Dropout ($0.2 \rightarrow 0.3 \rightarrow 0.4 \rightarrow 0.5$):** Prevents co-adaptation of neurons and strongly curtails overfitting, which is critical given the severe frequency imbalance among classes.
5. **Parameter Efficiency:** The entire network has **823,307 trainable parameters**, making it fast to train and evaluate while retaining expressive feature capacity.

---

## 3. Training Strategy & Hyperparameters

- **Dataset Split:** Trained on `data/train_split.csv` (31,379 images) and validated on `data/val_split.csv` (7,830 images) with strictly zero track leakage.
- **Batch Size:** 64
- **Input Resolution:** $32 \times 32 \times 3$ (RGB normalized with GTSRB empirical mean & std).
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 10^{-4}$).
- **Initial Learning Rate:** $10^{-3}$ with dynamic reduction on plateau (`ReduceLROnPlateau`, factor $= 0.5$, patience $= 2$).
- **Class-Weighted Cross-Entropy Loss:**
  $$w_c = \sqrt{\frac{N}{C \cdot N_c}}$$
  Rare classes receive higher penalty when misclassified, forcing the gradient descent optimizer to learn distinctive features for minority signs (e.g. *Speed Limit 20*, *End of Limits*).

---

## 4. Evaluation on Clean Test Set (12,630 Images)

The model was evaluated on the official clean test set using the best saved checkpoint (`checkpoints/best_baseline_cnn.pth`).

### Summary of Empirical Performance Metrics:
| Metric | Measured Value | Target in Published Literature |
| :--- | :---: | :---: |
| **Clean Test Accuracy** | **97.98%** | 95.0% - 98.0% |
| **Macro-F1 Score** | **97.23%** | > 95.0% |
| **Weighted-F1 Score** | **97.98%** | > 95.0% |
| **Macro-Precision** | **97.08%** | > 95.0% |
| **Macro-Recall** | **97.57%** | > 95.0% |

Recorded in `results/metrics.csv` and detailed per-class stats exported to `results/baseline_cnn_classification_report.csv`.

---

## 5. Error Analysis & Confusion Matrix

### Top 5 Most Frequently Confused Sign Pairs:
Empirical analysis of the normalized confusion matrix reveals that misclassifications occur primarily between signs sharing identical geometric shapes, color palettes, and subtle interior symbology:
1. **Speed limit (120 km/h) [Class 8] $\rightarrow$ Speed limit (60 km/h) [Class 3] (27 errors):** Both are red circular speed limit signs where downsampled digits ('120' vs '60') blur together at low resolution.
2. **Priority road [Class 12] $\rightarrow$ No vehicles [Class 15] (17 errors):** Shared circular / diamond yellow-white center profile.
3. **End of speed limit (80 km/h) [Class 6] $\rightarrow$ End of no passing for > 3.5t [Class 42] (13 errors):** Both are gray circular signs with diagonal slash derestriction bands.
4. **Road work [Class 25] $\rightarrow$ Right-of-way at intersection [Class 11] (12 errors):** Both are red warning triangles with black interior iconography.
5. **General caution [Class 18] $\rightarrow$ Wild animals crossing [Class 31] (12 errors):** Red warning triangles with complex central animal/exclamation silhouettes.

The high-resolution 43×43 confusion matrix heatmap is saved to `results/confusion_matrices/baseline_cnn_cm.png`.

---

## 6. Code Deliverables for Phase 2
- **Model Architecture:** [src/models.py](file:///c:/Users/Raju%20Kumar/Downloads/CNN-TRAFFIC/src/models.py)
- **Training Pipeline:** [src/train.py](file:///c:/Users/Raju%20Kumar/Downloads/CNN-TRAFFIC/src/train.py)
- **Clean Evaluation Suite:** [src/evaluate.py](file:///c:/Users/Raju%20Kumar/Downloads/CNN-TRAFFIC/src/evaluate.py)
- **Interactive Notebook:** [notebooks/02_baseline_cnn.ipynb](file:///c:/Users/Raju%20Kumar/Downloads/CNN-TRAFFIC/notebooks/02_baseline_cnn.ipynb)
- **Saved Figures & Artifacts:**
  - `results/figures/baseline_cnn_learning_curves.png`
  - `results/confusion_matrices/baseline_cnn_cm.png`
  - `results/metrics.csv`
  - `checkpoints/best_baseline_cnn.pth`

---

## 7. Next Step: Phase 3 — Robustness Test Bench
With the clean baseline model trained and evaluated, we move to the central novel research phase:
Building the **Robustness Test Bench** (`src/corruptions.py`) featuring **5 real-world driving corruption types** across **3 severity levels** (15 conditions) and stress-testing our baseline model.
