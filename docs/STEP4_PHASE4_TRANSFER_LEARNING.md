# Step 4: Phase 4 — Robustness Improvement & Transfer Learning

**Project:** RoadSense: Robust Traffic Sign Recognition  
**Group:** Group 4 | Atria University  
**Phase Lead:** Manoj (Augmentation & Transfer Learning)  
**Team Members:** Aseem (Team Leader), Raju, Sohail, Manoj  

---

## 1. Overview & Objectives

Phase 4 builds on the clean baseline (97.98% accuracy) and robustness benchmark from Phases 2–3. It has two parallel tracks:

> **Phase 4A:** Train the *same* BaselineCNN architecture with **label-safe augmentations** to directly address the corruption vulnerabilities identified in Phase 3.  
> **Phase 4B:** Fine-tune **ResNet-18** and **MobileNetV2** — pretrained ImageNet backbones — for 43-class GTSRB classification using a two-stage strategy.

---

## 2. Phase 4A — Improved CNN with Label-Safe Augmentations

### 2.1 Augmentation Strategy

The robustness benchmark revealed the model degrades most under brightness/contrast changes, Gaussian noise, and motion blur. We apply the following training-time augmentations:

| Augmentation | Parameters | Simulates |
| :--- | :--- | :--- |
| `RandomRotation` | ±10° | Camera tilt, sign angle variation |
| `RandomAffine` | translate=(6%,6%), scale=(0.95–1.05) | Sign position and distance variation |
| `ColorJitter` | brightness=0.2, contrast=0.2, saturation=0.2 | Lighting conditions (glare, shadow, overcast) |
| ❌ Horizontal Flip | **NEVER** | Would change Keep-Left → Keep-Right |
| ❌ Vertical Flip | **NEVER** | Inverts directional signs and Yield |

All augmentations are applied **only during training**, not at evaluation time.

### 2.2 Training Configuration

- **Architecture:** `BaselineCNN` (identical to Phase 2)
- **Model name:** `improved_cnn`
- **Epochs:** 15 | **Batch:** 64 | **LR:** 1e-3 (Adam + ReduceLROnPlateau)
- **Class-weighted CE loss:** Yes
- **Command:**

```bash
python src/train.py --model_type improved_cnn --epochs 15 --use_aug --name improved_cnn
python src/evaluate.py --name improved_cnn
python src/robustness_bench.py --name improved_cnn
```

### 2.3 Expected Outcomes

- Clean test accuracy remains near Phase 2 levels (slight drop acceptable, the model now sees harder training examples)
- Mean Corruption Accuracy (mCA) increases, demonstrating real robustness improvement
- The robustness gap (Clean Acc − mCA) narrows

---

## 3. Phase 4B — Transfer Learning

### 3.1 Architecture Justification

#### ResNet-18

ResNet-18 uses **residual connections** (skip connections) that allow gradients to flow directly through the network, enabling training of much deeper networks without vanishing gradients. It has 8 residual blocks (16 conv layers + 1 initial conv + FC), totalling **~11.7M parameters**.

Key advantage for GTSRB: The ImageNet features (edges, corners, texture gradients, shape detectors) transfer strongly to traffic sign classification which requires detecting geometric shapes (circles, triangles) and interior symbol patterns.

#### MobileNetV2

MobileNetV2 uses **inverted residual blocks with depthwise separable convolutions** — factorizing a standard convolution into a depthwise convolution (per-channel spatial filtering) + pointwise convolution (channel mixing). This reduces computation by ~8× while preserving accuracy.

Key advantage: **~3.4M parameters** total, making it suitable for edge deployment. Linear bottlenecks prevent information loss in the narrow latent space.

### 3.2 Two-Stage Fine-Tuning Strategy

```
Stage 1 — Frozen Backbone (epochs 1–5):
  Backbone LR   : 0     (frozen, no updates)
  Head LR       : 1e-3  (only new 43-class head trains)
  Purpose       : Warm up the head without large gradient signals
                  disrupting pretrained ImageNet features

Stage 2 — Full Fine-Tuning (epochs 6–15):
  Backbone LR   : 1e-4  (10× lower than head)
  Head LR       : 1e-3
  Purpose       : Gently adapt pretrained features to GTSRB domain
                  (48×48 traffic signs vs 224×224 ImageNet objects)
```

### 3.3 Input Resolution

Transfer models use **48×48** input instead of 32×32. This provides more spatial detail for the deeper backbone to process while staying computationally feasible. The `get_data_loaders()` function handles this via the `img_size` argument.

### 3.4 New Classification Head Design

Both backbones have their original 1000-class head replaced with:

```
Dropout(0.4)
→ Linear(in_features → 256, bias=False)
→ BatchNorm1D(256)
→ ReLU
→ Dropout(0.2)
→ Linear(256 → 43)
```

The BatchNorm + Dropout combination in the head prevents the head from overfitting during Stage 1 when only it is being trained.

### 3.5 Commands

```bash
# ResNet-18
python src/train.py --model_type resnet18 --epochs 15 --use_aug --stage1_epochs 5 --name resnet18
python src/evaluate.py --name resnet18
python src/robustness_bench.py --name resnet18

# MobileNetV2
python src/train.py --model_type mobilenet_v2 --epochs 15 --use_aug --stage1_epochs 5 --name mobilenet_v2
python src/evaluate.py --name mobilenet_v2
python src/robustness_bench.py --name mobilenet_v2
```

---

## 4. Four-Way Model Comparison

The fair comparison uses the **same test set** (`Test.csv`, 12,630 images) and **identical 15-condition robustness benchmark** for all models.

| Model | Parameters | Input Size | Clean Acc | Macro-F1 | mCA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline CNN (scratch) | 823K | 32×32 | 97.98% | 97.23% | *Run Phase 3* |
| Improved CNN (+aug) | 823K | 32×32 | *Measured* | *Measured* | *Measured* |
| ResNet-18 (fine-tuned) | 11.7M | 48×48 | *Measured* | *Measured* | *Measured* |
| MobileNetV2 (fine-tuned) | 3.4M | 48×48 | *Measured* | *Measured* | *Measured* |

Results saved to: `results/model_comparison.csv`

---

## 5. Code Deliverables for Phase 4

- **Updated Models:** [`src/models.py`](../src/models.py) — Added `TransferModel`, `get_transfer_model`, `count_all_parameters`
- **Updated Training Pipeline:** [`src/train.py`](../src/train.py) — Added `--model_type`, two-stage fine-tuning, differential LR
- **Updated Evaluation:** [`src/evaluate.py`](../src/evaluate.py) — Auto-detects model type from checkpoint
- **Updated Robustness Bench:** [`src/robustness_bench.py`](../src/robustness_bench.py) — Supports all model types
- **Phase 4 Notebook:** [`notebooks/03_transfer_learning.ipynb`](../notebooks/03_transfer_learning.ipynb)
- **Saved Figures:**
  - `results/figures/phase4a_augmentation_comparison.png`
  - `results/figures/four_way_model_comparison.png`
  - `results/figures/all_models_robustness_overlay.png`
  - `results/figures/phase4_learning_curves_comparison.png`

---

## 6. Next Step: Phase 5 — Grad-CAM Explainability

With all four models trained and benchmarked, Phase 5 generates Grad-CAM heatmaps to explain *where* each model looks when predicting a traffic sign class — on both clean and corrupted images.
