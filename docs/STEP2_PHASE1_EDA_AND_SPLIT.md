# Step 2: Phase 1 — Exploratory Data Analysis (EDA) & Leakage-Free Validation Split

**Project:** RoadSense: Robust Traffic Sign Recognition  
**Group:** Group 4 | Atria University  
**Team Members:** Aseem (Team Leader), Raju, Sohail, Manoj  

---

## 1. Overview & Objectives
Phase 1 focuses on thoroughly understanding the GTSRB dataset before constructing models. As emphasized in the Capstone Project Brief, building a sound perception pipeline requires addressing **class imbalance**, establishing **label-safe data transformations**, and eliminating **track-based data leakage** between training and validation partitions.

---

## 2. Key Findings & Empirical Statistics

### A. Class Distribution & Imbalance
The dataset contains 43 distinct classes with severe real-world frequency imbalance:
- **Total Training Images:** 39,209
- **Total Clean Test Images:** 12,630
- **Most Populated Class:** Class 2 (*Speed limit 50 km/h*) with **2,250 images**
- **Least Populated Class:** Class 0 (*Speed limit 20 km/h*) with **210 images**
- **Class Imbalance Ratio:** **10.71x** ($\approx 2,250 / 210$)

> **Modeling Implication:** Standard overall accuracy can be misleading due to speed limit signs dominating the dataset. In accordance with the brief, all subsequent evaluations will report **Macro-F1** and per-class precision/recall alongside overall accuracy.

---

### B. Image Dimension & Geometry Analysis
- **Width:** Minimum 25 px, Median 43.0 px, Mean 50.8 px, Maximum 243 px
- **Height:** Minimum 25 px, Median 43.0 px, Mean 50.3 px, Maximum 225 px
- **Aspect Ratio:** Mean = $1.007 \pm 0.071$

> **Preprocessing Implication:** Traffic signs in GTSRB are uniformly square-cropped bounding boxes. This justifies standard resizing to fixed dimensions of **$32 \times 32$** or **$48 \times 48$** pixels without aspect distortion.

---

### C. Brightness & Contrast Variations
- Real-world dashcam images capture extreme environmental conditions, from high-noon glare and sun flare to dim overcast weather, dusk, and shadowed underpasses.
- Mean luminance spans across the entire 0–255 spectrum with a high standard deviation in local contrast.

---

## 3. Disjoint Track-Based Validation Split (Zero-Leakage Architecture)

### The Threat of "Track Leakage"
In GTSRB, training images were acquired from dashcam video sequences recorded at 25–30 frames per second. Consecutive frames capture the **exact same physical sign** as the vehicle drives closer:
$$\text{Filename Scheme: } \texttt{000}\langle\text{ClassId}\rangle\_\langle\text{TrackId}\rangle\_\langle\text{FrameId}\rangle\texttt{.png}$$
If individual images are shuffled and randomly split into 80% train / 20% validation:
- ~24 frames of a sign are in training, and ~6 frames of the exact same sign under nearly identical lighting and angle land in validation.
- The model memorizes instance-level background artifacts rather than generalized sign features, producing artificially high validation accuracy (data leakage).

### Our Leakage-Free Solution
We implemented an **atomic track partitioning strategy** in `src/eda_and_split.py`:
1. All unique $\langle\text{ClassId}, \text{TrackId}\rangle$ video sequences were identified.
2. For each class, tracks were shuffled with a fixed random seed (`seed=42`).
3. ~80% of tracks were assigned to Training, and ~20% of tracks (minimum 1 per class) were assigned to Validation.

### Verified Split Results:
- **Training Set (`data/train_split.csv`):** 31,379 images (80.0%)
- **Validation Set (`data/val_split.csv`):** 7,830 images (20.0%)
- **Leakage Verification:** $\mathbf{0}$ overlapping tracks between training and validation partitions ($\text{Train} \cap \text{Val} = \emptyset$).

---

## 4. Discussion: Label-Safe vs Dangerous Data Augmentations

The capstone brief requires addressing:
> *"Careful with augmentation: horizontally flipping a 'keep left' sign turns it into a 'keep right' sign. Discuss which augmentations are label-safe."*

| Augmentation Technique | Safety Classification | Impact on Semantic Labels |
| :--- | :---: | :--- |
| **Horizontal Flip** | ❌ **DANGEROUS** | **Fatal label corruption.** Transforms *Keep Left* (Class 39) into *Keep Right* (Class 38); transforms *Curve Left* (Class 19) into *Curve Right* (Class 20). Must **not** be applied unconditionally. |
| **Vertical Flip** | ❌ **DANGEROUS** | **Fatal label corruption.** Inverts directional arrows and turns warning triangles upside down. |
| **Small Rotation ($\pm 10^\circ$)** |  **LABEL-SAFE** | Simulates vehicle camera roll and road inclination while maintaining sign legibility. |
| **Color Jitter (Brightness / Contrast)** |  **LABEL-SAFE** | Simulates sunny glare, tunnel darkness, and twilight driving. Label is 100% preserved. |
| **Gaussian Noise** |  **LABEL-SAFE** | Simulates low-light sensor noise and ISO sensor grain. |
| **Motion Blur / Gaussian Blur** |  **LABEL-SAFE** | Simulates vehicle speed and camera focal blur. |
| **Random Occlusion / Cutout** |  **LABEL-SAFE** | Simulates signs partially obscured by branches, lampposts, or dirt patches. |

---

## 5. Artifacts and Figures Generated

All figures are rendered at 300 DPI and saved in `results/figures/`:
1. [class_distribution.png](file:///c:/Users/Raju%20Kumar/Downloads/CNN-TRAFFIC/results/figures/class_distribution.png): Visualizing class imbalance across all 43 classes.
2. [sample_grid_43_classes.png](file:///c:/Users/Raju%20Kumar/Downloads/CNN-TRAFFIC/results/figures/sample_grid_43_classes.png): 7×7 grid displaying clear samples of all 43 sign categories.
3. [image_size_distribution.png](file:///c:/Users/Raju%20Kumar/Downloads/CNN-TRAFFIC/results/figures/image_size_distribution.png): Width vs Height distribution and aspect ratio scatter.
4. [brightness_distribution.png](file:///c:/Users/Raju%20Kumar/Downloads/CNN-TRAFFIC/results/figures/brightness_distribution.png): Luminance and contrast analysis.
5. [track_split_distribution.png](file:///c:/Users/Raju%20Kumar/Downloads/CNN-TRAFFIC/results/figures/track_split_distribution.png): Disjoint train/validation track breakdown per class.

---

## 6. Code Deliverables for Phase 1
- **Standalone EDA & Split Script:** [src/eda_and_split.py](file:///c:/Users/Raju%20Kumar/Downloads/CNN-TRAFFIC/src/eda_and_split.py)
- **Interactive Jupyter Notebook:** [notebooks/01_eda.ipynb](file:///c:/Users/Raju%20Kumar/Downloads/CNN-TRAFFIC/notebooks/01_eda.ipynb)
- **Split CSV Files:** `data/train_split.csv` and `data/val_split.csv`
- **Class Mapping Utilities:** [src/utils.py](file:///c:/Users/Raju%20Kumar/Downloads/CNN-TRAFFIC/src/utils.py)

---

## 7. Next Step: Phase 2 — Baseline CNN from Scratch
Now that the data is partitioned without leakage and the preprocessing constraints are set, we will design and train a custom Convolutional Neural Network from scratch, evaluate it on the clean test set, plot learning curves, and analyze the confusion matrix.
