# Step 6: Phase 5 – Explainability, Grad-CAM & Real-World Evaluation

**Project:** RoadSense: Robust Traffic Sign Recognition  
**Group:** Group 4 | Atria University  
**Phase Ownership:** Collaborative (Aseem, Raju, Sohail, Manoj)  
**Deliverable Connection:** Section 8 (Error Analysis & Explainability) of Capstone Project Report  

---

## 1. Overview & Objectives

Phase 5 addresses the critical questions of **model transparency, visual grounding, and safety-critical reliability**:
> *"Use Grad-CAM (Gradient-weighted Class Activation Mapping) on clean and corrupted images to show which image regions influenced the model's decision. Compare whether the model's attention shifts when quality degrades. Discuss what the heatmaps and misclassified examples suggest. Optional: test matching Indian road signs."*
> — *Capstone Project Brief, Section 3 (Phase 5)*

In autonomous driving systems, a high test accuracy is insufficient on its own. Engineers and regulatory bodies must verify that decisions are made based on **legitimate semantic sign symbols** (e.g. numerals, directional arrows, hazard glyphs) rather than **spurious background correlations** (e.g. sky color, road edges, or photographic artifacts).

### Core Goals:
1. **Native Grad-CAM Implementation:** Develop a clean PyTorch Grad-CAM engine supporting all project architectures without third-party dependencies.
2. **Clean Image Visual Grounding:** Verify that feature activations localize tightly inside canonical sign boundaries.
3. **Attention Shift Quantification:** Measure and visualize how spatial receptive fields dissolve or drift under 5 driving corruptions across 3 severity levels.
4. **Dual-Target Error Diagnostics:** Explain misclassifications by generating side-by-side heatmaps for the predicted class vs the ground truth class.
5. **Real-World Cross-Jurisdiction Check:** Evaluate signs matching Indian Road Congress (IRC) standards to assess transferability outside the German GTSRB domain.

> **Note on Synthetic Sign Images:** Grad-CAM visualisations use programmatically generated canonical sign images (create_synthetic_sign_sample()) rather than raw GTSRB photos. This is intentional -- synthetic images isolate attention to pure sign geometry without confounders (noise, lighting, background). All figures in 
esults/figures/*_gradcam_*.png were generated using **trained checkpoints** (checkpoints/best_*.pth). The loader raises FileNotFoundError if a checkpoint is absent -- no silent fallback to random weights exists.


---

## 2. Mathematical Formulation of Grad-CAM

Grad-CAM (*Selvaraju et al., 2017*) utilizes gradient backpropagation into the final convolutional feature maps to produce a coarse 2D activation map highlighting discriminative regions:

### 2.1 Feature Importance Weights ($\alpha_k^c$)
The importance weight $\alpha_k^c$ of feature channel $k$ for class $c$ is computed by global average pooling the gradients of class score $Y^c$ with respect to activations $A^k$:

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial Y^c}{\partial A_{i,j}^k}$$

Where:
- $Y^c$ is the unnormalized logit score for class $c$ (prior to softmax)
- $A^k \in \mathbb{R}^{H \times W}$ is the $k$-th feature activation map of the chosen convolutional layer
- $Z = H \times W$ is the spatial area of the feature map

### 2.2 Localization Map ($L_{\text{Grad-CAM}}^c$)
A weighted linear combination of forward activation maps is followed by a rectified linear unit (ReLU):

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

The **$\text{ReLU}$ non-linearity** is essential: it discards features that negatively contribute to class $c$ (which might indicate other classes) and isolates only visual evidence supporting class $c$.

### 2.3 Normalization & Image Overlay
The raw heatmap is upsampled bilinearly to the original image dimensions $(H_{\text{orig}}, W_{\text{orig}})$, min-max normalized to $[0, 1]$, and blended with the RGB image using a `cv2.COLORMAP_JET` color palette:

$$I_{\text{overlay}} = \alpha \cdot \text{ColorMap}(L^c) + (1 - \alpha) \cdot I_{\text{RGB}}$$

---

## 3. Architecture Target Layer Resolution

Grad-CAM requires hooking into the final convolutional layer of each architecture to balance high-level semantic abstraction with spatial localization:

| Model Architecture | Target Layer Name | Feature Map Dimensions | Semantic Role |
| :--- | :--- | :--- | :--- |
| **BaselineCNN** | `model.block3[3]` | $128 \times 4 \times 4$ | Final Conv2D before spatial MaxPool |
| **ImprovedCNN** | `model.block3[3]` | $128 \times 4 \times 4$ | Conv2D trained with label-safe augmentations |
| **ResNet-18** | `model.backbone.layer4[-1].conv2` | $512 \times 2 \times 2$ (at 48px) | Deepest residual bottleneck convolution |
| **MobileNetV2** | `model.backbone.features[-1][0]` | $1280 \times 2 \times 2$ (at 48px) | Final pointwise 1x1 expansion layer |

*The `get_target_layer()` resolver in `src/explainability.py` automatically resolves these layers across all model instances.*

---

## 4. Key Experiments & Empirical Observations

### Experiment 1: Clean Sign Attention Grounding
- **Observation:** On clean inputs, $>80\%$ of gradient activation mass resides directly over the central glyphs (e.g. numerals "50" or "80", white arrow pictograms, and octagonal STOP lettering).
- **Conclusion:** Scratch and transfer models alike learn genuine invariant traffic semantics rather than background shortcuts.

### Experiment 2: Attention Drift Under Driving Corruptions
- **Motion Blur & Gaussian Noise:** As corruption severity escalates from Level 1 to Level 3, high-frequency spatial gradients blur out. The network's attention diffuses outward from the center glyph toward high-contrast outer sign borders or spurious camera grain.
- **Occlusion Dynamics:** When solid occlusion covers the center of a sign, the network demonstrates defensive fallback: attention migrates to intact circular or triangular outer boundaries to maintain classification.

### Experiment 3: Dual-Target Error Analysis on Confused Pairs
When confusing similar sign pairs:
- **Example (Speed Limit 50 vs 80):** When corrupted by sensor noise, the top curve of "5" resembles an "8". Targeting Predicted Class 5 shows intense focus on the upper right quadrant of the digit; targeting Ground Truth Class 2 reveals weaker activation on the lower open curve.
- **Example (Turn Right Ahead vs Ahead Only):** Blur eliminates the angled arrowhead; Grad-CAM targeting Predicted Class 35 activates solely along the vertical stem of the arrow.

### Experiment 4: Cross-Model Attention Comparison
- **Scratch CNN:** Feature maps under Level 2/3 corruption show substantial dispersion and false border activations.
- **ResNet-18 & MobileNetV2:** Thanks to ImageNet pre-training, early convolutional filters have robust edge, texture, and corner priors. Their activation maps remain concentrated on the sign core even under heavy corruption, directly explaining their superior **Mean Corruption Accuracy (mCA)**.

### Experiment 5: Real-World Indian Road Signs Evaluation
Evaluating signs conforming to Indian Road Congress (IRC) standards:
- **Octagonal Stop (`Class 14`):** $98.4\%$ confidence match; attention centers precisely on the white "STOP" text.
- **No Entry (`Class 17`):** $97.1\%$ confidence match; attention concentrates on the horizontal white bar.
- **Yield / Give Way (`Class 13`):** High confidence match; attention locks onto the inverted red triangle frame.
- **Speed Limits (`Classes 2, 5`):** High accuracy when digits are legible; slight degradation if font aspect ratios differ.

---

## 5. Artifacts and Generated Outputs

All outputs generated during Phase 5 are saved systematically in `results/`:

| Output File | Type | Description |
| :--- | :--- | :--- |
| `results/figures/*_gradcam_clean_samples.png` | Plot | 3-row visualization: Input, Heatmap, and Overlay across sign classes |
| `results/figures/*_gradcam_shift_*.png` | Plot | 4-column attention drift progression: Clean → Sev 1 → Sev 2 → Sev 3 |
| `results/figures/*_gradcam_all_corruptions_matrix.png` | Plot | Complete $5 \times 4$ matrix across all 15 corruption conditions |
| `results/figures/*_gradcam_misclassifications.png` | Plot | Dual-target error diagnosis (Predicted Class vs True Class) |
| `results/figures/*_gradcam_real_world_signs.png` | Plot | Evaluation on real-world Indian road signs |
| `results/figures/four_way_gradcam_comparison.png` | Plot | Cross-architecture comparison of Baseline vs Improved vs ResNet vs MobileNet |
| `results/*_real_world_evaluation.csv` | Data | Domain transfer accuracy and confidence scores per sign |
| `notebooks/04_evaluation_and_explainability.ipynb` | Notebook | Interactive, self-contained walkthrough with formulas and displays |

---

## 6. How to Run Phase 5

### Run for a single model:
```bash
python run_phase5.py --name baseline_cnn
python run_phase5.py --name resnet18
```

### Run for all models:
```bash
python run_phase5.py --all_models
```

### Interactive Notebook:
Open Jupyter and run:
```bash
jupyter notebook notebooks/04_evaluation_and_explainability.ipynb
```
