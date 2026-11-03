# Step 4: Phase 3 — Robustness Test Bench

**Group 4 | Atria University | RoadSense Project**  
**Owner:** Raju (Code & Testing)  
**Phase:** 3 of 6

---

## Objective

The Baseline CNN achieved **97.98% accuracy** on clean, ideal-quality test images. Phase 3 asks:

> *Does the model maintain this performance when images are degraded by real-world driving conditions?*

We stress-test the trained model under **5 realistic corruption types** across **3 severity levels** — a total of **15 test conditions** — on the full 12,630-image GTSRB test set.

---

## The 5 Corruption Types

| # | Corruption | Real-World Scenario | Severity Parameters |
|---|---|---|---|
| 1 | **Gaussian Noise** | Camera sensor grain in low light / high ISO | σ = 18, 38, 65 |
| 2 | **Motion Blur** | Fast vehicle speed, camera shake | radius = 1.0, 2.0, 3.5 |
| 3 | **Brightness/Contrast** | Sun glare, night driving, tunnel underpass | contrast & brightness factors |
| 4 | **Random Occlusion** | Tree branch, lamppost, sticker on sign | patch = 19%, 31%, 44% of image |
| 5 | **JPEG Compression** | Low-bitrate dashcam video, network loss | quality = 40, 20, 8 |

### Severity Levels

| Level | Description |
|:---:|:---|
| **1 (Mild)** | Barely noticeable — slight degradation |
| **2 (Moderate)** | Clearly visible but sign still readable |
| **3 (Severe)** | Heavily corrupted — hard even for humans |

---

## Implementation

### Files

| File | Description |
|:---|:---|
| [`src/corruptions.py`](../src/corruptions.py) | 5 corruption functions using PIL + NumPy (no OpenCV dependency) |
| [`src/robustness_bench.py`](../src/robustness_bench.py) | Full benchmark runner — loads model, applies corruptions, saves CSV & plots |
| [`notebooks/03_robustness.ipynb`](../notebooks/03_robustness.ipynb) | Interactive notebook documenting Phase 3 end-to-end |

### Corruption Design Decisions

- **Deterministic seeding:** Every corruption uses `seed = 42 + image_index` for full reproducibility
- **PIL-only pipeline:** No OpenCV required — all corruptions use PIL, ImageFilter, and NumPy
- **On-the-fly application:** Corruptions are applied in `__getitem__` of `CorruptedTestDataset` — no disk writes
- **Label safety:** Corruptions are purely visual and do not alter class identity

---

## How to Run

```bash
# From project root
python src/robustness_bench.py --name baseline_cnn

# Optional: subsample for speed (e.g., 2000 images)
python src/robustness_bench.py --name baseline_cnn --sample 2000

# Or run the interactive notebook
jupyter notebook notebooks/03_robustness.ipynb
```

---

## Results

### Clean vs Corrupted Accuracy

| Metric | Value |
|:---|:---:|
| **Clean Test Accuracy** | **97.98%** |
| **Mean Corruption Accuracy (mCA)** | *(run benchmark to fill)* |
| **Robustness Gap** | *(Clean − mCA)* |

### Per-Condition Results

| Corruption | Severity 1 | Severity 2 | Severity 3 | Avg Drop |
|:---|:---:|:---:|:---:|:---:|
| Gaussian Noise | — | — | — | — |
| Motion Blur | — | — | — | — |
| Brightness/Contrast | — | — | — | — |
| Random Occlusion | — | — | — | — |
| JPEG Compression | — | — | — | — |

> **Note:** Run `python src/robustness_bench.py --name baseline_cnn` to populate the table above.  
> Results are saved to `results/baseline_cnn_robustness.csv`.

---

## Output Artifacts

| Artifact | Path |
|:---|:---|
| Robustness results CSV | `results/baseline_cnn_robustness.csv` |
| Degradation curves plot | `results/figures/baseline_cnn_robustness_curves.png` |
| Heatmap (all 15 conditions) | `results/figures/baseline_cnn_robustness_heatmap.png` |
| Accuracy drop bar chart | `results/figures/baseline_cnn_robustness_drop_bar.png` |
| Corruption samples grid | `results/figures/phase3_corruption_samples_grid.png` |

---

## Key Findings & Implications

- **Robustness matters:** Real dashcam images are never clean — a model that fails on degraded images cannot be deployed safely.
- **mCA < Clean Accuracy:** The gap (Robustness Gap) tells us how much the model degrades under real-world noise.
- **Worst corruption → Phase 4 target:** Whichever corruption drops accuracy the most becomes the primary target for improvement via transfer learning and augmentation in Phase 4.

---

## Connection to Other Phases

| Phase | Connection |
|:---|:---|
| **Phase 2** | Baseline CNN checkpoint (`best_baseline_cnn.pth`) is the model being tested here |
| **Phase 4** | Transfer learning (ResNet-18/MobileNetV2) aims to close the Robustness Gap identified in Phase 3 |
| **Phase 5** | Grad-CAM explainability will visualise *why* certain corruptions confuse the model |

---

*Documentation maintained by Aseem (Phase 6)*
