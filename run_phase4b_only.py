"""
RoadSense — Phase 4B Only Runner
Trains ResNet-18 and MobileNetV2 (pretrained weights already cached locally).
Phase 4A (improved_cnn) already completed — this picks up from Phase 4B.
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.train import train
from src.evaluate import evaluate
from src.robustness_bench import run_robustness_benchmark
from src.utils import set_seed
from run_phase4 import generate_comparison_plots, section

set_seed(42)
t_start = time.time()

# ── Phase 4B — ResNet-18 ──────────────────────────────
section("Phase 4B — ResNet-18 (two-stage fine-tuning)")
train(
    epochs=15, batch_size=64, lr=0.001,
    use_augmentation=True, use_class_weights=True,
    model_type="resnet18", model_name="resnet18",
    stage1_epochs=5,
)
evaluate(model_name="resnet18")
run_robustness_benchmark(model_name="resnet18")

# ── Phase 4B — MobileNetV2 ───────────────────────────
section("Phase 4B — MobileNetV2 (two-stage fine-tuning)")
train(
    epochs=15, batch_size=64, lr=0.001,
    use_augmentation=True, use_class_weights=True,
    model_type="mobilenet_v2", model_name="mobilenet_v2",
    stage1_epochs=5,
)
evaluate(model_name="mobilenet_v2")
run_robustness_benchmark(model_name="mobilenet_v2")

# ── Comparison plots ─────────────────────────────────
generate_comparison_plots()

elapsed = (time.time() - t_start) / 60
section(f"Phase 4B COMPLETE — Total time: {elapsed:.1f} minutes")
print("Generated files:")
print("  checkpoints/best_resnet18.pth")
print("  checkpoints/best_mobilenet_v2.pth")
print("  results/model_comparison.csv")
print("  results/figures/four_way_model_comparison.png")
print("  results/figures/phase4a_augmentation_comparison.png")
print("  results/figures/all_models_robustness_overlay.png")
print("  results/figures/phase4_learning_curves_comparison.png")
