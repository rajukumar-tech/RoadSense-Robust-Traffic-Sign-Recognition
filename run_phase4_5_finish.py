"""
RoadSense - Finish Phase 4 comparison plots + run full Phase 5
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# ── Phase 4: Generate comparison plots ───────────────
print("\n" + "="*65)
print("  Generating Phase 4 Comparison Plots")
print("="*65)
from run_phase4 import generate_comparison_plots
generate_comparison_plots()
print("Phase 4 comparison plots done!")

# ── Phase 5: Full explainability on all models ───────
print("\n" + "="*65)
print("  Phase 5 - Grad-CAM Explainability (all 4 models)")
print("="*65)
from run_phase5 import run_phase5
run_phase5(model_name="baseline_cnn", run_all=True)
print("Phase 5 complete!")
