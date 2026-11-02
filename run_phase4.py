"""
RoadSense — Phase 4 Complete Runner
====================================
Runs the full Phase 4 pipeline in order:
  Step 1: Phase 4A — Train Improved CNN with augmentation
  Step 2: Phase 4A — Evaluate Improved CNN (clean test set)
  Step 3: Phase 4A — Robustness benchmark for Improved CNN
  Step 4: Phase 4B — Train ResNet-18 (two-stage fine-tuning)
  Step 5: Phase 4B — Evaluate ResNet-18
  Step 6: Phase 4B — Robustness benchmark for ResNet-18
  Step 7: Phase 4B — Train MobileNetV2 (two-stage fine-tuning)
  Step 8: Phase 4B — Evaluate MobileNetV2
  Step 9: Phase 4B — Robustness benchmark for MobileNetV2
  Step 10: Generate four-way comparison table and plots

Usage:
    python run_phase4.py

Requires:
    - data/Train/ and data/Test/ image folders (run src/download_data.py first)
    - checkpoints/best_baseline_cnn.pth  (run Phase 2 first via src/train.py)
    - All dependencies from requirements.txt
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.train import train
from src.evaluate import evaluate
from src.robustness_bench import run_robustness_benchmark
from src.utils import set_seed

# ──────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}\n")


def check_prerequisites():
    """Exit early with a clear message if data or baseline checkpoint are missing."""
    errors = []
    if not (ROOT_DIR / "data" / "Train").exists():
        errors.append("  • data/Train/ folder missing — run: python src/download_data.py")
    if not (ROOT_DIR / "data" / "Test").exists():
        errors.append("  • data/Test/  folder missing — run: python src/download_data.py")
    ckpt = ROOT_DIR / "checkpoints" / "best_baseline_cnn.pth"
    if not ckpt.exists():
        errors.append(
            "  • checkpoints/best_baseline_cnn.pth missing — "
            "run: python src/train.py --model_type baseline_cnn --epochs 15 --name baseline_cnn"
        )
    if errors:
        print("\n[ERROR] Prerequisites not met:\n")
        for e in errors:
            print(e)
        print()
        sys.exit(1)


def get_mca(model_name: str) -> float:
    """Read mean corruption accuracy from saved robustness CSV."""
    p = ROOT_DIR / "results" / f"{model_name}_robustness.csv"
    if p.exists():
        return round(pd.read_csv(p)["Accuracy_%"].mean(), 2)
    return float("nan")


# ──────────────────────────────────────────────────
# Comparison table + plots
# ──────────────────────────────────────────────────

def generate_comparison_plots():
    section("Generating Four-Way Comparison Plots")

    RESULTS_DIR = ROOT_DIR / "results"
    FIG_DIR = RESULTS_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    metrics_csv = RESULTS_DIR / "metrics.csv"
    if not metrics_csv.exists():
        print("No metrics.csv found — skipping comparison plots.")
        return

    df_metrics = pd.read_csv(metrics_csv)

    model_order = ["baseline_cnn", "improved_cnn", "resnet18", "mobilenet_v2"]
    display_names = {
        "baseline_cnn":  "Baseline CNN\n(scratch, no aug)",
        "improved_cnn":  "Improved CNN\n(scratch + aug)",
        "resnet18":      "ResNet-18\n(fine-tuned)",
        "mobilenet_v2":  "MobileNetV2\n(fine-tuned)",
    }
    params_map = {
        "baseline_cnn": "823K",
        "improved_cnn": "823K",
        "resnet18":     "11.7M",
        "mobilenet_v2": "3.4M",
    }

    rows = []
    for m in model_order:
        row_m = df_metrics[df_metrics["Model"] == m]
        if row_m.empty:
            continue
        row_m = row_m.iloc[0]
        rows.append({
            "Model":             display_names.get(m, m),
            "Parameters":        params_map.get(m, "?"),
            "Clean_Acc":         row_m["Clean_Accuracy_%"],
            "Macro_F1":          row_m["Macro_F1_%"],
            "mCA":               get_mca(m),
            "model_id":          m,
        })

    if not rows:
        print("No model metrics found to compare.")
        return

    df_compare = pd.DataFrame(rows)
    df_compare.drop(columns=["model_id"]).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(f"Saved comparison table → results/model_comparison.csv")
    print(df_compare.drop(columns=["model_id"]).to_string(index=False))

    # ── Plot 1: Grouped bar — Clean Acc vs mCA ──
    labels      = df_compare["Model"].tolist()
    clean_accs  = df_compare["Clean_Acc"].tolist()
    mca_accs    = df_compare["mCA"].tolist()

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - w/2, clean_accs, w, label="Clean Test Accuracy",         color="#2b5c8f", alpha=0.9)
    b2 = ax.bar(x + w/2, mca_accs,   w, label="Mean Corruption Accuracy (mCA)", color="#e07a5f", alpha=0.9)

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("RoadSense: Four-Way Model Comparison\nClean Accuracy vs Mean Corruption Accuracy (mCA)", fontsize=13)
    ax.set_ylim(0, 108)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "four_way_model_comparison.png", dpi=300)
    plt.close()
    print(f"Saved → results/figures/four_way_model_comparison.png")

    # ── Plot 2: Phase 4A before/after augmentation ──
    b_path = RESULTS_DIR / "baseline_cnn_robustness.csv"
    i_path = RESULTS_DIR / "improved_cnn_robustness.csv"
    if b_path.exists() and i_path.exists():
        db = pd.read_csv(b_path).groupby("Corruption")["Accuracy_%"].mean().reset_index()
        di = pd.read_csv(i_path).groupby("Corruption")["Accuracy_%"].mean().reset_index()
        mg = db.merge(di, on="Corruption", suffixes=("_baseline", "_improved"))
        mg["Gain"] = mg["Accuracy_%_improved"] - mg["Accuracy_%_baseline"]
        mg["Corruption"] = mg["Corruption"].str.replace("_", " ").str.title()

        x2 = np.arange(len(mg))
        fig, ax2 = plt.subplots(figsize=(13, 6))
        ax2.bar(x2 - w/2, mg["Accuracy_%_baseline"], w, label="Baseline CNN (no aug)", color="#2b5c8f", alpha=0.85)
        ax2.bar(x2 + w/2, mg["Accuracy_%_improved"],  w, label="Improved CNN (+aug)",   color="#81b29a", alpha=0.85)
        for i, (_, row) in enumerate(mg.iterrows()):
            gain = row["Gain"]
            colour = "#2d6a4f" if gain >= 0 else "#c0392b"
            ax2.text(x2[i] + w/2, row["Accuracy_%_improved"] + 1,
                     f"+{gain:.1f}%" if gain >= 0 else f"{gain:.1f}%",
                     ha="center", va="bottom", fontsize=9, color=colour, fontweight="bold")
        ax2.set_xticks(x2)
        ax2.set_xticklabels(mg["Corruption"], rotation=15, ha="right")
        ax2.set_ylabel("Mean Accuracy (%) across 3 severity levels")
        ax2.set_title("Phase 4A: Robustness Improvement through Label-Safe Augmentation")
        ax2.legend(); ax2.set_ylim(0, 108); ax2.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "phase4a_augmentation_comparison.png", dpi=300)
        plt.close()
        print(f"Saved → results/figures/phase4a_augmentation_comparison.png")

    # ── Plot 3: Robustness overlay per corruption ──
    model_styles = [
        ("baseline_cnn",  "Baseline CNN",       "#2b5c8f", "-"),
        ("improved_cnn",  "Improved CNN (+aug)", "#81b29a", "--"),
        ("resnet18",      "ResNet-18",           "#e07a5f", "-."),
        ("mobilenet_v2",  "MobileNetV2",         "#f4a261", ":"),
    ]
    corruption_types = ["gaussian_noise", "motion_blur", "brightness_contrast",
                        "random_occlusion", "jpeg_compression"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    for ax_c, corr in zip(axes, corruption_types):
        for mid, label, color, ls in model_styles:
            p = RESULTS_DIR / f"{mid}_robustness.csv"
            if not p.exists():
                continue
            df_c = pd.read_csv(p)
            df_c = df_c[df_c["Corruption"] == corr].sort_values("Severity")
            ax_c.plot([1, 2, 3], df_c["Accuracy_%"].values,
                      marker="o", linestyle=ls, color=color, label=label, linewidth=2)
        ax_c.set_title(corr.replace("_", "\n").title(), fontsize=9)
        ax_c.set_xlabel("Severity"); ax_c.set_xticks([1, 2, 3])
        ax_c.set_ylim(0, 105); ax_c.grid(True, alpha=0.3)
    axes[0].set_ylabel("Accuracy (%)")
    handles = [mpatches.Patch(color=c, label=l) for _, l, c, _ in model_styles]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    plt.suptitle("Robustness Degradation Curves: All Models vs 5 Corruption Types", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "all_models_robustness_overlay.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → results/figures/all_models_robustness_overlay.png")

    # ── Plot 4: Learning curves overlay ──
    model_hist = [
        ("baseline_cnn",  "Baseline CNN",       "#2b5c8f"),
        ("improved_cnn",  "Improved CNN (+aug)", "#81b29a"),
        ("resnet18",      "ResNet-18",           "#e07a5f"),
        ("mobilenet_v2",  "MobileNetV2",         "#f4a261"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    for mid, label, color in model_hist:
        hp = RESULTS_DIR / f"{mid}_history.json"
        if not hp.exists():
            continue
        with open(hp) as f:
            h = json.load(f)
        ep = range(1, len(h["val_acc"]) + 1)
        axs[0].plot(ep, h["val_loss"],           marker=".", color=color, label=label)
        axs[1].plot(ep, [a*100 for a in h["val_acc"]], marker=".", color=color, label=label)

    for axi, (ytitle, title) in enumerate([
        ("Loss",         "Validation Loss — All Models"),
        ("Accuracy (%)", "Validation Accuracy — All Models"),
    ]):
        axs[axi].set_title(title); axs[axi].set_xlabel("Epoch")
        axs[axi].set_ylabel(ytitle); axs[axi].legend(); axs[axi].grid(True, alpha=0.3)
    plt.suptitle("Phase 4: Learning Curves Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phase4_learning_curves_comparison.png", dpi=300)
    plt.close()
    print(f"Saved → results/figures/phase4_learning_curves_comparison.png")


# ──────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────

def main():
    set_seed(42)
    t_start = time.time()

    check_prerequisites()

    # ── Phase 4A ──────────────────────────────────
    section("Phase 4A — Improved CNN (label-safe augmentation)")
    train(
        epochs=15, batch_size=64, lr=0.001,
        use_augmentation=True, use_class_weights=True,
        model_type="improved_cnn", model_name="improved_cnn",
        stage1_epochs=5,
    )
    evaluate(model_name="improved_cnn")
    run_robustness_benchmark(model_name="improved_cnn")

    # ── Phase 4B — ResNet-18 ──────────────────────
    section("Phase 4B — ResNet-18 (two-stage fine-tuning)")
    train(
        epochs=15, batch_size=64, lr=0.001,
        use_augmentation=True, use_class_weights=True,
        model_type="resnet18", model_name="resnet18",
        stage1_epochs=5,
    )
    evaluate(model_name="resnet18")
    run_robustness_benchmark(model_name="resnet18")

    # ── Phase 4B — MobileNetV2 ───────────────────
    section("Phase 4B — MobileNetV2 (two-stage fine-tuning)")
    train(
        epochs=15, batch_size=64, lr=0.001,
        use_augmentation=True, use_class_weights=True,
        model_type="mobilenet_v2", model_name="mobilenet_v2",
        stage1_epochs=5,
    )
    evaluate(model_name="mobilenet_v2")
    run_robustness_benchmark(model_name="mobilenet_v2")

    # ── Comparison plots ─────────────────────────
    generate_comparison_plots()

    elapsed = (time.time() - t_start) / 60
    section(f"Phase 4 COMPLETE — Total time: {elapsed:.1f} minutes")
    print("Generated files:")
    print("  checkpoints/best_improved_cnn.pth")
    print("  checkpoints/best_resnet18.pth")
    print("  checkpoints/best_mobilenet_v2.pth")
    print("  results/model_comparison.csv")
    print("  results/figures/four_way_model_comparison.png")
    print("  results/figures/phase4a_augmentation_comparison.png")
    print("  results/figures/all_models_robustness_overlay.png")
    print("  results/figures/phase4_learning_curves_comparison.png")
    print("\nNext step: open notebooks/04_evaluation_and_explainability.ipynb for Phase 5 Grad-CAM.")


if __name__ == "__main__":
    main()
