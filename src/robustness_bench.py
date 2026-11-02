"""
RoadSense: Robustness Test Bench (Phase 3).
Evaluates models against 5 real-world driving corruptions across 3 severity levels (15 conditions).
Computes Clean Accuracy, per-condition Accuracy, and Mean Corruption Accuracy (mCA).
Generates robustness comparison table and curves.

Supports all model types: baseline_cnn, improved_cnn, resnet18, mobilenet_v2.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.dataset import DATA_DIR, get_transforms
from src.corruptions import CORRUPTIONS, SEVERITIES, apply_corruption
from src.models import BaselineCNN, get_transfer_model

RESULTS_DIR = ROOT_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

TRANSFER_ARCHS = {"resnet18", "mobilenet_v2"}
IMG_SIZE_MAP = {
    "baseline_cnn": 32,
    "improved_cnn": 32,
    "resnet18": 48,
    "mobilenet_v2": 48,
}


class CorruptedTestDataset(Dataset):
    """Dataset applying a specific corruption condition on-the-fly to test images."""

    def __init__(self, df: pd.DataFrame, corruption_type: str, severity: int, base_transform, data_root=DATA_DIR):
        self.df = df
        self.corruption_type = corruption_type
        self.severity = severity
        self.base_transform = base_transform
        self.data_root = data_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_root / row["Path"]
        label = int(row["ClassId"])

        img = Image.open(img_path).convert("RGB")
        corrupted_img = apply_corruption(img, self.corruption_type, severity=self.severity, seed=42 + idx)
        tensor_img = self.base_transform(corrupted_img)
        return tensor_img, label


def _build_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    """Load the correct model class based on 'model_type' stored in the checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "baseline_cnn")

    if model_type in TRANSFER_ARCHS:
        model = get_transfer_model(arch=model_type, num_classes=43, freeze_backbone=False)
    else:
        model = BaselineCNN(num_classes=43)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_type


def evaluate_loader(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def run_robustness_benchmark(
    model_name: str = "baseline_cnn",
    checkpoint_file: str = None,
    sample_size: int = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 65)
    print(f"RUNNING ROBUSTNESS TEST BENCH: {model_name} on {device}")
    print("5 Corruptions x 3 Severity Levels (15 Test Conditions)")
    print("=" * 65)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load test metadata
    test_csv = DATA_DIR / "Test.csv"
    df_test = pd.read_csv(test_csv)
    if sample_size and sample_size < len(df_test):
        df_test = df_test.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Subsampled test set to {sample_size:,} images for speed.")
    else:
        print(f"Evaluating full test set of {len(df_test):,} images.")

    # Load model
    ckpt_path = Path(checkpoint_file) if checkpoint_file else CHECKPOINT_DIR / f"best_{model_name}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    model, model_type = _build_model_from_checkpoint(ckpt_path, device)
    img_size = IMG_SIZE_MAP.get(model_type, 32)
    print(f"Loaded: {ckpt_path}  (type={model_type}, img={img_size}x{img_size})")

    eval_transform = get_transforms(img_size=img_size, is_training=False, use_augmentation=False,
                                     use_imagenet_norm=(model_type in TRANSFER_ARCHS))

    # 1. Clean Baseline Accuracy
    from src.dataset import GTSRBDataset
    clean_ds = GTSRBDataset(csv_file=test_csv, transform=eval_transform)
    clean_loader = DataLoader(clean_ds, batch_size=128, shuffle=False)
    clean_acc = evaluate_loader(model, clean_loader, device)
    print(f"\n--> Clean Test Set Accuracy: {clean_acc * 100:.2f}%\n")

    # 2. Corrupted Conditions
    results = []
    plot_data = {"Corruption": [], "Severity": [], "Accuracy": []}

    for corr in CORRUPTIONS:
        print(f"Evaluating corruption: {corr.upper()}")
        for sev in SEVERITIES:
            ds = CorruptedTestDataset(df_test, corr, sev, eval_transform)
            loader = DataLoader(ds, batch_size=128, shuffle=False)
            acc = evaluate_loader(model, loader, device)
            drop = (clean_acc - acc) * 100

            results.append({
                "Model":          model_name,
                "Corruption":     corr,
                "Severity":       sev,
                "Accuracy_%":     round(acc * 100, 2),
                "Accuracy_Drop_%": round(drop, 2),
            })

            plot_data["Corruption"].append(corr.replace("_", " ").title())
            plot_data["Severity"].append(f"Level {sev}")
            plot_data["Accuracy"].append(acc * 100)

            print(f"  Severity {sev}: {acc * 100:.2f}%  (Drop: -{drop:.2f}%)")

    df_results = pd.DataFrame(results)
    mean_corr_acc = df_results["Accuracy_%"].mean()
    mean_drop = (clean_acc * 100) - mean_corr_acc

    print("\n" + "=" * 65)
    print(f"SUMMARY FOR {model_name}:")
    print(f"  Clean Accuracy                : {clean_acc * 100:.2f}%")
    print(f"  Mean Corruption Accuracy (mCA): {mean_corr_acc:.2f}%")
    print(f"  Mean Robustness Gap (Drop)    : -{mean_drop:.2f}%")
    print("=" * 65)

    # Save CSV
    out_csv = RESULTS_DIR / f"{model_name}_robustness.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"Saved robustness table to {out_csv}")

    # Degradation curves
    plt.figure(figsize=(10, 6))
    df_plot = pd.DataFrame(plot_data)
    palette = sns.color_palette("tab10", n_colors=len(CORRUPTIONS))

    for idx, corr_name in enumerate(df_plot["Corruption"].unique()):
        subset = df_plot[df_plot["Corruption"] == corr_name]
        plt.plot(
            [1, 2, 3],
            subset["Accuracy"].values,
            marker="o", linewidth=2.5,
            label=corr_name, color=palette[idx],
        )

    plt.axhline(
        clean_acc * 100, color="black", linestyle="--",
        label=f"Clean Accuracy ({clean_acc*100:.1f}%)"
    )
    plt.title(
        f"Robustness Benchmark: {model_name}\n"
        f"Degradation Curves Under 5 Real-World Corruptions",
        fontsize=13
    )
    plt.xlabel("Corruption Severity Level", fontsize=11)
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.xticks([1, 2, 3], ["Level 1 (Mild)", "Level 2 (Moderate)", "Level 3 (Severe)"])
    plt.ylim(0, 105)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_fig = FIG_DIR / f"{model_name}_robustness_curves.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved robustness curves to {out_fig}")

    return df_results, clean_acc, mean_corr_acc


def main():
    parser = argparse.ArgumentParser(description="Run Robustness Test Bench on GTSRB Model")
    parser.add_argument("--name",       type=str, default="baseline_cnn", help="Model name identifier")
    parser.add_argument("--checkpoint", type=str, default=None,           help="Checkpoint file path")
    parser.add_argument("--sample",     type=int, default=None,           help="Optional test set sample size")
    args = parser.parse_args()
    run_robustness_benchmark(
        model_name=args.name,
        checkpoint_file=args.checkpoint,
        sample_size=args.sample,
    )


if __name__ == "__main__":
    main()
