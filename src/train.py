"""
RoadSense: Training Pipeline for Traffic Sign Recognition Models.
Supports Baseline CNN and Improved CNN with class-weighting, checkpointing,
and learning curve visualization.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.dataset import get_data_loaders, DATA_DIR
from src.models import BaselineCNN, count_parameters
from src.utils import set_seed

RESULTS_DIR = ROOT_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"


def calculate_class_weights(csv_path: Path, device: torch.device) -> torch.Tensor:
    """
    Computes inverse class frequencies to mitigate severe class imbalance.
    Weights are smoothed with square root to prevent over-penalizing rare classes.
    """
    df = pd.read_csv(csv_path)
    class_counts = df["ClassId"].value_counts().sort_index().values
    total_samples = len(df)
    num_classes = len(class_counts)

    # Smoothed inverse class frequency: sqrt(total / count)
    weights = np.sqrt(total_samples / (num_classes * class_counts))
    weights = weights / weights.sum() * num_classes  # Normalize mean to 1.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False, ncols=80):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False, ncols=80):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_learning_curves(history: dict, model_name: str, out_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axs[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#2b5c8f")
    axs[0].plot(epochs, history["val_loss"], "s-", label="Validation Loss", color="#e07a5f")
    axs[0].set_title(f"{model_name}: Cross-Entropy Loss Curve")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Accuracy curve
    axs[1].plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Accuracy", color="#2b5c8f")
    axs[1].plot(epochs, [a * 100 for a in history["val_acc"]], "s-", label="Validation Accuracy", color="#81b29a")
    axs[1].set_title(f"{model_name}: Accuracy Curve (%)")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved learning curves to {out_path}")


def train(
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    use_augmentation: bool = False,
    use_class_weights: bool = True,
    model_name: str = "baseline_cnn",
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n============================================================")
    print(f"Starting Training: {model_name} on {device}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print(f"Class-weighted Loss: {use_class_weights} | Augmentation: {use_augmentation}")
    print(f"============================================================")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=batch_size,
        img_size=32,
        use_augmentation=use_augmentation,
    )

    # Initialize model
    model = BaselineCNN(num_classes=43).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Loss function with optional class weighting
    if use_class_weights:
        class_weights = calculate_class_weights(DATA_DIR / "train_split.csv", device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_checkpoint_path = CHECKPOINT_DIR / f"best_{model_name}.pth"

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "model_name": model_name,
                },
                best_checkpoint_path,
            )
            print(f" -> Checkpoint saved! Best Val Acc: {best_val_acc*100:.2f}%")

    total_time = time.time() - start_total_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")

    # Save history json
    history_file = RESULTS_DIR / f"{model_name}_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_file}")

    # Plot learning curves
    learning_curve_plot = FIG_DIR / f"{model_name}_learning_curves.png"
    plot_learning_curves(history, model_name, learning_curve_plot)

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train Traffic Sign Recognition Model")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--use_aug", action="store_true", help="Enable label-safe data augmentation")
    parser.add_argument("--no_class_weights", action="store_true", help="Disable class weighting")
    parser.add_argument("--name", type=str, default="baseline_cnn", help="Model name identifier")

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_augmentation=args.use_aug,
        use_class_weights=not args.no_class_weights,
        model_name=args.name,
    )


if __name__ == "__main__":
    main()
