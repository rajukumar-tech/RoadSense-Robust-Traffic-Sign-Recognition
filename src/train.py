"""
RoadSense: Unified Training Pipeline for Traffic Sign Recognition Models.

Supports three model types via --model_type:
  baseline_cnn   : Custom CNN from scratch (Phase 2)
  improved_cnn   : Same CNN trained with label-safe augmentations (Phase 4A)
  resnet18       : Fine-tuned ResNet-18 backbone (Phase 4B)
  mobilenet_v2   : Fine-tuned MobileNetV2 backbone (Phase 4B)

Transfer models use a two-stage strategy:
  Stage 1 (--stage1_epochs)  : Frozen backbone; only the new head is trained.
  Stage 2 (remaining epochs) : Full fine-tuning with a lower LR for the backbone.
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
from src.models import BaselineCNN, get_transfer_model, count_parameters, count_all_parameters
from src.utils import set_seed

RESULTS_DIR = ROOT_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

TRANSFER_ARCHS = {"resnet18", "mobilenet_v2"}
# Input image size: scratch CNNs use 32x32; transfer models use 48x48
IMG_SIZE_MAP = {
    "baseline_cnn": 32,
    "improved_cnn": 32,
    "resnet18": 48,
    "mobilenet_v2": 48,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def build_model(model_type: str, device: torch.device):
    """Instantiate the correct model for the given model_type string."""
    if model_type in ("baseline_cnn", "improved_cnn"):
        return BaselineCNN(num_classes=43).to(device)
    elif model_type in TRANSFER_ARCHS:
        # Start with frozen backbone for Stage 1
        return get_transfer_model(arch=model_type, num_classes=43, freeze_backbone=True).to(device)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: baseline_cnn, improved_cnn, resnet18, mobilenet_v2"
        )


def build_optimizer(model, model_type: str, lr: float, weight_decay: float):
    """
    Build optimiser with separate parameter groups for transfer models.
    Stage 1 (frozen backbone): only head params are in the optimiser.
    """
    if model_type in TRANSFER_ARCHS:
        # Only the head is trainable at this point (backbone frozen)
        head_params = model.get_head_parameters()
        return Adam(head_params, lr=lr, weight_decay=weight_decay)
    else:
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def unfreeze_and_rebuild_optimizer(model, model_type: str, lr: float, weight_decay: float):
    """
    Stage 2: unfreeze backbone and create a new optimiser with differential LRs.
    The pretrained backbone gets a 10x lower learning rate than the head to prevent
    catastrophic forgetting of ImageNet features.
    """
    model.unfreeze()
    backbone_params = model.get_backbone_parameters()
    head_params = model.get_head_parameters()
    return Adam(
        [
            {"params": backbone_params, "lr": lr / 10},
            {"params": head_params,    "lr": lr},
        ],
        weight_decay=weight_decay,
    )


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

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

    return running_loss / total, correct / total


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

    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_learning_curves(history: dict, model_name: str, out_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(epochs, history["train_loss"], "o-", label="Train Loss",      color="#2b5c8f")
    axs[0].plot(epochs, history["val_loss"],   "s-", label="Validation Loss", color="#e07a5f")
    axs[0].set_title(f"{model_name}: Cross-Entropy Loss Curve")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Mark stage boundary for transfer models
    if "stage2_start" in history and history["stage2_start"]:
        s2 = history["stage2_start"]
        for ax in axs:
            ax.axvline(s2, color="grey", linestyle="--", alpha=0.6, label=f"Unfreeze (epoch {s2})")

    axs[1].plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Accuracy", color="#2b5c8f")
    axs[1].plot(epochs, [a * 100 for a in history["val_acc"]],   "s-", label="Val Accuracy",   color="#81b29a")
    axs[1].set_title(f"{model_name}: Accuracy Curve (%)")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved learning curves to {out_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    use_augmentation: bool = False,
    use_class_weights: bool = True,
    model_type: str = "baseline_cnn",
    model_name: str = "baseline_cnn",
    stage1_epochs: int = 5,        # Transfer only: epochs with frozen backbone
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_transfer = model_type in TRANSFER_ARCHS
    img_size = IMG_SIZE_MAP.get(model_type, 32)

    print(f"\n{'='*65}")
    print(f"Starting Training: {model_name}  [{model_type}]  on {device}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | ImgSize: {img_size}x{img_size}")
    print(f"Class-weighted Loss: {use_class_weights} | Augmentation: {use_augmentation}")
    if is_transfer:
        print(f"Transfer Strategy : Stage-1 (frozen, {stage1_epochs} ep) -> Stage-2 (full fine-tune)")
    print(f"{'='*65}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Data — use ImageNet normalization for pretrained transfer models
    train_loader, val_loader, _ = get_data_loaders(
        batch_size=batch_size,
        img_size=img_size,
        use_augmentation=use_augmentation,
        use_imagenet_norm=is_transfer,
    )

    # Model
    model = build_model(model_type, device)
    trainable = count_parameters(model)
    total = count_all_parameters(model)
    print(f"Trainable params  : {trainable:,}")
    if total != trainable:
        print(f"Total params      : {total:,}  (backbone frozen for Stage 1)")

    # Loss
    if use_class_weights:
        class_weights = calculate_class_weights(DATA_DIR / "train_split.csv", device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimiser (Stage 1 for transfer models)
    optimizer = build_optimizer(model, model_type, lr, weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lr": [], "stage2_start": None,
    }

    best_val_acc = 0.0
    best_ckpt = CHECKPOINT_DIR / f"best_{model_name}.pth"
    start_total = time.time()

    for epoch in range(1, epochs + 1):

        # --- Stage 2 transition (transfer only) ---
        if is_transfer and epoch == stage1_epochs + 1:
            print(f"\n>>> Unfreezing backbone for Stage 2 fine-tuning (epoch {epoch}) <<<")
            optimizer = unfreeze_and_rebuild_optimizer(model, model_type, lr, weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
            history["stage2_start"] = epoch
            trainable2 = count_parameters(model)
            print(f"    Trainable params now: {trainable2:,}\n")

        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        stage_label = ""
        if is_transfer:
            stage_label = f"[S{'1' if epoch <= stage1_epochs else '2'}] "

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] {stage_label}({time.time()-t0:.1f}s) | "
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
                    "model_type": model_type,
                },
                best_ckpt,
            )
            print(f" -> Checkpoint saved! Best Val Acc: {best_val_acc*100:.2f}%")

    total_time = time.time() - start_total
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")

    # Save history
    history_file = RESULTS_DIR / f"{model_name}_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_file}")

    # Learning curves
    plot_learning_curves(history, model_name, FIG_DIR / f"{model_name}_learning_curves.png")

    return model, history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Traffic Sign Recognition Model (RoadSense)"
    )
    parser.add_argument(
        "--model_type", type=str, default="baseline_cnn",
        choices=["baseline_cnn", "improved_cnn", "resnet18", "mobilenet_v2"],
        help="Architecture to train. Use 'improved_cnn' for augmented scratch CNN.",
    )
    parser.add_argument("--name",         type=str,   default=None,   help="Override saved model name (default: same as model_type)")
    parser.add_argument("--epochs",       type=int,   default=15,     help="Total training epochs")
    parser.add_argument("--batch_size",   type=int,   default=64,     help="Batch size")
    parser.add_argument("--lr",           type=float, default=0.001,  help="Initial learning rate")
    parser.add_argument("--use_aug",      action="store_true",        help="Enable label-safe data augmentation")
    parser.add_argument("--no_class_weights", action="store_true",    help="Disable class-weighted loss")
    parser.add_argument("--stage1_epochs",type=int,   default=5,      help="(Transfer only) Epochs with frozen backbone")

    args = parser.parse_args()

    # Default model_name == model_type unless overridden
    model_name = args.name if args.name else args.model_type

    # For 'improved_cnn', the underlying architecture is still BaselineCNN
    # but augmentation is forced on
    use_aug = args.use_aug
    if args.model_type == "improved_cnn":
        use_aug = True

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_augmentation=use_aug,
        use_class_weights=not args.no_class_weights,
        model_type=args.model_type,
        model_name=model_name,
        stage1_epochs=args.stage1_epochs,
    )


if __name__ == "__main__":
    main()
