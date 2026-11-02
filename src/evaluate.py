"""
RoadSense: Comprehensive Evaluation Pipeline.
Evaluates models on the clean test set, computes Accuracy, Macro-F1,
generates the 43x43 Confusion Matrix, and identifies the most confused sign pairs.
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.dataset import get_data_loaders, DATA_DIR
from src.models import BaselineCNN
from src.utils import CLASS_NAMES

RESULTS_DIR = ROOT_DIR / "results"
CM_DIR = RESULTS_DIR / "confusion_matrices"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"


def evaluate_model(model, test_loader, device):
    """Run model on test loader and return predictions, targets, and probabilities."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False, ncols=80):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_targets), np.array(all_probs)


def plot_and_save_confusion_matrix(y_true, y_pred, model_name: str, out_path: Path):
    """Generate high-resolution 43x43 confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    # Normalize by true class row for clearer visualization
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(18, 16))
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Blues",
        xticklabels=[str(i) for i in range(43)],
        yticklabels=[str(i) for i in range(43)],
        cbar_kws={"label": "Normalized Recall"},
    )
    plt.title(f"Confusion Matrix: {model_name} on Clean Test Set (43 Classes)", fontsize=14, pad=12)
    plt.xlabel("Predicted Class ID", fontsize=12)
    plt.ylabel("True Class ID", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {out_path}")
    return cm


def identify_most_confused_pairs(cm, top_k: int = 5):
    """
    Find the sign pairs with the highest error counts.
    Ignores the diagonal (correct predictions).
    """
    confused_pairs = []
    num_classes = cm.shape[0]

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "True_Class_ID": i,
                    "True_Sign_Name": CLASS_NAMES.get(i, f"Class {i}"),
                    "Pred_Class_ID": j,
                    "Pred_Sign_Name": CLASS_NAMES.get(j, f"Class {j}"),
                    "Error_Count": int(cm[i, j]),
                })

    df_confused = pd.DataFrame(confused_pairs)
    if not df_confused.empty:
        df_confused = df_confused.sort_values(by="Error_Count", ascending=False).reset_index(drop=True)
    return df_confused.head(top_k)


def append_metrics_csv(model_name: str, metrics: dict, csv_path: Path):
    """Record evaluation results into results/metrics.csv."""
    row = {
        "Model": model_name,
        "Clean_Accuracy_%": round(metrics["accuracy"] * 100, 2),
        "Macro_F1_%": round(metrics["macro_f1"] * 100, 2),
        "Weighted_F1_%": round(metrics["weighted_f1"] * 100, 2),
        "Macro_Precision_%": round(metrics["macro_precision"] * 100, 2),
        "Macro_Recall_%": round(metrics["macro_recall"] * 100, 2),
    }

    df_row = pd.DataFrame([row])
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        # Update or append
        df_existing = df_existing[df_existing["Model"] != model_name]
        df_combined = pd.concat([df_existing, df_row], ignore_index=True)
    else:
        df_combined = df_row

    df_combined.to_csv(csv_path, index=False)
    print(f"Recorded clean metrics to {csv_path}")


def evaluate(model_name: str = "baseline_cnn", checkpoint_file: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 65)
    print(f"EVALUATING MODEL: {model_name} ON CLEAN TEST SET")
    print("=" * 65)

    CM_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load test data
    _, _, test_loader = get_data_loaders(batch_size=64, img_size=32, use_augmentation=False)

    # 2. Instantiate and load checkpoint
    model = BaselineCNN(num_classes=43).to(device)

    ckpt_path = Path(checkpoint_file) if checkpoint_file else CHECKPOINT_DIR / f"best_{model_name}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded weights from {ckpt_path} (Trained for {checkpoint.get('epoch', '?')} epochs)")

    # 3. Perform inference
    y_pred, y_true, y_prob = evaluate_model(model, test_loader, device)

    # 4. Compute metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
    }

    print("\n--- Clean Test Set Performance ---")
    print(f" Overall Accuracy: {acc*100:.2f}%")
    print(f" Macro-F1 Score:   {macro_f1*100:.2f}%")
    print(f" Weighted-F1:      {weighted_f1*100:.2f}%")
    print(f" Macro-Precision:  {macro_prec*100:.2f}%")
    print(f" Macro-Recall:     {macro_rec*100:.2f}%")

    # 5. Confusion Matrix & Most Confused Pairs
    cm_path = CM_DIR / f"{model_name}_cm.png"
    cm = plot_and_save_confusion_matrix(y_true, y_pred, model_name, cm_path)

    top_confused = identify_most_confused_pairs(cm, top_k=5)
    print("\n--- Top 5 Most Confused Traffic Sign Pairs ---")
    for idx, row in top_confused.iterrows():
        print(
            f"  {idx+1}. True: [{row['True_Class_ID']}] '{row['True_Sign_Name']}' "
            f"--> Predicted: [{row['Pred_Class_ID']}] '{row['Pred_Sign_Name']}' "
            f"(Count: {row['Error_Count']})"
        )

    # 6. Save metrics
    metrics_csv = RESULTS_DIR / "metrics.csv"
    append_metrics_csv(model_name, metrics, metrics_csv)

    # Save detailed classification report
    report_dict = classification_report(y_true, y_pred, target_names=[CLASS_NAMES[i] for i in range(43)], output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose()
    report_csv = RESULTS_DIR / f"{model_name}_classification_report.csv"
    df_report.to_csv(report_csv)
    print(f"Saved full classification report to {report_csv}")

    print("=" * 65)
    print("EVALUATION COMPLETE")
    print("=" * 65)

    return metrics, top_confused


def main():
    parser = argparse.ArgumentParser(description="Evaluate Model on GTSRB Test Set")
    parser.add_argument("--name", type=str, default="baseline_cnn", help="Model name identifier")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth file")

    args = parser.parse_args()
    evaluate(model_name=args.name, checkpoint_file=args.checkpoint)


if __name__ == "__main__":
    main()
