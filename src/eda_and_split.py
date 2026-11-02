"""
RoadSense: Phase 1 - Exploratory Data Analysis & Leakage-Free Validation Split
Performs full EDA on GTSRB and creates an atomic track-based train/validation split.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Import shared utilities
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils import CLASS_NAMES, CLASS_CATEGORIES, set_seed

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_FIG_DIR = ROOT_DIR / "results" / "figures"


def setup_plotting_style():
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 14,
    })


def load_datasets():
    train_csv = DATA_DIR / "Train.csv"
    test_csv = DATA_DIR / "Test.csv"
    meta_csv = DATA_DIR / "Meta.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Missing CSVs in {DATA_DIR}. Please run src/download_data.py first.")

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    df_meta = pd.read_csv(meta_csv) if meta_csv.exists() else None

    # Parse track ID and frame ID from path: Train/<ClassId>/<ClassId>_<TrackId>_<FrameId>.png
    df_train["filename"] = df_train["Path"].apply(lambda p: Path(p).name)
    df_train["track_id"] = df_train["filename"].apply(lambda fn: fn.split("_")[1] if len(fn.split("_")) > 1 else "0")
    df_train["frame_id"] = df_train["filename"].apply(lambda fn: fn.split("_")[2].split(".")[0] if len(fn.split("_")) > 2 else "0")
    df_train["class_name"] = df_train["ClassId"].map(CLASS_NAMES)

    df_test["filename"] = df_test["Path"].apply(lambda p: Path(p).name)
    df_test["class_name"] = df_test["ClassId"].map(CLASS_NAMES)

    return df_train, df_test, df_meta


def analyze_and_plot_class_distribution(df_train, df_test):
    print("\n--- 1. Class Distribution Analysis ---")
    train_counts = df_train["ClassId"].value_counts().sort_index()
    test_counts = df_test["ClassId"].value_counts().sort_index()

    dist_df = pd.DataFrame({
        "ClassId": train_counts.index,
        "Sign_Name": [CLASS_NAMES.get(i, f"Class {i}") for i in train_counts.index],
        "Train_Count": train_counts.values,
        "Test_Count": test_counts.values,
        "Train_Share_%": np.round((train_counts.values / len(df_train)) * 100, 2),
    })

    print(f"Total Training Samples: {len(df_train)}")
    print(f"Max class count: {train_counts.max()} ({CLASS_NAMES[train_counts.idxmax()]})")
    print(f"Min class count: {train_counts.min()} ({CLASS_NAMES[train_counts.idxmin()]})")
    print(f"Class Imbalance Ratio (Max / Min): {train_counts.max() / train_counts.min():.2f}x")

    # Plot Distribution
    plt.figure(figsize=(16, 7))
    x = np.arange(43)
    width = 0.4

    plt.bar(x - width/2, train_counts.values, width, label="Train Set (39,209)", color="#2b5c8f", alpha=0.9)
    plt.bar(x + width/2, test_counts.values, width, label="Test Set (12,630)", color="#e07a5f", alpha=0.9)

    plt.xlabel("Traffic Sign Class ID (0 to 42)")
    plt.ylabel("Number of Images")
    plt.title("GTSRB Class Frequency Distribution: Severe Class Imbalance (Up to 10.5x Ratio)")
    plt.xticks(x, [f"{i}" for i in range(43)], rotation=0)
    plt.legend()
    plt.tight_layout()

    out_file = RESULTS_FIG_DIR / "class_distribution.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved class distribution plot to {out_file}")

    return dist_df


def plot_sample_grid_43_classes(df_train):
    print("\n--- 2. Generating 43-Class Sample Grid ---")
    fig, axes = plt.subplots(7, 7, figsize=(18, 18))
    axes = axes.flatten()

    for class_id in range(43):
        ax = axes[class_id]
        # Pick a clear representative sample (e.g. median size frame)
        class_samples = df_train[df_train["ClassId"] == class_id]
        sample_row = class_samples.iloc[len(class_samples) // 2]
        img_path = DATA_DIR / sample_row["Path"]

        img = Image.open(img_path)
        ax.imshow(img)
        title = f"[{class_id}] {CLASS_NAMES.get(class_id, '')}"
        if len(title) > 23:
            title = title[:21] + "..."
        ax.set_title(title, fontsize=8, pad=4)
        ax.axis("off")

    # Hide extra subplots (since 7x7 = 49 subplots for 43 classes)
    for idx in range(43, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("RoadSense: Representative Samples of All 43 GTSRB Traffic Sign Classes", fontsize=15, y=0.99)
    plt.tight_layout()

    out_file = RESULTS_FIG_DIR / "sample_grid_43_classes.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved 43-class sample grid to {out_file}")


def analyze_image_dimensions_and_brightness(df_train):
    print("\n--- 3. Analyzing Image Dimensions & Brightness ---")
    widths = df_train["Width"].values
    heights = df_train["Height"].values
    aspect_ratios = widths / heights

    print(f"Image Widths:  min={widths.min()}px, median={np.median(widths)}px, mean={widths.mean():.1f}px, max={widths.max()}px")
    print(f"Image Heights: min={heights.min()}px, median={np.median(heights)}px, mean={heights.mean():.1f}px, max={heights.max()}px")
    print(f"Aspect Ratios: mean={aspect_ratios.mean():.3f}, std={aspect_ratios.std():.3f}")

    # Plot size distributions
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(widths, bins=35, kde=True, ax=axs[0], color="#2b5c8f", label="Width")
    sns.histplot(heights, bins=35, kde=True, ax=axs[0], color="#81b29a", label="Height", alpha=0.6)
    axs[0].set_title("GTSRB Image Dimension Distribution (High Resolution Variance)")
    axs[0].set_xlabel("Pixels")
    axs[0].set_ylabel("Frequency")
    axs[0].legend()

    axs[1].scatter(widths, heights, alpha=0.15, color="#3d405b", s=10)
    axs[1].plot([15, 250], [15, 250], "--r", label="Square (1:1 Aspect)")
    axs[1].set_title("Width vs Height Scatter (Predominantly Square / Circular Signs)")
    axs[1].set_xlabel("Width (px)")
    axs[1].set_ylabel("Height (px)")
    axs[1].legend()

    plt.tight_layout()
    out_size_file = RESULTS_FIG_DIR / "image_size_distribution.png"
    plt.savefig(out_size_file, dpi=300)
    plt.close()
    print(f"Saved image size distribution plot to {out_size_file}")

    # Brightness and contrast analysis on sample of dataset for speed
    print("Computing brightness and contrast on 5,000 representative images...")
    sample_rows = df_train.sample(n=min(5000, len(df_train)), random_state=42)
    brightness_vals = []
    contrast_vals = []

    for _, row in sample_rows.iterrows():
        img_path = DATA_DIR / row["Path"]
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        # Perceived luminance: 0.299 R + 0.587 G + 0.114 B
        lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        brightness_vals.append(lum.mean())
        contrast_vals.append(lum.std())

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(brightness_vals, bins=40, kde=True, ax=axs[0], color="#e76f51")
    axs[0].set_title("Image Brightness (Luminance) Distribution")
    axs[0].set_xlabel("Mean Luminance (0=Black, 255=White)")
    axs[0].set_ylabel("Count")

    sns.histplot(contrast_vals, bins=40, kde=True, ax=axs[1], color="#f4a261")
    axs[1].set_title("Image Contrast (Luminance Std Dev) Distribution")
    axs[1].set_xlabel("Standard Deviation of Luminance")
    axs[1].set_ylabel("Count")

    plt.tight_layout()
    out_bright_file = RESULTS_FIG_DIR / "brightness_distribution.png"
    plt.savefig(out_bright_file, dpi=300)
    plt.close()
    print(f"Saved brightness distribution plot to {out_bright_file}")


def create_leakage_free_validation_split(df_train, val_ratio: float = 0.20, seed: int = 42):
    """
    Splits the training data by TRACK to avoid data leakage.
    Each unique track consists of ~30 consecutive video frames of the exact same physical sign.
    Partitioning tracks (rather than individual images) guarantees train and val sets are disjoint in real-world signs.
    """
    print("\n--- 4. Creating Leakage-Free Track-Based Validation Split ---")
    set_seed(seed)

    train_splits = []
    val_splits = []

    # Group by class to ensure stratified class representation
    for class_id, class_group in df_train.groupby("ClassId"):
        unique_tracks = class_group["track_id"].unique()
        n_tracks = len(unique_tracks)

        # Shuffle tracks
        shuffled_tracks = np.random.permutation(unique_tracks)

        # Decide validation track count: at least 1 track, up to ~20%
        n_val = max(1, int(np.round(n_tracks * val_ratio)))
        val_tracks = set(shuffled_tracks[:n_val])
        train_tracks = set(shuffled_tracks[n_val:])

        class_train = class_group[class_group["track_id"].isin(train_tracks)]
        class_val = class_group[class_group["track_id"].isin(val_tracks)]

        train_splits.append(class_train)
        val_splits.append(class_val)

    train_df = pd.concat(train_splits).reset_index(drop=True)
    val_df = pd.concat(val_splits).reset_index(drop=True)

    print(f"Split Result:")
    print(f"  Training Set:   {len(train_df):,} images ({len(train_df)/len(df_train)*100:.1f}%)")
    print(f"  Validation Set: {len(val_df):,} images ({len(val_df)/len(df_train)*100:.1f}%)")
    print(f"  Total:          {len(train_df) + len(val_df):,} images")

    # Double check strict track disjointness
    for class_id in range(43):
        tr_t = set(train_df[train_df["ClassId"] == class_id]["track_id"].unique())
        va_t = set(val_df[val_df["ClassId"] == class_id]["track_id"].unique())
        overlap = tr_t.intersection(va_t)
        assert len(overlap) == 0, f"LEAKAGE DETECTED in class {class_id}! Overlapping tracks: {overlap}"

    print(" Verified: Exactly 0 overlapping tracks between Train and Validation sets (Zero Leakage).")

    # Save to data directory
    train_split_path = DATA_DIR / "train_split.csv"
    val_split_path = DATA_DIR / "val_split.csv"

    train_df.to_csv(train_split_path, index=False)
    val_df.to_csv(val_split_path, index=False)
    print(f"Saved train split CSV: {train_split_path}")
    print(f"Saved val split CSV:   {val_split_path}")

    # Plot train vs val counts
    plt.figure(figsize=(15, 6))
    tr_counts = train_df["ClassId"].value_counts().sort_index()
    va_counts = val_df["ClassId"].value_counts().sort_index()

    x = np.arange(43)
    width = 0.45
    plt.bar(x, tr_counts.values, width, label=f"Train Tracks ({len(train_df):,} imgs)", color="#2b5c8f")
    plt.bar(x, va_counts.values, width, bottom=tr_counts.values, label=f"Val Tracks ({len(val_df):,} imgs)", color="#81b29a")

    plt.xlabel("Class ID (0 to 42)")
    plt.ylabel("Number of Images")
    plt.title("Stratified Track-Based Disjoint Train / Validation Split (Leak-Free)")
    plt.xticks(x, [str(i) for i in range(43)], rotation=0)
    plt.legend()
    plt.tight_layout()

    out_split_plot = RESULTS_FIG_DIR / "track_split_distribution.png"
    plt.savefig(out_split_plot, dpi=300)
    plt.close()
    print(f"Saved track split plot to {out_split_plot}")

    return train_df, val_df


def main():
    setup_plotting_style()
    RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ROADSENSE - PHASE 1: EXPLORATORY DATA ANALYSIS & PREPROCESSING")
    print("=" * 65)

    df_train, df_test, df_meta = load_datasets()

    # 1. Class Distribution
    dist_df = analyze_and_plot_class_distribution(df_train, df_test)

    # 2. 43-Class Sample Grid
    plot_sample_grid_43_classes(df_train)

    # 3. Size & Brightness Distributions
    analyze_image_dimensions_and_brightness(df_train)

    # 4. Leakage-Free Validation Split
    train_df, val_df = create_leakage_free_validation_split(df_train, val_ratio=0.20, seed=42)

    print("\n" + "=" * 65)
    print("PHASE 1 COMPLETE: All figures generated and split saved.")
    print("=" * 65)


if __name__ == "__main__":
    main()
