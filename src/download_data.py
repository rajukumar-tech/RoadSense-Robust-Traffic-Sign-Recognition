"""
RoadSense: GTSRB Dataset Downloader & Preprocessor
Downloads the GTSRB dataset (Kaggle/ERDA format) from the verified Hugging Face mirror
and extracts images into standard folders:
  data/
  |-- Train.csv
  |-- Test.csv
  |-- Meta.csv
  |-- Train/<ClassId>/<image>.png
  |-- Test/<image>.png
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BASE_URL = "https://huggingface.co/datasets/bazyl/GTSRB/resolve/main"

FILES = {
    "Train.csv": f"{BASE_URL}/Train.csv",
    "Test.csv": f"{BASE_URL}/Test.csv",
    "Meta.csv": f"{BASE_URL}/Meta.csv",
    "train_data.parquet": f"{BASE_URL}/data/train-00000-of-00001-dc762c064c221993.parquet",
    "test_data.parquet": f"{BASE_URL}/data/test-00000-of-00001-747a54d4a6461a97.parquet",
}


def download_file_stream(url: str, dest_path: Path):
    if dest_path.exists():
        print(f"File already exists: {dest_path.name} ({dest_path.stat().st_size / 1e6:.1f} MB). Skipping download.")
        return

    print(f"Downloading {dest_path.name} from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1MB chunks

    with open(dest_path, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=dest_path.name,
        ncols=80,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"Successfully saved {dest_path.name}")


def extract_images_from_parquet(parquet_path: Path, csv_path: Path, base_dir: Path):
    print(f"\nExtracting images from {parquet_path.name} using {csv_path.name}...")
    df_parquet = pd.read_parquet(parquet_path)
    df_csv = pd.read_csv(csv_path)

    if len(df_parquet) != len(df_csv):
        raise ValueError(f"Mismatch in count: {len(df_parquet)} vs {len(df_csv)}")

    print(f"Processing {len(df_parquet)} images...")

    # Extract images and save to corresponding path
    for idx in tqdm(range(len(df_parquet)), desc=f"Writing {parquet_path.stem}", ncols=80):
        row_csv = df_csv.iloc[idx]
        rel_path = row_csv["Path"]  # e.g. Train/20/00020_00000_00000.png or Test/00000.png
        out_path = base_dir / rel_path

        if out_path.exists():
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img_dict = df_parquet.iloc[idx]["Path"]
        img_bytes = img_dict["bytes"]

        with open(out_path, "wb") as f:
            f.write(img_bytes)

    print(f"Extraction complete for {parquet_path.name}!")


def verify_dataset(base_dir: Path):
    print("\n" + "=" * 60)
    print("GTSRB DATASET VERIFICATION SUMMARY")
    print("=" * 60)

    train_csv = base_dir / "Train.csv"
    test_csv = base_dir / "Test.csv"

    if not train_csv.exists() or not test_csv.exists():
        print("ERROR: Metadata CSVs missing!")
        return

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    print(f"Train records in CSV: {len(df_train)} (across {df_train['ClassId'].nunique()} classes)")
    print(f"Test records in CSV:  {len(df_test)} (across {df_test['ClassId'].nunique()} classes)")

    train_images = list((base_dir / "Train").rglob("*.png"))
    test_images = list((base_dir / "Test").rglob("*.png"))

    print(f"Train images on disk: {len(train_images)}")
    print(f"Test images on disk:  {len(test_images)}")

    # Check track ID structure in training data
    df_train["track_id"] = df_train["Path"].apply(lambda p: Path(p).stem.split("_")[1])
    tracks_per_class = df_train.groupby("ClassId")["track_id"].nunique()
    print(f"\nUnique tracks identified: {df_train['track_id'].nunique()} across all classes.")
    print(f"Average tracks per class: {tracks_per_class.mean():.1f} (min: {tracks_per_class.min()}, max: {tracks_per_class.max()})")
    print("=" * 60)
    print("READY FOR PHASE 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download metadata and parquet archives
    for filename, url in FILES.items():
        dest = DATA_DIR / filename
        download_file_stream(url, dest)

    # 2. Extract train images
    extract_images_from_parquet(
        parquet_path=DATA_DIR / "train_data.parquet",
        csv_path=DATA_DIR / "Train.csv",
        base_dir=DATA_DIR,
    )

    # 3. Extract test images
    extract_images_from_parquet(
        parquet_path=DATA_DIR / "test_data.parquet",
        csv_path=DATA_DIR / "Test.csv",
        base_dir=DATA_DIR,
    )

    # 4. Verify all files
    verify_dataset(DATA_DIR)


if __name__ == "__main__":
    main()
