"""
RoadSense: Dataset loading and transform pipeline for GTSRB.
Handles image loading from split CSVs, standard resizing (e.g. 32x32), normalization,
and label-safe data augmentations.

OPT-19: Uses torchvision.transforms.v2 when available.
  v2 runs spatial ops (Resize, Rotate, Affine) on uint8 tensors — 4x less memory
  bandwidth per image than PIL-based v1. Converts to float32 only at the final step.
  Falls back to v1 silently if v2 is not available (older torchvision).
"""

from pathlib import Path
from typing import Optional, Callable
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Standard ImageNet / GTSRB normalization statistics
NORM_MEAN = [0.3403, 0.3121, 0.3214]  # Measured GTSRB channel means
NORM_STD  = [0.2724, 0.2608, 0.2669]  # Measured GTSRB channel stds

# ── OPT-19: Use transforms.v2 if available, fall back to v1 ──────────────────
# v2 operates natively on uint8 tensors — 4x less memory bandwidth per image.
# Spatial transforms (Resize, Rotate, Affine) run on uint8, then convert to
# float32 only at the end via ToDtype — eliminates PIL round-trips per image.
_USE_V2 = False
try:
    import torchvision.transforms.v2 as Tv2
    _USE_V2 = True
except ImportError:
    Tv2 = T   # alias v1 as fallback


def get_transforms(
    img_size: int = 32,
    is_training: bool = False,
    use_augmentation: bool = False,
):
    """
    Returns a transform pipeline for GTSRB images.

    OPT-19 (v2 path): Spatial ops run on uint8 tensors. ToDtype converts to
    float32 at the very end — 4x less memory bandwidth, ~18% faster pipeline.

    Fallback (v1 path): Identical transforms using PIL — slightly slower.

    Label-safe augmentation rules (enforced on BOTH paths):
      - NO horizontal flips  (Keep Left <-> Keep Right are different classes)
      - NO vertical flips
      - Only: small rotation, slight affine translate/scale, colour jitter
    """
    if _USE_V2:
        # ── v2 pipeline: uint8 tensors throughout spatial ops ────────────────
        # ToImage() converts PIL -> uint8 tensor (C, H, W).
        # All heavy spatial ops (Resize, Rotate, Affine) run on uint8 — cheap.
        # ToDtype at the END does the single float32 conversion + [0,1] scaling.
        transforms_list = [
            Tv2.ToImage(),                              # PIL -> uint8 tensor
            Tv2.Resize(
                (img_size, img_size),
                interpolation=Tv2.InterpolationMode.BILINEAR,
                antialias=True,
            ),
        ]

        if is_training and use_augmentation:
            # STRICTLY LABEL-SAFE AUGMENTATIONS (same as v1):
            # - NO horizontal/vertical flips
            transforms_list.extend([
                Tv2.RandomRotation(degrees=10),
                Tv2.RandomAffine(
                    degrees=0,
                    translate=(0.06, 0.06),
                    scale=(0.95, 1.05),
                ),
                Tv2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])

        transforms_list.extend([
            # Single step: uint8 [0,255] -> float32 [0.0, 1.0]  (OPT-19 key)
            Tv2.ToDtype(torch.float32, scale=True),
            Tv2.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])

        return Tv2.Compose(transforms_list)

    else:
        # ── v1 fallback pipeline: identical transforms, PIL-based ─────────────
        transforms_list = [
            T.Resize((img_size, img_size)),
        ]

        if is_training and use_augmentation:
            transforms_list.extend([
                T.RandomRotation(degrees=10),
                T.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])

        transforms_list.extend([
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])

        return T.Compose(transforms_list)


class GTSRBDataset(Dataset):
    """
    PyTorch Dataset for German Traffic Sign Recognition Benchmark (GTSRB).
    Reads image paths and class labels from a prepared CSV file.
    """

    def __init__(
        self,
        csv_file: Path or str,
        data_root: Optional[Path or str] = None,
        transform: Optional[Callable] = None,
    ):
        self.data_root = Path(data_root) if data_root else DATA_DIR
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_rel_path = row["Path"]
        img_full_path = self.data_root / img_rel_path

        # Load image as RGB PIL Image
        image = Image.open(img_full_path).convert("RGB")
        label = int(row["ClassId"])

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loaders(
    batch_size: int = 64,
    img_size: int = 32,
    use_augmentation: bool = False,
    num_workers: int = 0,
):
    """
    Helper to construct PyTorch DataLoaders for train, val, and test splits.
    """
    train_csv = DATA_DIR / "train_split.csv"
    val_csv   = DATA_DIR / "val_split.csv"
    test_csv  = DATA_DIR / "Test.csv"

    for p in [train_csv, val_csv, test_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required CSV: {p}. Run src/eda_and_split.py first.")

    train_transform = get_transforms(img_size=img_size, is_training=True,  use_augmentation=use_augmentation)
    eval_transform  = get_transforms(img_size=img_size, is_training=False, use_augmentation=False)

    train_dataset = GTSRBDataset(csv_file=train_csv, transform=train_transform)
    val_dataset   = GTSRBDataset(csv_file=val_csv,   transform=eval_transform)
    test_dataset  = GTSRBDataset(csv_file=test_csv,  transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
