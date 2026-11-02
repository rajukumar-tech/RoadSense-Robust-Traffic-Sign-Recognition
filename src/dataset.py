"""
RoadSense: Dataset loading and transform pipeline for GTSRB.
Handles image loading from split CSVs, standard resizing (e.g. 32x32), normalization,
and label-safe data augmentations.
"""

from pathlib import Path
from typing import Optional, Callable, Union
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# GTSRB-specific normalization statistics (for scratch CNNs)
NORM_MEAN = [0.3403, 0.3121, 0.3214]  # Measured GTSRB channel means
NORM_STD = [0.2724, 0.2608, 0.2669]   # Measured GTSRB channel stds

# ImageNet normalization statistics (for pretrained transfer models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(
    img_size: int = 32,
    is_training: bool = False,
    use_augmentation: bool = False,
    use_imagenet_norm: bool = False,
) -> T.Compose:
    """
    Returns torchvision transforms for GTSRB images.
    If use_augmentation is True, applies strictly label-safe transformations
    (small rotation, brightness/contrast jitter, slight affine translation).
    If use_imagenet_norm is True, uses ImageNet mean/std (for pretrained transfer models).
    """
    transforms_list = [
        T.Resize((img_size, img_size)),
    ]

    if is_training and use_augmentation:
        # STRICTLY LABEL-SAFE AUGMENTATIONS:
        # - NO horizontal flips (would convert Keep Left to Keep Right)
        # - NO vertical flips
        transforms_list.extend([
            T.RandomRotation(degrees=10),
            T.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

    # Use ImageNet stats for pretrained backbones, GTSRB stats for scratch CNNs
    mean = IMAGENET_MEAN if use_imagenet_norm else NORM_MEAN
    std = IMAGENET_STD if use_imagenet_norm else NORM_STD

    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return T.Compose(transforms_list)


class GTSRBDataset(Dataset):
    """
    PyTorch Dataset for German Traffic Sign Recognition Benchmark (GTSRB).
    Reads image paths and class labels from a prepared CSV file.
    """

    def __init__(
        self,
        csv_file: Union[Path, str],
        data_root: Optional[Union[Path, str]] = None,
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

        # Load image as RGB
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
    use_imagenet_norm: bool = False,
):
    """
    Helper to construct PyTorch DataLoaders for train, val, and test splits.
    Set use_imagenet_norm=True for pretrained transfer models.
    """
    train_csv = DATA_DIR / "train_split.csv"
    val_csv = DATA_DIR / "val_split.csv"
    test_csv = DATA_DIR / "Test.csv"

    for p in [train_csv, val_csv, test_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required CSV: {p}. Run src/eda_and_split.py first.")

    train_transform = get_transforms(
        img_size=img_size, is_training=True,
        use_augmentation=use_augmentation, use_imagenet_norm=use_imagenet_norm,
    )
    eval_transform = get_transforms(
        img_size=img_size, is_training=False,
        use_augmentation=False, use_imagenet_norm=use_imagenet_norm,
    )

    train_dataset = GTSRBDataset(csv_file=train_csv, transform=train_transform)
    val_dataset = GTSRBDataset(csv_file=val_csv, transform=eval_transform)
    test_dataset = GTSRBDataset(csv_file=test_csv, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
