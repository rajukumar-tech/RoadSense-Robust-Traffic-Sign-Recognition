"""
RoadSense: Neural Network Architectures for GTSRB.

Contains:
  - BaselineCNN       : 6-layer custom CNN trained from scratch (Phase 2)
  - TransferModel     : ResNet-18 / MobileNetV2 fine-tuned for 43 GTSRB classes (Phase 4B)
  - get_transfer_model: factory that builds either transfer architecture by name
  - count_parameters  : utility to count trainable params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# Phase 2 — Baseline CNN (from scratch)
# ---------------------------------------------------------------------------

class BaselineCNN(nn.Module):
    """
    Custom 6-layer Convolutional Neural Network designed from scratch for 32x32 GTSRB images.

    Architecture:
      ConvBlock 1: Conv(3->32, 3x3) -> BN -> ReLU -> Conv(32->32, 3x3) -> BN -> ReLU -> MaxPool(2x2) -> Dropout(0.2)
      ConvBlock 2: Conv(32->64, 3x3) -> BN -> ReLU -> Conv(64->64, 3x3) -> BN -> ReLU -> MaxPool(2x2) -> Dropout(0.3)
      ConvBlock 3: Conv(64->128, 3x3) -> BN -> ReLU -> Conv(128->128, 3x3) -> BN -> ReLU -> MaxPool(2x2) -> Dropout(0.4)
      Classifier:  Flatten(2048) -> Linear(2048->256) -> BN -> ReLU -> Dropout(0.5) -> Linear(256->43)
    """

    def __init__(self, num_classes: int = 43):
        super(BaselineCNN, self).__init__()

        # Block 1: 32x32 -> 16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),
        )

        # Block 2: 16x16 -> 8x8
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3),
        )

        # Block 3: 8x8 -> 4x4
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.4),
        )

        # Fully Connected Classifier: 128 * 4 * 4 = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        logits = self.classifier(x)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map before pooling/classifier (useful for Grad-CAM)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


# ---------------------------------------------------------------------------
# Phase 4B — Transfer Learning Wrappers
# ---------------------------------------------------------------------------

class TransferModel(nn.Module):
    """
    Fine-tunable wrapper around ImageNet-pretrained backbones for 43-class GTSRB.

    Supports:
      - 'resnet18'      : ResNet-18 (~11.7M total params, ~500K trainable when frozen)
      - 'mobilenet_v2'  : MobileNetV2 (~3.4M total params, ~330K trainable when frozen)

    Strategy (two-stage fine-tuning):
      Stage 1 — Frozen backbone: Only the new classification head is trained for a few
                epochs until convergence. This warms up the head without corrupting the
                pretrained features with a large gradient signal.
      Stage 2 — Full fine-tune: Unfreeze all layers and train end-to-end with a lower
                learning rate (e.g. 1e-4) so the pretrained backbone adapts gently to the
                GTSRB domain (small 48x48 signs vs large ImageNet objects).

    Args:
        arch         : One of 'resnet18' or 'mobilenet_v2'
        num_classes  : Number of output classes (43 for GTSRB)
        freeze_backbone : If True, only the classification head is trainable initially.
                          Call model.unfreeze() to switch to full fine-tuning.
        dropout_p    : Dropout probability applied inside the new head.
    """

    def __init__(
        self,
        arch: str = "resnet18",
        num_classes: int = 43,
        freeze_backbone: bool = True,
        dropout_p: float = 0.4,
    ):
        super(TransferModel, self).__init__()
        self.arch = arch
        self.num_classes = num_classes

        if arch == "resnet18":
            backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
            in_features = backbone.fc.in_features  # 512

            # Replace the original 1000-class head with a GTSRB-tuned head
            backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p * 0.5),
                nn.Linear(256, num_classes),
            )

            self.backbone = backbone
            # Reference to the new head for parameter group separation
            self._head_params = list(backbone.fc.parameters())

        elif arch == "mobilenet_v2":
            backbone = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1)
            in_features = backbone.classifier[1].in_features  # 1280

            # Replace MobileNetV2's classifier
            backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p * 0.5),
                nn.Linear(256, num_classes),
            )

            self.backbone = backbone
            self._head_params = list(backbone.classifier.parameters())

        else:
            raise ValueError(
                f"Unknown architecture '{arch}'. Choose from: 'resnet18', 'mobilenet_v2'"
            )

        if freeze_backbone:
            self.freeze()

    # ------------------------------------------------------------------
    # Freeze / Unfreeze helpers for two-stage fine-tuning
    # ------------------------------------------------------------------

    def freeze(self):
        """Freeze all backbone parameters; only the head trains."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Always keep the new head unfrozen
        if self.arch == "resnet18":
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif self.arch == "mobilenet_v2":
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True

    def unfreeze(self):
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_head_parameters(self):
        """Return only head parameters (for Stage 1 optimiser)."""
        if self.arch == "resnet18":
            return list(self.backbone.fc.parameters())
        return list(self.backbone.classifier.parameters())

    def get_backbone_parameters(self):
        """Return only backbone parameters (for separate LR group in Stage 2)."""
        if self.arch == "resnet18":
            head_ids = {id(p) for p in self.backbone.fc.parameters()}
        else:
            head_ids = {id(p) for p in self.backbone.classifier.parameters()}
        return [p for p in self.backbone.parameters() if id(p) not in head_ids]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def get_transfer_model(
    arch: str = "resnet18",
    num_classes: int = 43,
    freeze_backbone: bool = True,
    dropout_p: float = 0.4,
) -> TransferModel:
    """
    Build and return a TransferModel for the given architecture.

    Args:
        arch            : 'resnet18' or 'mobilenet_v2'
        num_classes     : 43 for GTSRB
        freeze_backbone : Whether to freeze the pretrained backbone initially
        dropout_p       : Dropout probability in the new head

    Returns:
        TransferModel instance ready for .to(device)
    """
    return TransferModel(
        arch=arch,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        dropout_p=dropout_p,
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """Return total number of parameters (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # BaselineCNN
    model = BaselineCNN(num_classes=43)
    dummy = torch.randn(4, 3, 32, 32)
    out = model(dummy)
    print("=== BaselineCNN ===")
    print(f"  Output shape      : {out.shape}")
    print(f"  Trainable params  : {count_parameters(model):,}")

    # ResNet-18
    print("\n=== TransferModel: resnet18 (frozen backbone) ===")
    r18 = get_transfer_model("resnet18", freeze_backbone=True)
    dummy48 = torch.randn(4, 3, 48, 48)
    out_r18 = r18(dummy48)
    print(f"  Output shape      : {out_r18.shape}")
    print(f"  Trainable params  : {count_parameters(r18):,}")
    print(f"  Total params      : {count_all_parameters(r18):,}")

    # MobileNetV2
    print("\n=== TransferModel: mobilenet_v2 (frozen backbone) ===")
    mv2 = get_transfer_model("mobilenet_v2", freeze_backbone=True)
    out_mv2 = mv2(dummy48)
    print(f"  Output shape      : {out_mv2.shape}")
    print(f"  Trainable params  : {count_parameters(mv2):,}")
    print(f"  Total params      : {count_all_parameters(mv2):,}")
