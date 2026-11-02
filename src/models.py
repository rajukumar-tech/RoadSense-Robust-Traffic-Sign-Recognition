"""
RoadSense: Neural Network Architectures for GTSRB.
Contains custom Baseline CNN from scratch and transfer learning wrappers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = BaselineCNN(num_classes=43)
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)
    print("BaselineCNN initialized successfully!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape} (43 logits)")
    print(f"Total trainable parameters: {count_parameters(model):,}")
