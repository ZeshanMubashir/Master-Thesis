"""Model definitions for PV anomaly classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class ViTClassifier(nn.Module):
    """ViT-B/32 backbone with a custom classification head."""

    def __init__(
        self,
        *,
        input_shape: tuple[int, int, int],
        num_classes: int = 2,
        model_name: str = "vit_base_patch32_224",
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            img_size=input_shape[0],
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1])
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        print(f"Model created with feature dimension: {feature_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
