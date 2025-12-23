"""EfficientNet classifier for Steel Defect Classification."""

from typing import Tuple

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetClassifier(nn.Module):
    """EfficientNet-based classifier for defect types."""

    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        """
        Initialize EfficientNet classifier.

        Args:
            model_name: EfficientNet model variant
            num_classes: Number of defect classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
        """
        super().__init__()

        if pretrained:
            self.model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        else:
            self.model = EfficientNet.from_name(model_name, num_classes=num_classes)

        # Modify dropout
        self.model._dropout = nn.Dropout(p=dropout)

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class probabilities and labels.

        Args:
            x: Input tensor

        Returns:
            Tuple of (probabilities, predicted_labels)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return probs, preds

    def freeze_encoder(self):
        """Freeze encoder layers for transfer learning."""
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze classifier head
        for param in self.model._fc.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True
