"""
DenseNet121 — Model definition with transfer learning for Pneumonia detection.

Architecture (from diagram):
  - Base: DenseNet121 pretrained on ImageNet (torchvision)
  - Modified classifier head: 1024 → dropout → num_classes
  - Optional backbone freezing for feature extraction mode
  - GradCAM-compatible hook registration
"""

import torch
from torch import nn
from torchvision import models
from torchvision.models import DenseNet121_Weights


class DenseNet121Classifier(nn.Module):
    """
    DenseNet121 for binary classification (NORMAL vs PNEUMONIA).

    Uses transfer learning from ImageNet pretrained weights.
    The classifier head is replaced with a custom fully connected layer.

    Args:
        num_classes: Number of output classes (default 2).
        pretrained: Whether to load ImageNet pretrained weights.
        dropout: Dropout rate before the final classifier layer.
        freeze_backbone: If True, freeze all DenseNet layers except classifier.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Load pretrained DenseNet121
        if pretrained:
            weights = DenseNet121_Weights.IMAGENET1K_V1
            self.backbone = models.densenet121(weights=weights)
        else:
            self.backbone = models.densenet121(weights=None)

        # Get the number of features from the original classifier
        num_features = self.backbone.classifier.in_features  # 1024

        # Replace the classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()

        # GradCAM hook storage
        self._gradients = None
        self._activations = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities using softmax.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            Probability tensor of shape (batch_size, num_classes).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def _freeze_backbone(self):
        """Freeze all layers except the classifier head."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self, num_layers: int | None = None):
        """
        Unfreeze backbone layers for fine-tuning.

        Args:
            num_layers: If specified, only unfreeze the last N DenseBlocks.
                        If None, unfreeze all layers.
        """
        if num_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # DenseNet121 has 4 DenseBlocks
            dense_blocks = ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]
            blocks_to_unfreeze = dense_blocks[-num_layers:]

            for name, param in self.backbone.named_parameters():
                if (
                    any(block in name for block in blocks_to_unfreeze)
                    or "classifier" in name
                ):
                    param.requires_grad = True

    def register_gradcam_hooks(self):
        """
        Register forward and backward hooks on the last convolutional layer
        for GradCAM visualization.
        """
        # The last feature layer in DenseNet121 is backbone.features.denseblock4
        target_layer = self.backbone.features[-1]  # Last norm layer

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def get_gradcam(
        self, x: torch.Tensor, target_class: int | None = None
    ) -> torch.Tensor:
        """
        Generate GradCAM heatmap for the input image.

        Args:
            x: Input tensor of shape (1, 3, 224, 224).
            target_class: Class index to compute GradCAM for.
                          If None, uses the predicted class.

        Returns:
            Heatmap tensor of shape (H, W) with values in [0, 1].
        """
        self.eval()
        self.register_gradcam_hooks()

        # Forward pass
        x.requires_grad_(True)
        logits = self.forward(x)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward pass for the target class
        self.zero_grad()
        logits[0, target_class].backward()

        # Compute GradCAM
        gradients = self._gradients  # (1, C, H, W)
        activations = self._activations  # (1, C, H, W)

        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)  # ReLU to keep only positive contributions

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input size
        cam = torch.nn.functional.interpolate(
            cam, size=(224, 224), mode="bilinear", align_corners=False
        )

        return cam.squeeze().detach()

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Count model parameters.

        Args:
            trainable_only: If True, count only trainable parameters.

        Returns:
            Total number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def build_densenet121(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> DenseNet121Classifier:
    """
    Factory function to build a DenseNet121 classifier.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.
        dropout: Dropout rate.
        freeze_backbone: Whether to freeze backbone layers.

    Returns:
        DenseNet121Classifier model instance.
    """
    model = DenseNet121Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )

    total_params = model.get_num_parameters()
    trainable_params = model.get_num_parameters(trainable_only=True)
    print("DenseNet121 built:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model
