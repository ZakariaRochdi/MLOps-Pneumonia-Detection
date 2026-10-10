"""DenseNet121 model for pneumonia detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DenseNet121(nn.Module):
    """DenseNet121 wrapper for binary pneumonia classification."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        if self.training and x.size(0) == 1:
            self.eval()
            out = self.backbone(x)
            self.train()
            return out
        return self.backbone(x)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def get_gradcam(self, x, target_layer=None):
        """Grad-CAM via forward hook + autograd.grad (robuste)."""
        if target_layer is None:
            target_layer = self.backbone.features.denseblock4

        was_training = self.training
        self.eval()

        activations = {}

        def forward_hook(module, input, output):
            activations["value"] = output

        fh = target_layer.register_forward_hook(forward_hook)

        try:
            x = x.clone().detach().requires_grad_(True)
            logits = self.forward(x)
            pred_class = int(logits.argmax(dim=1).item())
            score = logits[0, pred_class]

            feat = activations["value"]

            grads = torch.autograd.grad(
                outputs=score,
                inputs=feat,
                retain_graph=False,
                allow_unused=True,
            )[0]

            if grads is None:
                grads = torch.ones_like(feat)

            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * feat).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam,
                size=(x.size(2), x.size(3)),
                mode="bilinear",
                align_corners=False,
            )
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            return cam.detach().squeeze().cpu().numpy()
        finally:
            fh.remove()
            if was_training:
                self.train()


# Alias used by src/models/utils.py
DenseNet121Classifier = DenseNet121


def build_densenet121(num_classes: int = 2, pretrained: bool = True) -> DenseNet121:
    """Factory function."""
    return DenseNet121(num_classes=num_classes, pretrained=pretrained)
