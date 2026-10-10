"""
Model Utilities — Save, load, and export model artifacts.

Provides standardized model persistence for:
  - PyTorch checkpoint save/load (state_dict + metadata)
  - ONNX export for production serving
  - Model info extraction
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.densenet import DenseNet121Classifier, build_densenet121


def save_model(
    model: nn.Module,
    path: str,
    optimizer=None,
    epoch: int = None,
    metrics: Dict = None,
    config: Dict = None,
):
    """
    Save model checkpoint with metadata.

    Saves:
      - model state_dict
      - optimizer state_dict (optional)
      - epoch number
      - metrics (accuracy, f1, etc.)
      - config (hyperparameters)
      - timestamp

    Args:
        model: PyTorch model to save.
        path: File path for the checkpoint (.pth).
        optimizer: Optimizer to save state (optional).
        epoch: Current epoch number.
        metrics: Dictionary of evaluation metrics.
        config: Training configuration dictionary.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if metrics is not None:
        checkpoint["metrics"] = metrics
    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(
    path: str,
    num_classes: int = 2,
    device: str = None,
    load_weights_only: bool = True,
) -> DenseNet121Classifier:
    """
    Load a DenseNet121 model from a checkpoint.

    Args:
        path: Path to the checkpoint file (.pth).
        num_classes: Number of output classes.
        device: Device to load model onto (auto-detected if None).
        load_weights_only: If True, only load model weights (not optimizer state).

    Returns:
        Loaded DenseNet121Classifier model in eval mode.
    """
    if device is None:
        device = get_device()

    # Build model architecture
    model = build_densenet121(
        num_classes=num_classes,
        pretrained=False,  # Don't load ImageNet weights, we'll load our own
    )

    # Load checkpoint
    checkpoint = torch.load(path, map_location=device, weights_only=load_weights_only)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {path}")
        if "epoch" in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if "metrics" in checkpoint:
            print(f"  Metrics: {checkpoint['metrics']}")
    else:
        # Direct state_dict
        model.load_state_dict(checkpoint)
        print(f"Model state_dict loaded from {path}")

    model = model.to(device)
    model.eval()
    return model


def export_to_onnx(
    model: nn.Module,
    path: str,
    input_size: tuple = (1, 3, 224, 224),
    device: str = None,
):
    """
    Export model to ONNX format for production serving.

    Args:
        model: PyTorch model to export.
        path: Output path for the ONNX file.
        input_size: Input tensor shape (batch, channels, height, width).
        device: Device for the dummy input.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_size).to(device)

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Model exported to ONNX: {path}")


def get_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device


def get_model_info(model: nn.Module) -> Dict:
    """
    Extract model information.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with model info (parameter counts, layer names, etc.).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "model_size_mb": sum(
            p.nelement() * p.element_size() for p in model.parameters()
        )
        / (1024 * 1024),
    }
