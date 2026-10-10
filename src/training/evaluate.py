"""
Evaluation — Comprehensive model evaluation with metrics and visualizations.

Computes:
  - Accuracy, F1-Score (weighted & per-class), Precision, Recall
  - Confusion matrix (seaborn heatmap)
  - ROC curve with AUC
  - Classification report
  - Outputs metrics as JSON for DVC tracking
"""

import os
import sys
import json
from typing import Dict, Tuple

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CI/CD
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from tqdm import tqdm

from src.training.config import load_config
from src.models.densenet import build_densenet121
from src.models.utils import load_model
from src.data.dataset import get_dataloaders, CLASS_NAMES


@torch.no_grad()
def evaluate_model(
    model, dataloader, device: str
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model in eval mode.
        dataloader: DataLoader to evaluate on.
        device: Device string.

    Returns:
        Tuple of (metrics_dict, all_labels, all_probs).
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1_weighted": float(f1_score(all_labels, all_preds, average="weighted")),
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro")),
        "f1_per_class": {
            CLASS_NAMES[i]: float(v)
            for i, v in enumerate(f1_score(all_labels, all_preds, average=None))
        },
        "precision_weighted": float(
            precision_score(all_labels, all_preds, average="weighted")
        ),
        "recall_weighted": float(
            recall_score(all_labels, all_preds, average="weighted")
        ),
        "precision_per_class": {
            CLASS_NAMES[i]: float(v)
            for i, v in enumerate(precision_score(all_labels, all_preds, average=None))
        },
        "recall_per_class": {
            CLASS_NAMES[i]: float(v)
            for i, v in enumerate(recall_score(all_labels, all_preds, average=None))
        },
        "total_samples": int(len(all_labels)),
        "class_distribution": {
            CLASS_NAMES[i]: int(np.sum(all_labels == i))
            for i in range(len(CLASS_NAMES))
        },
    }

    return metrics, all_labels, all_probs


def plot_confusion_matrix(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: str,
    class_names: list = None,
):
    """
    Plot and save a confusion matrix heatmap.

    Args:
        labels: Ground truth labels.
        probs: Predicted probabilities.
        save_path: Path to save the plot.
        class_names: List of class names.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    preds = np.argmax(probs, axis=1)
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix — DenseNet121", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: str,
):
    """
    Plot and save ROC curve with AUC.

    Args:
        labels: Ground truth labels.
        probs: Predicted probabilities (shape: [N, num_classes]).
        save_path: Path to save the plot.
    """
    # Binary ROC: use probability of PNEUMONIA class
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr,
        color="#1e88e5",
        linewidth=2,
        label=f"ROC Curve (AUC = {roc_auc:.4f})",
    )
    plt.plot(
        [0, 1], [0, 1],
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Random Baseline",
    )
    plt.fill_between(fpr, tpr, alpha=0.1, color="#1e88e5")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve — DenseNet121", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved to {save_path}")

    return roc_auc


def run_evaluation(config=None):
    """
    Full evaluation pipeline.

    1. Load config
    2. Load best model
    3. Evaluate on test set
    4. Generate confusion matrix & ROC curve plots
    5. Save metrics JSON for DVC

    Args:
        config: Config object. If None, loaded from YAML files.
    """
    if config is None:
        config = load_config()

    device = config.device

    print(f"\n{'='*60}")
    print(f"  Evaluating DenseNet121 — Chest X-ray Pneumonia Detection")
    print(f"{'='*60}")

    # Load model
    model_path = config.paths.best_model
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("  Run training first: python -m src.training.train")
        sys.exit(1)

    model = load_model(model_path, num_classes=config.train.num_classes, device=device)

    # Load test data
    data_dir = config.data.processed_dir
    if not os.path.exists(data_dir):
        data_dir = config.data.raw_dir

    _, _, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        image_size=config.data.image_size,
    )

    # Evaluate
    metrics, all_labels, all_probs = evaluate_model(model, test_loader, device)

    # Print results
    print(f"\n  Results on Test Set ({metrics['total_samples']} samples)")
    print(f"  {'─'*40}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  F1 (weighted):     {metrics['f1_weighted']:.4f}")
    print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"  Precision:         {metrics['precision_weighted']:.4f}")
    print(f"  Recall:            {metrics['recall_weighted']:.4f}")
    print(f"\n  Per-class F1:")
    for cls_name, f1_val in metrics["f1_per_class"].items():
        print(f"    {cls_name}: {f1_val:.4f}")

    # Classification report
    preds = np.argmax(all_probs, axis=1)
    print(f"\n  Classification Report:")
    print(classification_report(all_labels, preds, target_names=CLASS_NAMES))

    # Generate plots
    plot_confusion_matrix(
        all_labels, all_probs, config.paths.confusion_matrix
    )

    roc_auc = plot_roc_curve(
        all_labels, all_probs, config.paths.roc_curve
    )
    metrics["roc_auc"] = float(roc_auc)

    # Save metrics JSON for DVC
    os.makedirs(os.path.dirname(config.paths.eval_metrics), exist_ok=True)
    with open(config.paths.eval_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nEvaluation metrics saved to {config.paths.eval_metrics}")

    # Validation gates
    print(f"\n  Validation Gates:")
    f1_pass = metrics["f1_weighted"] >= config.validation.f1_threshold
    acc_pass = metrics["accuracy"] >= config.validation.accuracy_threshold
    print(f"  F1 ≥ {config.validation.f1_threshold}:       {'✓ PASS' if f1_pass else '✗ FAIL'} ({metrics['f1_weighted']:.4f})")
    print(f"  Accuracy ≥ {config.validation.accuracy_threshold}:  {'✓ PASS' if acc_pass else '✗ FAIL'} ({metrics['accuracy']:.4f})")

    print(f"\n{'='*60}")

    return metrics


if __name__ == "__main__":
    run_evaluation()
