"""
Evaluation — Comprehensive model evaluation with metrics and visualizations.
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from tqdm import tqdm

from src.data.dataset import CLASS_NAMES, get_dataloaders
from src.models.utils import load_model
from src.training.config import load_config

matplotlib.use("Agg")


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def run_inference(model, dataloader, device):
    model.eval()
    model.to(device)
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    return (
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_probs, axis=0),
    )


def compute_metrics(labels, preds):
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
        "precision_weighted": float(
            precision_score(labels, preds, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(labels, preds, average="weighted", zero_division=0)
        ),
    }


def per_class_metrics(labels, preds, class_names):
    return classification_report(
        labels,
        preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def plot_confusion_matrix(labels, preds, class_names, output_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curve(labels, probs, class_names, output_path):
    num_classes = probs.shape[1]
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        binary_labels = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    config = load_config()
    device = _get_device()

    reports_dir = "reports"
    figures_dir = os.path.join(reports_dir, "figures")
    _ensure_dir(reports_dir)
    _ensure_dir(figures_dir)

    _, _, test_loader = get_dataloaders(config)

    model_path = config.get("model_path", "models/best_model.pth")
    model = load_model(model_path, device=device)

    labels, preds, probs = run_inference(model, test_loader, device)

    metrics = compute_metrics(labels, preds)
    report = per_class_metrics(labels, preds, CLASS_NAMES)

    metrics_path = os.path.join(reports_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_path = os.path.join(reports_dir, "classification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    plot_confusion_matrix(
        labels, preds, CLASS_NAMES,
        os.path.join(figures_dir, "confusion_matrix.png"),
    )
    plot_roc_curve(
        labels, probs, CLASS_NAMES,
        os.path.join(figures_dir, "roc_curve.png"),
    )

    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key:>22}: {value:.4f}")
    print("=" * 60)
    print(f"Metrics saved to   : {metrics_path}")
    print(f"Report saved to    : {report_path}")
    print(f"Figures saved to   : {figures_dir}")


if __name__ == "__main__":
    main()
