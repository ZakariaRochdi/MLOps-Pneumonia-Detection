"""
Training Loop — Full training pipeline with MLflow/W&B experiment tracking.

Implements:
  - Training loop with weighted loss for class imbalance
  - Validation at each epoch
  - Learning rate scheduling (ReduceLROnPlateau)
  - Early stopping
  - MLflow experiment tracking + model registry
  - Weights & Biases logging (optional)
  - Checkpointing (best model by val F1-Score)
  - Mixed precision training (AMP)
  - Training history CSV export
"""

import os
import sys
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

from src.training.config import load_config
from src.models.densenet import build_densenet121
from src.models.utils import save_model, get_device
from src.data.dataset import get_dataloaders


def train_one_epoch(
    model, train_loader, criterion, optimizer, device, scaler=None, gradient_clip=1.0
):
    """
    Train for one epoch.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device string.
        scaler: GradScaler for mixed precision (optional).
        gradient_clip: Max gradient norm for clipping.

    Returns:
        Tuple of (average_loss, accuracy, f1_score).
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc="  Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        if scaler is not None and device == "cuda":
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(train_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, acc, f1


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """
    Validate the model.

    Args:
        model: PyTorch model.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: Device string.

    Returns:
        Tuple of (average_loss, accuracy, f1_score).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(val_loader, desc="  Validating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(val_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, acc, f1


def train(config=None):
    """
    Full training pipeline.

    1. Load configuration
    2. Create dataloaders
    3. Build DenseNet121 model
    4. Set up optimizer, scheduler, loss
    5. Train with experiment tracking
    6. Save best model

    Args:
        config: Config object. If None, loaded from YAML files.
    """
    # ── Load config ──────────────────────────────────────────────────────────
    if config is None:
        config = load_config()

    # Seed for reproducibility
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.train.seed)

    device = config.device
    print(f"\n{'='*60}")
    print(f"  Training DenseNet121 — Chest X-ray Pneumonia Detection")
    print(f"{'='*60}")
    print(f"  Device:         {device}")
    print(f"  Epochs:         {config.train.epochs}")
    print(f"  Batch size:     {config.data.batch_size}")
    print(f"  Learning rate:  {config.train.learning_rate}")
    print(f"  Class weights:  {config.train.class_weights}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    data_dir = config.data.processed_dir
    if not os.path.exists(data_dir):
        data_dir = config.data.raw_dir  # Fallback to raw data
        print(f"  Using raw data directory: {data_dir}")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        image_size=config.data.image_size,
        val_split=config.data.val_split,
        pin_memory=config.data.pin_memory,
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}\n")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_densenet121(
        num_classes=config.train.num_classes,
        pretrained=config.train.pretrained,
        dropout=config.train.dropout,
        freeze_backbone=config.train.freeze_backbone,
    )
    model = model.to(device)

    # ── Loss with class weights ──────────────────────────────────────────────
    class_weights = torch.tensor(config.train.class_weights, dtype=torch.float32)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    # ── Scheduler ────────────────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.train.scheduler_factor,
        patience=config.train.scheduler_patience,
        verbose=True,
    )

    # ── Mixed precision ──────────────────────────────────────────────────────
    scaler = GradScaler() if config.train.mixed_precision and device == "cuda" else None

    # ── Experiment tracking setup ────────────────────────────────────────────
    mlflow_run = None
    if config.experiment.mlflow_enabled:
        try:
            import mlflow
            mlflow.set_tracking_uri(config.experiment.mlflow_tracking_uri)
            mlflow.set_experiment(config.experiment.name)
            mlflow_run = mlflow.start_run()
            mlflow.log_params({
                "model": config.train.model_name,
                "epochs": config.train.epochs,
                "batch_size": config.data.batch_size,
                "learning_rate": config.train.learning_rate,
                "weight_decay": config.train.weight_decay,
                "optimizer": config.train.optimizer,
                "class_weights": str(config.train.class_weights),
                "image_size": config.data.image_size,
                "pretrained": config.train.pretrained,
                "dropout": config.train.dropout,
                "freeze_backbone": config.train.freeze_backbone,
            })
            print("  MLflow tracking enabled")
        except Exception as e:
            print(f"  MLflow not available: {e}")
            mlflow_run = None

    wandb_run = None
    if config.experiment.wandb_enabled:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config.experiment.wandb_project,
                entity=config.experiment.wandb_entity,
                config={
                    "model": config.train.model_name,
                    "epochs": config.train.epochs,
                    "batch_size": config.data.batch_size,
                    "learning_rate": config.train.learning_rate,
                    "weight_decay": config.train.weight_decay,
                    "image_size": config.data.image_size,
                },
                name=f"densenet121_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            wandb.watch(model, log="all", log_freq=10)
            print("  W&B tracking enabled")
        except Exception as e:
            print(f"  W&B not available: {e}")
            wandb_run = None

    # ── Training loop ────────────────────────────────────────────────────────
    os.makedirs(config.paths.model_dir, exist_ok=True)
    os.makedirs(config.paths.reports_dir, exist_ok=True)

    best_val_f1 = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, config.train.epochs + 1):
        print(f"\nEpoch {epoch}/{config.train.epochs}")
        print("-" * 40)

        epoch_start = time.time()

        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, gradient_clip=config.train.gradient_clip,
        )

        # Validate
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device,
        )

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # Log metrics
        print(f"  Train  — loss: {train_loss:.4f}  acc: {train_acc:.4f}  f1: {train_f1:.4f}")
        print(f"  Val    — loss: {val_loss:.4f}  acc: {val_acc:.4f}  f1: {val_f1:.4f}")
        print(f"  LR: {current_lr:.6f}  Time: {epoch_time:.1f}s")

        # Record history
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": current_lr,
            "time": epoch_time,
        }
        history.append(epoch_record)

        # MLflow logging
        if mlflow_run:
            try:
                import mlflow
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "lr": current_lr,
                }, step=epoch)
            except Exception:
                pass

        # W&B logging
        if wandb_run:
            try:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "train/f1": train_f1,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "val/f1": val_f1,
                    "lr": current_lr,
                })
            except Exception:
                pass

        # Checkpointing — save best model by val F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            save_model(
                model=model,
                path=config.paths.best_model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_f1": val_f1, "val_acc": val_acc, "val_loss": val_loss},
                config={
                    "model": config.train.model_name,
                    "num_classes": config.train.num_classes,
                    "image_size": config.data.image_size,
                },
            )
            print(f"  ★ New best model saved (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.train.early_stopping_patience})")

        # Early stopping
        if patience_counter >= config.train.early_stopping_patience:
            print(f"\n  Early stopping triggered after {epoch} epochs")
            break

    # ── Save training history ────────────────────────────────────────────────
    history_df = pd.DataFrame(history)
    history_df.to_csv(config.paths.training_history, index=False)
    print(f"\nTraining history saved to {config.paths.training_history}")

    # Save training metrics
    train_metrics = {
        "best_val_f1": float(best_val_f1),
        "best_val_acc": float(history_df.loc[history_df["val_f1"].idxmax(), "val_acc"]),
        "total_epochs": len(history),
        "total_time_seconds": sum(h["time"] for h in history),
        "final_lr": float(current_lr),
        "timestamp": datetime.now().isoformat(),
    }
    with open(config.paths.train_metrics, "w") as f:
        json.dump(train_metrics, f, indent=2)
    print(f"Training metrics saved to {config.paths.train_metrics}")

    # ── Finalize experiment tracking ─────────────────────────────────────────
    if mlflow_run:
        try:
            import mlflow
            import mlflow.pytorch
            mlflow.log_artifact(config.paths.training_history)
            mlflow.log_artifact(config.paths.train_metrics)
            mlflow.log_artifact(config.paths.best_model)
            mlflow.pytorch.log_model(model, "model")
            mlflow.end_run()
            print("  MLflow run completed")
        except Exception as e:
            print(f"  MLflow finalization error: {e}")

    if wandb_run:
        try:
            import wandb
            wandb.save(config.paths.best_model)
            wandb.finish()
            print("  W&B run completed")
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"  Training complete! Best val F1: {best_val_f1:.4f}")
    print(f"{'='*60}")

    return model, history


if __name__ == "__main__":
    train()
