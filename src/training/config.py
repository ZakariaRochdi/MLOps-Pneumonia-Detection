"""
Training Configuration — Dataclass-based config loaded from YAML.

Centralizes all hyperparameters and settings for reproducible training runs.
Supports loading from both params.yaml (DVC) and configs/training_config.yaml.
"""

import os
from dataclasses import dataclass, field

import torch
import yaml


@dataclass
class DataConfig:
    """Data loading configuration."""

    raw_dir: str = "chest_xray"
    processed_dir: str = "data/processed"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    val_split: float = 0.15
    pin_memory: bool = True


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    epochs: int = 25
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    optimizer: str = "adam"
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 3
    scheduler_factor: float = 0.1
    early_stopping_patience: int = 7
    model_name: str = "densenet121"
    pretrained: bool = True
    num_classes: int = 2
    class_weights: list[float] = field(default_factory=lambda: [2.89, 1.0])
    freeze_backbone: bool = False
    dropout: float = 0.3
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    seed: int = 42


@dataclass
class AugmentationConfig:
    """Data augmentation parameters."""

    enabled: bool = True
    rotation_range: int = 15
    horizontal_flip: bool = True
    vertical_flip: bool = False
    zoom_range: float = 0.1
    brightness: float = 0.2
    contrast: float = 0.2
    gaussian_blur: bool = True
    gaussian_blur_kernel: int = 3
    random_erasing: bool = False


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""

    name: str = "chest-xray-pneumonia"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_enabled: bool = True
    wandb_project: str = "chest-xray-pneumonia"
    wandb_entity: str | None = None
    wandb_enabled: bool = False


@dataclass
class ValidationConfig:
    """Model validation gate thresholds."""

    f1_threshold: float = 0.95
    accuracy_threshold: float = 0.90


@dataclass
class PathsConfig:
    """File paths for artifacts."""

    model_dir: str = "models"
    reports_dir: str = "reports"
    best_model: str = "models/best_model.pth"
    training_history: str = "reports/training_history.csv"
    train_metrics: str = "reports/train_metrics.json"
    eval_metrics: str = "reports/eval_metrics.json"
    confusion_matrix: str = "reports/confusion_matrix.png"
    roc_curve: str = "reports/roc_curve.png"


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    device: str = "cpu"

    def __post_init__(self):
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"


def load_config(
    params_path: str = "params.yaml",
    config_path: str = "configs/training_config.yaml",
) -> Config:
    """
    Load configuration from YAML files.

    Priority: config_path > params_path > defaults.
    If config_path exists, it takes precedence. Otherwise, falls back to params.yaml.

    Args:
        params_path: Path to DVC params.yaml.
        config_path: Path to detailed training config YAML.

    Returns:
        Config dataclass with all settings.
    """
    config = Config()

    # Try detailed config first, then fall back to params.yaml
    yaml_path = config_path if os.path.exists(config_path) else params_path

    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        if raw:
            # Data config
            if "data" in raw:
                for key, val in raw["data"].items():
                    if hasattr(config.data, key):
                        setattr(config.data, key, val)

            # Training config
            train_key = "train" if "train" in raw else "training"
            if train_key in raw:
                for key, val in raw[train_key].items():
                    if hasattr(config.train, key):
                        setattr(config.train, key, val)

            # Augmentation config
            if "augmentation" in raw:
                for key, val in raw["augmentation"].items():
                    if hasattr(config.augmentation, key):
                        setattr(config.augmentation, key, val)

            # Experiment config
            if "experiment" in raw:
                for key, val in raw["experiment"].items():
                    if hasattr(config.experiment, key):
                        setattr(config.experiment, key, val)

            # MLflow config (from detailed config)
            if "mlflow" in raw:
                config.experiment.mlflow_enabled = raw["mlflow"].get("enabled", True)
                config.experiment.mlflow_tracking_uri = raw["mlflow"].get(
                    "tracking_uri", config.experiment.mlflow_tracking_uri
                )

            # W&B config (from detailed config)
            if "wandb" in raw:
                config.experiment.wandb_enabled = raw["wandb"].get("enabled", False)
                config.experiment.wandb_project = raw["wandb"].get(
                    "project", config.experiment.wandb_project
                )
                config.experiment.wandb_entity = raw["wandb"].get("entity", None)

            # Validation config
            if "validation" in raw:
                for key, val in raw["validation"].items():
                    if hasattr(config.validation, key):
                        setattr(config.validation, key, val)

            # Paths config
            if "paths" in raw:
                for key, val in raw["paths"].items():
                    if hasattr(config.paths, key):
                        setattr(config.paths, key, val)

        print(f"Configuration loaded from {yaml_path}")
    else:
        print("No config file found, using defaults")

    print(f"Device: {config.device}")
    return config
