"""
Dataset — PyTorch Dataset and DataLoader factory for Chest X-ray images.

Handles:
  - Loading images from the standard directory structure (train/val/test × NORMAL/PNEUMONIA)
  - Grayscale-to-RGB conversion
  - Applying preprocessing transforms
  - Weighted sampling to address class imbalance
  - Data preparation (copying + structuring processed data)
"""

import os
import sys
import shutil
import random
from pathlib import Path
from typing import Tuple, Optional, Dict

import yaml
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from PIL import Image

from src.data.preprocessing import get_train_transforms, get_val_transforms


# Class label mapping
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}


class ChestXrayDataset(Dataset):
    """
    Custom PyTorch Dataset for Chest X-ray images.

    Reads images from a directory with subdirectories per class:
        root/
            NORMAL/
                img1.jpeg
                img2.jpeg
            PNEUMONIA/
                img1.jpeg
                img2.jpeg

    Handles grayscale-to-RGB conversion automatically.

    Args:
        root_dir: Path to the dataset split directory.
        transform: torchvision transforms to apply.
        target_transform: Transform to apply to labels.
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        target_transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform

        # Collect all image paths and labels
        self.samples = []
        self.targets = []

        for class_name in CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            class_idx = CLASS_TO_IDX[class_name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in (".jpeg", ".jpg", ".png", ".bmp"):
                    self.samples.append((str(img_path), class_idx))
                    self.targets.append(class_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image and convert to RGB (chest X-rays may be grayscale)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Return count of samples per class."""
        dist = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            dist[CLASS_NAMES[label]] += 1
        return dist


def get_weighted_sampler(dataset: ChestXrayDataset) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to address class imbalance.

    Oversamples the minority class (NORMAL) so each batch has approximately
    equal representation of both classes.

    Args:
        dataset: ChestXrayDataset instance.

    Returns:
        WeightedRandomSampler for the DataLoader.
    """
    targets = dataset.targets
    class_counts = [0, 0]
    for t in targets:
        class_counts[t] += 1

    # Weight is inverse of class frequency
    class_weights = [1.0 / c for c in class_counts]
    sample_weights = [class_weights[t] for t in targets]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_dataloaders(
    data_dir: str = "chest_xray",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    val_split: float = 0.15,
    use_weighted_sampler: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function to create train/val/test DataLoaders.

    Since the original validation set is very small (16 images), this function
    optionally splits a portion of the training set for validation.

    Args:
        data_dir: Root data directory containing train/val/test subdirectories.
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers.
        image_size: Target image size (224 for DenseNet121).
        val_split: Fraction of training data to use as validation.
        use_weighted_sampler: Whether to use weighted sampling for training.
        pin_memory: Pin memory for GPU transfer.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_transforms = get_train_transforms(image_size)
    val_transforms = get_val_transforms(image_size)

    # Load full training set
    full_train = ChestXrayDataset(
        root_dir=os.path.join(data_dir, "train"),
        transform=train_transforms,
    )

    # Split training into train + validation
    total = len(full_train)
    val_size = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train, [train_size, val_size], generator=generator
    )

    # Override transform for validation subset (no augmentation)
    val_dataset_wrapper = _SubsetWithTransform(val_dataset, val_transforms)

    # Test dataset
    test_dataset = ChestXrayDataset(
        root_dir=os.path.join(data_dir, "test"),
        transform=val_transforms,
    )

    # Weighted sampler for training
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        # Build sampler from the training subset targets
        subset_targets = [full_train.targets[i] for i in train_dataset.indices]
        class_counts = [0, 0]
        for t in subset_targets:
            class_counts[t] += 1
        class_weights = [1.0 / c for c in class_counts]
        sample_weights = [class_weights[t] for t in subset_targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False  # Sampler and shuffle are mutually exclusive

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset_wrapper,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


class _SubsetWithTransform(Dataset):
    """
    Wrapper to apply a different transform to a Subset.

    When we split the training data, the validation subset should use
    validation transforms (no augmentation), not the training transforms.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        # image is already transformed by the parent dataset's transform.
        # We need to access the raw image instead.
        original_idx = self.subset.indices[idx]
        img_path, label = self.subset.dataset.samples[original_idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def prepare_data(params_path: str = "params.yaml"):
    """
    Prepare processed data directory from raw data.

    Reads parameters from params.yaml and copies/structures the data
    for reproducible pipeline execution via DVC.

    Args:
        params_path: Path to the DVC parameters file.
    """
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    raw_dir = params["data"]["raw_dir"]
    processed_dir = params["data"]["processed_dir"]

    # Create processed directory structure
    os.makedirs(processed_dir, exist_ok=True)

    for split in ["train", "test", "val"]:
        for cls in CLASS_NAMES:
            src = os.path.join(raw_dir, split, cls)
            dst = os.path.join(processed_dir, split, cls)

            if not os.path.exists(src):
                print(f"  [SKIP] {src} does not exist")
                continue

            os.makedirs(dst, exist_ok=True)

            # Copy images
            count = 0
            for img_file in os.listdir(src):
                if img_file.lower().endswith((".jpeg", ".jpg", ".png", ".bmp")):
                    shutil.copy2(
                        os.path.join(src, img_file),
                        os.path.join(dst, img_file),
                    )
                    count += 1

            print(f"  [OK] {split}/{cls}: {count} images copied")

    print(f"\nData prepared in '{processed_dir}'")


if __name__ == "__main__":
    if "--prepare" in sys.argv:
        print("=" * 60)
        print("Preparing data...")
        print("=" * 60)
        prepare_data()
    else:
        # Quick test: load dataloaders and print info
        print("Loading dataloaders...")
        train_loader, val_loader, test_loader = get_dataloaders()
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches:   {len(val_loader)}")
        print(f"Test batches:  {len(test_loader)}")

        # Print a sample batch shape
        images, labels = next(iter(train_loader))
        print(f"Batch shape:   {images.shape}")
        print(f"Labels shape:  {labels.shape}")
