import torch
"""
Preprocessing — Image transformation pipelines for training and inference.

Provides standardized transforms for chest X-ray images:
  - Resize to 224x224 (DenseNet121 input requirement)
  - ImageNet normalization
  - Grayscale-to-RGB conversion
"""

from torchvision import transforms

# ImageNet normalization statistics (used because DenseNet121 is pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Training transforms with data augmentation.

    Pipeline:
        1. Resize to slightly larger than target (256px)
        2. Random crop to 224x224
        3. Random horizontal flip
        4. Random rotation (±15°)
        5. Color jitter for brightness/contrast variation
        6. Convert to tensor
        7. Normalize with ImageNet stats

    Args:
        image_size: Target image size (default 224 for DenseNet121).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Validation/Test transforms — deterministic, no augmentation.

    Pipeline:
        1. Resize to 256px
        2. Center crop to 224x224
        3. Convert to tensor
        4. Normalize with ImageNet stats

    Args:
        image_size: Target image size (default 224 for DenseNet121).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Inference transforms — identical to validation but kept separate
    for clarity in the serving pipeline.

    Args:
        image_size: Target image size (default 224 for DenseNet121).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return get_val_transforms(image_size)


def denormalize(tensor):
    """
    Reverse ImageNet normalization for visualization purposes.

    Args:
        tensor: Normalized image tensor of shape (C, H, W).

    Returns:
        Denormalized tensor with pixel values in [0, 1].
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean


# Needed for denormalize — imported here to avoid top-level import overhead
# when only transforms are needed
