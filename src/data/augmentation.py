"""
Augmentation — Extended data augmentation strategies for chest X-ray images.

Provides configurable augmentation pipelines to improve model generalization
and address class imbalance in the Pneumonia dataset.
"""

from torchvision import transforms


def get_augmentation_transforms(config: dict = None) -> transforms.Compose:
    """
    Build an augmentation pipeline from configuration.

    Supports:
        - Random rotation
        - Horizontal/vertical flip
        - Random affine (translation + zoom)
        - Color jitter (brightness, contrast)
        - Gaussian blur
        - Random erasing

    Args:
        config: Dictionary with augmentation parameters.
                Expected keys: rotation_range, horizontal_flip, vertical_flip,
                zoom_range, brightness, contrast, gaussian_blur,
                gaussian_blur_kernel, random_erasing.

    Returns:
        torchvision.transforms.Compose pipeline with augmentation transforms.
    """
    if config is None:
        config = _default_config()

    augment_list = []

    # Rotation
    rotation_range = config.get("rotation_range", 15)
    if rotation_range > 0:
        augment_list.append(
            transforms.RandomRotation(degrees=rotation_range)
        )

    # Horizontal flip
    if config.get("horizontal_flip", True):
        augment_list.append(transforms.RandomHorizontalFlip(p=0.5))

    # Vertical flip
    if config.get("vertical_flip", False):
        augment_list.append(transforms.RandomVerticalFlip(p=0.5))

    # Random affine (translation + zoom)
    zoom_range = config.get("zoom_range", 0.1)
    if zoom_range > 0:
        augment_list.append(
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(1.0 - zoom_range, 1.0 + zoom_range),
            )
        )

    # Color jitter
    brightness = config.get("brightness", 0.2)
    contrast = config.get("contrast", 0.2)
    if brightness > 0 or contrast > 0:
        augment_list.append(
            transforms.ColorJitter(brightness=brightness, contrast=contrast)
        )

    # Gaussian blur
    if config.get("gaussian_blur", False):
        kernel_size = config.get("gaussian_blur_kernel", 3)
        augment_list.append(
            transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))
        )

    # Random erasing (applied after ToTensor)
    # This is returned separately since it must be applied after ToTensor
    return transforms.Compose(augment_list)


def get_post_tensor_augmentations(config: dict = None) -> transforms.Compose:
    """
    Augmentations that must be applied AFTER ToTensor (operate on tensors).

    Args:
        config: Dictionary with augmentation parameters.

    Returns:
        torchvision.transforms.Compose pipeline for post-tensor augmentations.
    """
    if config is None:
        config = _default_config()

    post_list = []

    if config.get("random_erasing", False):
        post_list.append(
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
        )

    if not post_list:
        return transforms.Compose([])  # No-op

    return transforms.Compose(post_list)


def build_full_train_pipeline(image_size: int = 224, config: dict = None):
    """
    Build the complete training pipeline combining preprocessing + augmentation.

    This is a convenience function that chains:
        1. Resize → RandomResizedCrop
        2. Augmentation transforms (rotation, flip, jitter, etc.)
        3. ToTensor → Normalize
        4. Post-tensor augmentations (random erasing)

    Args:
        image_size: Target image size (default 224).
        config: Augmentation configuration dictionary.

    Returns:
        torchvision.transforms.Compose — complete training pipeline.
    """
    from src.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD

    if config is None:
        config = _default_config()

    pre_tensor = []

    # Resize
    pre_tensor.append(transforms.Resize(256))
    pre_tensor.append(
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0))
    )

    # Augmentation transforms
    aug_transforms = get_augmentation_transforms(config)
    pre_tensor.append(aug_transforms)

    # To tensor + normalize
    pre_tensor.append(transforms.ToTensor())
    pre_tensor.append(
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    )

    # Post-tensor augmentations
    post_tensor = get_post_tensor_augmentations(config)
    pre_tensor.append(post_tensor)

    return transforms.Compose(pre_tensor)


def _default_config() -> dict:
    """Default augmentation configuration."""
    return {
        "rotation_range": 15,
        "horizontal_flip": True,
        "vertical_flip": False,
        "zoom_range": 0.1,
        "brightness": 0.2,
        "contrast": 0.2,
        "gaussian_blur": True,
        "gaussian_blur_kernel": 3,
        "random_erasing": False,
    }
