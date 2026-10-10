"""
Test Preprocessing — Unit tests for the image preprocessing pipeline.

Validates:
  - Output image dimensions (224×224)
  - Normalization value ranges
  - Grayscale-to-RGB conversion
  - Augmented output shapes
  - DataLoader batch shapes
  - Label encoding (NORMAL=0, PNEUMONIA=1)
"""

import os
import pytest
import torch
import numpy as np
from PIL import Image

from src.data.preprocessing import (
    get_train_transforms,
    get_val_transforms,
    get_inference_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.data.augmentation import (
    get_augmentation_transforms,
    build_full_train_pipeline,
)
from src.data.dataset import CLASS_TO_IDX, CLASS_NAMES


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_rgb_image():
    """Create a sample 512x512 RGB image."""
    return Image.fromarray(
        np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    )


@pytest.fixture
def sample_grayscale_image():
    """Create a sample 512x512 grayscale image."""
    return Image.fromarray(
        np.random.randint(0, 256, (512, 512), dtype=np.uint8), mode="L"
    )


@pytest.fixture
def sample_small_image():
    """Create a small 64x64 RGB image (smaller than target 224)."""
    return Image.fromarray(
        np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    )


# ── Resize Tests ─────────────────────────────────────────────────────────────


class TestImageResize:
    """Tests for image resize to 224×224."""

    def test_train_transform_output_size(self, sample_rgb_image):
        """Training transforms should produce 224×224 output."""
        transform = get_train_transforms(image_size=224)
        output = transform(sample_rgb_image)
        assert output.shape == (3, 224, 224), f"Expected (3, 224, 224), got {output.shape}"

    def test_val_transform_output_size(self, sample_rgb_image):
        """Validation transforms should produce 224×224 output."""
        transform = get_val_transforms(image_size=224)
        output = transform(sample_rgb_image)
        assert output.shape == (3, 224, 224), f"Expected (3, 224, 224), got {output.shape}"

    def test_inference_transform_output_size(self, sample_rgb_image):
        """Inference transforms should produce 224×224 output."""
        transform = get_inference_transforms(image_size=224)
        output = transform(sample_rgb_image)
        assert output.shape == (3, 224, 224)

    def test_custom_image_size(self, sample_rgb_image):
        """Transforms should support custom image sizes."""
        transform = get_val_transforms(image_size=128)
        output = transform(sample_rgb_image)
        assert output.shape == (3, 128, 128)

    def test_small_image_upscale(self, sample_small_image):
        """Small images should be upscaled to 224×224."""
        transform = get_val_transforms(image_size=224)
        output = transform(sample_small_image)
        assert output.shape == (3, 224, 224)


# ── Normalization Tests ──────────────────────────────────────────────────────


class TestNormalization:
    """Tests for ImageNet normalization."""

    def test_normalization_applied(self, sample_rgb_image):
        """Output tensor should be normalized (not in [0, 255] range)."""
        transform = get_val_transforms(image_size=224)
        output = transform(sample_rgb_image)
        # After normalization, values should NOT be in [0, 255]
        assert output.max() < 10.0, "Values too large, normalization may not be applied"

    def test_normalization_range(self, sample_rgb_image):
        """Normalized values should be roughly in [-3, 3] range."""
        transform = get_val_transforms(image_size=224)
        output = transform(sample_rgb_image)
        assert output.min() > -5.0, f"Min value too small: {output.min()}"
        assert output.max() < 5.0, f"Max value too large: {output.max()}"

    def test_output_is_float_tensor(self, sample_rgb_image):
        """Output should be a float tensor."""
        transform = get_val_transforms(image_size=224)
        output = transform(sample_rgb_image)
        assert output.dtype == torch.float32


# ── Grayscale-to-RGB Tests ───────────────────────────────────────────────────


class TestGrayscaleConversion:
    """Tests for grayscale-to-RGB conversion."""

    def test_grayscale_to_rgb_channels(self, sample_grayscale_image):
        """Grayscale images should produce 3-channel output."""
        rgb_image = sample_grayscale_image.convert("RGB")
        transform = get_val_transforms(image_size=224)
        output = transform(rgb_image)
        assert output.shape[0] == 3, f"Expected 3 channels, got {output.shape[0]}"

    def test_grayscale_to_rgb_shape(self, sample_grayscale_image):
        """Grayscale converted to RGB should have correct output shape."""
        rgb_image = sample_grayscale_image.convert("RGB")
        transform = get_val_transforms(image_size=224)
        output = transform(rgb_image)
        assert output.shape == (3, 224, 224)


# ── Augmentation Tests ───────────────────────────────────────────────────────


class TestAugmentation:
    """Tests for data augmentation transforms."""

    def test_augmentation_output_shape(self, sample_rgb_image):
        """Augmented images should maintain correct dimensions."""
        pipeline = build_full_train_pipeline(image_size=224)
        output = pipeline(sample_rgb_image)
        assert output.shape == (3, 224, 224)

    def test_augmentation_produces_variation(self, sample_rgb_image):
        """Multiple augmentations should produce different outputs."""
        transform = get_train_transforms(image_size=224)
        outputs = [transform(sample_rgb_image) for _ in range(5)]
        # At least some outputs should differ (probabilistic augmentation)
        unique_sums = len(set(o.sum().item() for o in outputs))
        assert unique_sums >= 2, "Augmentation should produce variation"

    def test_augmentation_default_config(self, sample_rgb_image):
        """Default augmentation config should work without errors."""
        augment = get_augmentation_transforms()
        output = augment(sample_rgb_image)
        assert isinstance(output, Image.Image)


# ── Label Encoding Tests ────────────────────────────────────────────────────


class TestLabelEncoding:
    """Tests for class label encoding."""

    def test_normal_class_index(self):
        """NORMAL class should be encoded as 0."""
        assert CLASS_TO_IDX["NORMAL"] == 0

    def test_pneumonia_class_index(self):
        """PNEUMONIA class should be encoded as 1."""
        assert CLASS_TO_IDX["PNEUMONIA"] == 1

    def test_class_names_order(self):
        """Class names should be in correct order."""
        assert CLASS_NAMES[0] == "NORMAL"
        assert CLASS_NAMES[1] == "PNEUMONIA"

    def test_num_classes(self):
        """Should have exactly 2 classes."""
        assert len(CLASS_NAMES) == 2
        assert len(CLASS_TO_IDX) == 2


# ── DataLoader Tests (requires data) ────────────────────────────────────────


class TestDataLoader:
    """Tests for DataLoader batch shapes. Skipped if data is not available."""

    @pytest.fixture
    def data_dir(self):
        """Check if data directory exists."""
        for path in ["data/processed", "chest_xray"]:
            if os.path.exists(os.path.join(path, "train")):
                return path
        pytest.skip("Dataset not available")

    def test_dataloader_batch_shape(self, data_dir):
        """DataLoader batches should have shape (B, 3, 224, 224)."""
        from src.data.dataset import get_dataloaders

        train_loader, _, _ = get_dataloaders(
            data_dir=data_dir, batch_size=4, num_workers=0
        )
        images, labels = next(iter(train_loader))
        assert images.shape == (4, 3, 224, 224), f"Got {images.shape}"
        assert labels.shape == (4,), f"Got {labels.shape}"

    def test_dataloader_label_range(self, data_dir):
        """Labels should be 0 or 1."""
        from src.data.dataset import get_dataloaders

        train_loader, _, _ = get_dataloaders(
            data_dir=data_dir, batch_size=16, num_workers=0
        )
        _, labels = next(iter(train_loader))
        assert labels.min() >= 0
        assert labels.max() <= 1
