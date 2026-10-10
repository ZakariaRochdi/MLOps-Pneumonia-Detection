"""
Test Model — Unit tests for the DenseNet121 model architecture.

Validates:
  - Output shape (batch_size, 2)
  - Forward pass without errors
  - Parameter count
  - Save/load roundtrip
  - Backbone freeze/unfreeze
  - GradCAM generation
"""

import os
import tempfile
import pytest
import torch
import numpy as np

from src.models.densenet import DenseNet121Classifier, build_densenet121
from src.models.utils import save_model, load_model, get_model_info


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def model():
    """Build a DenseNet121 model (not pretrained, for speed)."""
    return build_densenet121(
        num_classes=2, pretrained=False, dropout=0.3, freeze_backbone=False
    )


@pytest.fixture
def random_input():
    """Create a random input tensor matching DenseNet121 input spec."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def single_input():
    """Create a single random input tensor."""
    return torch.randn(1, 3, 224, 224)


# ── Output Shape Tests ──────────────────────────────────────────────────────


class TestModelOutputShape:
    """Tests for model output dimensions."""

    def test_output_shape_batch(self, model, random_input):
        """Output shape should be (batch_size, num_classes)."""
        output = model(random_input)
        assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"

    def test_output_shape_single(self, model, single_input):
        """Single input should produce (1, 2) output."""
        output = model(single_input)
        assert output.shape == (1, 2), f"Expected (1, 2), got {output.shape}"

    def test_output_shape_variable_batch(self, model):
        """Model should handle various batch sizes."""
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 224, 224)
            output = model(x)
            assert output.shape == (batch_size, 2)


# ── Forward Pass Tests ──────────────────────────────────────────────────────


class TestForwardPass:
    """Tests for model forward pass execution."""

    def test_forward_pass_no_error(self, model, random_input):
        """Forward pass should complete without errors."""
        output = model(random_input)
        assert output is not None
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_forward_pass_output_values(self, model, random_input):
        """Output logits should be finite numbers."""
        output = model(random_input)
        assert torch.isfinite(output).all(), "Output contains infinite values"

    def test_predict_proba(self, model, random_input):
        """predict_proba should return valid probabilities."""
        probs = model.predict_proba(random_input)
        assert probs.shape == (2, 2)
        # Probabilities should sum to 1
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)
        # All values should be in [0, 1]
        assert (probs >= 0).all()
        assert (probs <= 1).all()


# ── Parameter Count Tests ────────────────────────────────────────────────────


class TestParameterCount:
    """Tests for model parameter counting."""

    def test_model_has_parameters(self, model):
        """Model should have a reasonable number of parameters."""
        total = model.get_num_parameters()
        assert total > 1_000_000, f"Too few parameters: {total}"

    def test_trainable_parameters(self, model):
        """All parameters should be trainable by default."""
        total = model.get_num_parameters()
        trainable = model.get_num_parameters(trainable_only=True)
        assert trainable == total, "Some parameters are unexpectedly frozen"

    def test_model_info(self, model):
        """get_model_info should return complete info dict."""
        info = get_model_info(model)
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "frozen_parameters" in info
        assert "model_size_mb" in info
        assert info["model_size_mb"] > 0


# ── Save/Load Tests ─────────────────────────────────────────────────────────


class TestSaveLoad:
    """Tests for model save/load roundtrip."""

    def test_save_load_roundtrip(self, model, single_input):
        """Saved and loaded model should produce the same predictions."""
        model.eval()
        with torch.no_grad():
            original_output = model(single_input)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model.pth")
            save_model(model, path, epoch=1, metrics={"val_f1": 0.95})

            # Load
            loaded_model = load_model(path, num_classes=2, device="cpu")
            loaded_model.eval()

            with torch.no_grad():
                loaded_output = loaded_model(single_input)

        assert torch.allclose(
            original_output, loaded_output, atol=1e-5
        ), "Loaded model produces different output"

    def test_save_creates_file(self, model):
        """save_model should create the checkpoint file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "model.pth")
            save_model(model, path)
            assert os.path.exists(path)


# ── Backbone Freeze Tests ───────────────────────────────────────────────────


class TestBackboneFreeze:
    """Tests for backbone freezing/unfreezing."""

    def test_freeze_reduces_trainable_params(self):
        """Freezing backbone should reduce trainable parameters."""
        model_unfrozen = build_densenet121(num_classes=2, pretrained=False, freeze_backbone=False)
        model_frozen = build_densenet121(num_classes=2, pretrained=False, freeze_backbone=True)

        unfrozen_count = model_unfrozen.get_num_parameters(trainable_only=True)
        frozen_count = model_frozen.get_num_parameters(trainable_only=True)

        assert frozen_count < unfrozen_count, "Frozen model should have fewer trainable params"

    def test_unfreeze_restores_params(self):
        """Unfreezing should restore all parameters to trainable."""
        model = build_densenet121(num_classes=2, pretrained=False, freeze_backbone=True)
        frozen_count = model.get_num_parameters(trainable_only=True)

        model.unfreeze_backbone()
        unfrozen_count = model.get_num_parameters(trainable_only=True)

        assert unfrozen_count > frozen_count


# ── GradCAM Tests ────────────────────────────────────────────────────────────


class TestGradCAM:
    """Tests for GradCAM visualization."""

    def test_gradcam_output_shape(self, model, single_input):
        """GradCAM should produce a (224, 224) heatmap."""
        heatmap = model.get_gradcam(single_input)
        assert heatmap.shape == (224, 224), f"Expected (224, 224), got {heatmap.shape}"

    def test_gradcam_value_range(self, model, single_input):
        """GradCAM values should be in [0, 1]."""
        heatmap = model.get_gradcam(single_input)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
