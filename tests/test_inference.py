"""
Test Inference — Integration tests for the FastAPI inference API.

Validates:
  - /health endpoint returns 200
  - /predict endpoint handles valid images
  - /predict endpoint rejects invalid inputs
  - /model-info endpoint returns model metadata
"""

import io
import os
import pytest
import numpy as np
from PIL import Image

from fastapi.testclient import TestClient


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    # Set model path to a test path (may not exist in CI)
    os.environ.setdefault("MODEL_PATH", "models/best_model.pth")

    from api.app_fastapi import app

    return TestClient(app)


@pytest.fixture
def sample_jpeg_bytes():
    """Create a sample JPEG image as bytes."""
    img = Image.fromarray(
        np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    )
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_png_bytes():
    """Create a sample PNG image as bytes."""
    img = Image.fromarray(
        np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    )
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


# ── Health Endpoint Tests ────────────────────────────────────────────────────


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200 status code."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health response should have required fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data

    def test_health_status_value(self, client):
        """Health status should be 'healthy' or 'degraded'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] in ("healthy", "degraded")


# ── Predict Endpoint Tests ───────────────────────────────────────────────────


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    @pytest.mark.skipif(
        not os.path.exists("models/best_model.pth"),
        reason="Model file not available",
    )
    def test_predict_jpeg(self, client, sample_jpeg_bytes):
        """Predict endpoint should accept JPEG images."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpeg", sample_jpeg_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert data["label"] in ("NORMAL", "PNEUMONIA")
        assert 0 <= data["confidence"] <= 1

    @pytest.mark.skipif(
        not os.path.exists("models/best_model.pth"),
        reason="Model file not available",
    )
    def test_predict_png(self, client, sample_png_bytes):
        """Predict endpoint should accept PNG images."""
        response = client.post(
            "/predict",
            files={"file": ("test.png", sample_png_bytes, "image/png")},
        )
        assert response.status_code == 200

    def test_predict_invalid_file_type(self, client):
        """Predict endpoint should reject non-image files."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_predict_no_file(self, client):
        """Predict endpoint should error when no file is provided."""
        response = client.post("/predict")
        assert response.status_code == 422  # Unprocessable Entity


# ── Model Info Endpoint Tests ────────────────────────────────────────────────


class TestModelInfoEndpoint:
    """Tests for the /model-info endpoint."""

    @pytest.mark.skipif(
        not os.path.exists("models/best_model.pth"),
        reason="Model file not available",
    )
    def test_model_info_returns_200(self, client):
        """Model info endpoint should return 200 when model is loaded."""
        response = client.get("/model-info")
        assert response.status_code == 200

    @pytest.mark.skipif(
        not os.path.exists("models/best_model.pth"),
        reason="Model file not available",
    )
    def test_model_info_structure(self, client):
        """Model info should contain expected fields."""
        response = client.get("/model-info")
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "num_classes" in data
            assert "class_names" in data
            assert "input_size" in data
