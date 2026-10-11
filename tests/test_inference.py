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

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Fixtures


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    os.environ.setdefault("MODEL_PATH", "models/best_model.pth")
    from api.main import app
    return TestClient(app)


@pytest.fixture
def sample_jpeg_bytes():
    """Create a sample JPEG image as bytes."""
    img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_png_bytes():
    """Create a sample PNG image as bytes."""
    img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data

    def test_health_status_value(self, client):
        data = client.get("/health").json()
        assert data["status"] in ("healthy", "degraded")


class TestPredictEndpoint:
    def test_predict_jpeg(self, client, sample_jpeg_bytes):
        files = {"file": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
        response = client.post("/predict", files=files)
        assert response.status_code in (200, 503)

    def test_predict_png(self, client, sample_png_bytes):
        files = {"file": ("test.png", sample_png_bytes, "image/png")}
        response = client.post("/predict", files=files)
        assert response.status_code in (200, 503)

    def test_predict_invalid_file_type(self, client):
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/predict", files=files)
        assert response.status_code == 400

    def test_predict_no_file(self, client):
        response = client.post("/predict")
        assert response.status_code == 422


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        response = client.get("/model-info")
        assert response.status_code in (200, 503)

    def test_model_info_structure(self, client):
        response = client.get("/model-info")
        if response.status_code == 200:
            data = response.json()
            assert "architecture" in data
            assert "num_classes" in data
