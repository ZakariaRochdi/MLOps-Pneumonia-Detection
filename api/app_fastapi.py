"""
FastAPI Inference Application — REST API for Pneumonia detection.

Endpoints:
  POST /predict       — Upload a chest X-ray image, returns prediction
  GET  /health        — Health check
  GET  /model-info    — Model metadata

The model is loaded once at startup and shared across requests.
"""

import io
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.data.dataset import CLASS_NAMES
from src.inference.predict import PneumoniaPredictor
from src.models.utils import get_device, get_model_info

# ── Global predictor instance ────────────────────────────────────────────────
predictor: PneumoniaPredictor | None = None

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pth")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global predictor
    print("Loading model...")
    try:
        predictor = PneumoniaPredictor(model_path=MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"WARNING: Failed to load model: {e}")
        predictor = None
    yield
    # Cleanup
    predictor = None
    print("Model unloaded")


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Chest X-ray Pneumonia Detection API",
    description=(
        "MLOps-powered REST API for detecting pneumonia in chest X-ray images "
        "using DenseNet121 deep learning model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Predict pneumonia from chest X-ray",
    description="Upload a chest X-ray image (JPEG/PNG) and receive a prediction.",
)
async def predict(
    file: UploadFile = File(..., description="Chest X-ray image file (JPEG/PNG)"),
):
    """
    Predict whether a chest X-ray shows pneumonia or is normal.

    Accepts JPEG and PNG image uploads.
    Returns predicted label, confidence score, and class probabilities.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later.",
        )

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected JPEG or PNG.",
        )

    try:
        # Read image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run inference
        start_time = time.time()
        result = predictor.predict(image)
        time.time() - start_time

        return PredictionResponse(
            label=result["label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e!s}",
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service is running and the model is loaded.",
)
async def health():
    """Service health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        model_version=MODEL_VERSION,
        device=get_device() if predictor else "unknown",
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model information",
    description="Get metadata about the loaded model.",
)
async def model_info():
    """Return model metadata."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )

    info = get_model_info(predictor.model)

    return ModelInfoResponse(
        model_name="DenseNet121",
        num_classes=len(CLASS_NAMES),
        class_names=list(CLASS_NAMES),
        input_size="224x224",
        total_parameters=info["total_parameters"],
    )
