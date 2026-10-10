"""
API Schemas — Pydantic models for request/response validation.

Defines the data contracts for the FastAPI inference API.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response schema for the /predict endpoint."""

    label: str = Field(
        ...,
        description="Predicted class label: 'NORMAL' or 'PNEUMONIA'",
        examples=["PNEUMONIA"],
    )
    confidence: float = Field(
        ...,
        description="Confidence score for the predicted class (0-1)",
        ge=0.0,
        le=1.0,
        examples=[0.97],
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability for each class",
        examples=[{"NORMAL": 0.03, "PNEUMONIA": 0.97}],
    )


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: str = Field(
        ...,
        description="Service health status",
        examples=["healthy"],
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded and ready",
    )
    model_version: Optional[str] = Field(
        None,
        description="Version identifier of the loaded model",
    )
    device: str = Field(
        ...,
        description="Compute device in use (cpu/cuda/mps)",
        examples=["cuda"],
    )


class ModelInfoResponse(BaseModel):
    """Response schema for the /model-info endpoint."""

    model_name: str = Field(
        ...,
        description="Model architecture name",
        examples=["DenseNet121"],
    )
    num_classes: int = Field(
        ...,
        description="Number of output classes",
        examples=[2],
    )
    class_names: list = Field(
        ...,
        description="List of class names",
        examples=[["NORMAL", "PNEUMONIA"]],
    )
    input_size: str = Field(
        ...,
        description="Expected input image size",
        examples=["224x224"],
    )
    total_parameters: int = Field(
        ...,
        description="Total number of model parameters",
    )


class ErrorResponse(BaseModel):
    """Response schema for error cases."""

    error: str = Field(
        ...,
        description="Error message",
    )
    detail: Optional[str] = Field(
        None,
        description="Detailed error information",
    )
