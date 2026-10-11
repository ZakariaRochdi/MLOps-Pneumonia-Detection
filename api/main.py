"""FastAPI inference service for pneumonia detection."""

import io
import os

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from src.models.densenet import DenseNet121

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pth")
IMAGE_SIZE = 224

app = FastAPI(title="Pneumonia Detection API", version="1.0.0")

# ---------------------------------------------------------------------------
# Model loading (optional — service starts even without model)
# ---------------------------------------------------------------------------

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    if os.path.exists(MODEL_PATH):
        model = DenseNet121(num_classes=2, pretrained=False)
        state = torch.load(MODEL_PATH, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: Model not found at {MODEL_PATH} - running in degraded mode")
except Exception as e:
    print(f"WARNING: Failed to load model: {e}")
    model = None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "device": str(device),
        "model_path": MODEL_PATH,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict pneumonia from a chest X-ray image."""

    # 1. VALIDATE FILE TYPE FIRST (before checking model)
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. "
                   f"Allowed: {sorted(ALLOWED_TYPES)}",
        )

    # 2. THEN check model availability
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )

    # 3. Read and validate image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {e}",
        )

    # 4. Preprocess
    tensor = preprocess(image).unsqueeze(0).to(device)

    # 5. Inference
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i].item())
            for i in range(len(CLASS_NAMES))
        },
    }


@app.get("/model-info")
async def model_info():
    """Return model metadata."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "architecture": "DenseNet121",
        "num_classes": 2,
        "class_names": CLASS_NAMES,
        "input_size": [IMAGE_SIZE, IMAGE_SIZE],
        "num_parameters": n_params,
        "device": str(device),
    }
