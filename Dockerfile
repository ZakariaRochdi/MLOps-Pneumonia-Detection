# ==============================================================================
# Dockerfile — Dockerized Inference API for Chest X-ray Pneumonia Detection
#
# Multi-stage build:
#   Stage 1 (builder): Install dependencies
#   Stage 2 (runtime): Copy only what's needed for serving
# ==============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Install runtime system dependencies (OpenCV, image processing)
# NOTE: libgl1-mesa-glx was renamed to libgl1 in Debian Trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source code
COPY src/ ./src/
COPY api/ ./api/

# Create configs/ and models/ dirs first, then copy their content if present
RUN mkdir -p /app/configs /app/models

COPY configs/ ./configs/
COPY models/ ./models/

# params.yaml is optional too
COPY params.yaml* ./

# Environment variables
ENV MODEL_PATH=models/best_model.pth
ENV MODEL_VERSION=1.0.0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose API port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "api.app_fastapi:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]