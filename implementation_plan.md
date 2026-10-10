# MLOps Workflow for Chest X-ray Pneumonia Detection — Implementation Plan

![Architecture Diagram](file:///c:/Users/SALMA/Downloads/MLOPS PFA/Architecture.png)

## Overview

This plan implements the complete **MLOps pipeline** shown in the architecture diagram above, covering all 5 stages for a **Chest X-ray Pneumonia classification** system using **DenseNet121**. The project takes raw chest X-ray images (Normal vs. Pneumonia), versions the data, preprocesses and augments it, trains a deep learning model with experiment tracking, validates via CI, and deploys as a Dockerized inference API with monitoring.

### Dataset Summary

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341 | 3,875 | 5,216 |
| Test | 234 | 390 | 624 |
| Val | 8 | 8 | 16 |

> [!IMPORTANT]
> The dataset is **imbalanced** (≈74% Pneumonia in training). The plan addresses this with weighted loss functions and stratified sampling. The validation set is also very small (16 images) — we will use the test set for robust evaluation and optionally merge val into training with cross-validation.

---

## Technology Stack (from Architecture Diagram)

| Category | Technologies |
|---|---|
| **Data Versioning** | DVC (Data Version Control), S3 or GCS cloud storage |
| **Preprocessing & Augmentation** | PyTorch `torchvision.transforms` (resize 224×224, normalization, rotation, flip, resize augmentation) |
| **Deep Learning Framework** | PyTorch / TensorFlow (plan uses **PyTorch** as primary) |
| **Model Architecture** | **DenseNet121** (pretrained, transfer learning) |
| **Experiment Tracking & Model Registry** | **MLflow** and/or **Weights & Biases (W&B)** |
| **CI/CD** | **GitHub Actions** + **GitLab CI** |
| **Unit Testing** | **pytest** (preprocessing tests, F1-Score > 0.95 validation gate) |
| **Inference API** | **FastAPI** and/or **Gradio** |
| **Containerization** | **Docker**, Container Registry (Docker Hub / GCR / ECR) |
| **Deployment** | Staging → Production pipeline (cloud deployment) |
| **Monitoring** | Data drift detection (e.g., **Evidently AI** or custom) |
| **Version Control** | **Git** + **GitHub/GitLab** |

---

## Project Structure

```
MLOPS PFA/
├── chest_xray/                    # Raw dataset (tracked by DVC)
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── test/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── val/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── src/                           # Core source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # PyTorch Dataset & DataLoader classes
│   │   ├── preprocessing.py       # Image transformations (resize, normalize)
│   │   └── augmentation.py        # Data augmentation (rotation, flip, resize)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── densenet.py            # DenseNet121 model definition & transfer learning
│   │   └── utils.py               # Model utilities (save, load, export)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py               # Training loop with MLflow/W&B logging
│   │   ├── evaluate.py            # Evaluation metrics (accuracy, F1, confusion matrix)
│   │   └── config.py              # Hyperparameter configuration
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predict.py             # Single-image prediction logic
│   └── monitoring/
│       ├── __init__.py
│       └── drift.py               # Data drift detection logic
│
├── api/                           # Inference API
│   ├── app_fastapi.py             # FastAPI application
│   ├── app_gradio.py              # Gradio application (alternative UI)
│   └── schemas.py                 # Request/Response Pydantic models
│
├── tests/                         # Unit & integration tests
│   ├── __init__.py
│   ├── test_preprocessing.py      # Preprocessing pipeline tests
│   ├── test_model.py              # Model architecture tests
│   ├── test_inference.py          # Inference endpoint tests
│   └── test_validation.py         # F1-Score > 0.95 validation gate
│
├── configs/                       # Configuration files
│   ├── training_config.yaml       # Training hyperparameters
│   └── dvc.yaml                   # DVC pipeline definition
│
├── notebooks/                     # Exploration notebooks
│   └── eda.ipynb                  # Exploratory Data Analysis
│
├── .dvc/                          # DVC internal config
├── .github/
│   └── workflows/
│       └── ci_cd.yml              # GitHub Actions CI/CD pipeline
├── .gitlab-ci.yml                 # GitLab CI/CD pipeline (alternative)
│
├── Dockerfile                     # Docker image for inference API
├── docker-compose.yml             # Multi-container setup (API + monitoring)
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── params.yaml                    # DVC parameters
├── dvc.yaml                       # DVC pipeline stages
├── dvc.lock                       # DVC lock file (auto-generated)
├── .dvcignore                     # DVC ignore file
├── .gitignore                     # Git ignore file
└── README.md                      # Project documentation
```

---

## Proposed Changes — Stage by Stage

---

### Stage 1 — Data Versioning (DVC + S3/GCS)

This stage sets up **DVC** to version-control the raw chest X-ray dataset and store it on remote cloud storage (S3 or GCS), keeping the Git repo lightweight while ensuring full data reproducibility.

#### [NEW] [.dvcignore](file:///c:/Users/SALMA/Downloads/MLOPS PFA/.dvcignore)
- Standard DVC ignore patterns (e.g., IDE files, `__pycache__`)

#### [NEW] [.gitignore](file:///c:/Users/SALMA/Downloads/MLOPS PFA/.gitignore)
- Ignore `/chest_xray` data directory (tracked by DVC instead)
- Ignore model artifacts, `__pycache__`, `.env`, virtual environments
- Ignore MLflow/W&B local artifacts

#### [NEW] [dvc.yaml](file:///c:/Users/SALMA/Downloads/MLOPS PFA/dvc.yaml)
- **DVC pipeline definition** with stages:
  - `prepare_data`: depends on raw data, outputs processed data
  - `train`: depends on processed data + config, outputs model artifacts
  - `evaluate`: depends on trained model, outputs metrics JSON
- Each stage specifies `cmd`, `deps`, `params`, `outs`, and `metrics`

#### [NEW] [params.yaml](file:///c:/Users/SALMA/Downloads/MLOPS PFA/params.yaml)
- Central parameters file referenced by DVC:
  ```yaml
  data:
    raw_dir: chest_xray
    processed_dir: data/processed
    image_size: 224
    batch_size: 32
  
  train:
    epochs: 25
    learning_rate: 0.0001
    weight_decay: 0.0001
    model_name: densenet121
    pretrained: true
    num_classes: 2
    class_weights: [2.88, 1.0]  # Compensate for imbalance (3875/1341)
  
  augmentation:
    rotation_range: 15
    horizontal_flip: true
    vertical_flip: false
    zoom_range: 0.1
  ```

#### Setup commands:
```bash
# Initialize DVC
dvc init

# Track dataset with DVC
dvc add chest_xray

# Configure remote storage (choose S3 or GCS)
dvc remote add -d myremote s3://your-bucket/mlops-pneumonia
# OR
dvc remote add -d myremote gs://your-bucket/mlops-pneumonia

# Push data to remote
dvc push
```

---

### Stage 2 — Automated Preprocessing

Image transformations including resize to 224×224, normalization (ImageNet stats), and data augmentation (rotation, flip, resize) for the Pneumonia class to handle class imbalance.

#### [NEW] [src/\_\_init\_\_.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/src/__init__.py)
- Empty init file for the `src` package

#### [NEW] [src/data/\_\_init\_\_.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/src/data/__init__.py)
- Empty init file for the `data` subpackage

#### [NEW] [src/data/preprocessing.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/src/data/preprocessing.py)
- **Image transformation pipelines** using `torchvision.transforms`:
  - **Train transforms**: `Resize(256)` → `CenterCrop(224)` → `RandomHorizontalFlip` → `RandomRotation(15)` → `RandomResizedCrop(224)` → `ToTensor` → `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`
  - **Validation/Test transforms**: `Resize(256)` → `CenterCrop(224)` → `ToTensor` → `Normalize(mean, std)`
- Handles grayscale-to-RGB conversion (chest X-rays are often single-channel)

#### [NEW] [src/data/augmentation.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/src/data/augmentation.py)
- **Extended augmentation** specifically for the Pneumonia detection task:
  - `RandomAffine(degrees=15, translate=(0.05, 0.05))`
  - `ColorJitter(brightness=0.2, contrast=0.2)`
  - `GaussianBlur(kernel_size=3)`
- Configurable via `params.yaml`

#### [NEW] [src/data/dataset.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/src/data/dataset.py)
- **`ChestXrayDataset(Dataset)`**: Custom PyTorch Dataset class
  - Reads images from the `chest_xray/` directory structure
  - Applies transforms from `preprocessing.py`
  - Returns `(image_tensor, label)` where label: 0=NORMAL, 1=PNEUMONIA
- **`get_dataloaders()`**: Factory function returning train/val/test `DataLoader` objects
  - Uses `WeightedRandomSampler` for train loader to address class imbalance
  - Configurable `batch_size`, `num_workers` from `params.yaml`

---

### Stage 3 — Model Development & Training

DenseNet121 with transfer learning, experiment tracking via MLflow and W&B, and a model registry for versioning trained models.

#### [NEW] [src/models/\_\_init\_\_.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/src/models/__init__.py)
- Empty init file

#### [NEW] [src/models/densenet.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/src/models/densenet.py)
- **`build_densenet121(num_classes=2, pretrained=True)`**:
  - Loads `torchvision.models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)`
  - Freezes early layers for transfer learning
  - Replaces the classifier head: `nn.Linear(1024, num_classes)`
  - Optionally unfreezes later DenseBlocks for fine-tuning
- **`DenseNet121Classifier(nn.Module)`**: Wrapper class with:
  - Forward method returning logits
  - `predict_proba()` method returning softmax probabilities
  - GradCAM-compatible hooks for interpretability

#### [NEW] [src/models/utils.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/src/models/utils.py)
- `save_model(model, path)` — saves state dict + architecture info
- `load_model(path)` — loads model for inference
- `export_to_onnx(model, path)` — ONNX export for production

#### [NEW] [src/training/\_\_init\_\_.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/src/training/__init__.py)
- Empty init file

#### [NEW] [src/training/config.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/src/training/config.py)
- **`TrainingConfig` dataclass** loading from `params.yaml`:
  - `epochs`, `learning_rate`, `weight_decay`, `batch_size`
  - `model_name`, `pretrained`, `num_classes`
  - `class_weights` for `nn.CrossEntropyLoss(weight=...)`
  - Device selection (CUDA/MPS/CPU auto-detection)

#### [NEW] [src/training/train.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/src/training/train.py)
- **Main training script** with full MLOps integration:
  - **Optimizer**: `Adam` with configurable LR and weight decay
  - **Scheduler**: `ReduceLROnPlateau` monitoring validation loss
  - **Loss**: `CrossEntropyLoss` with class weights
  - **MLflow integration**:
    - `mlflow.start_run()` context manager
    - Log params: all training config values
    - Log metrics per epoch: `train_loss`, `val_loss`, `train_acc`, `val_acc`, `val_f1`
    - Log artifacts: model checkpoint, confusion matrix plot, ROC curve
    - `mlflow.pytorch.log_model()` for model registry
  - **W&B integration** (parallel):
    - `wandb.init(project="chest-xray-pneumonia")`
    - `wandb.log()` for all metrics
    - `wandb.watch(model)` for gradient tracking
    - Log image predictions as `wandb.Image` tables
  - **Early stopping** with configurable patience
  - **Checkpointing**: Save best model based on val F1-Score

#### [NEW] [src/training/evaluate.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/src/training/evaluate.py)
- **`evaluate(model, dataloader)`**: Computes and returns:
  - **Accuracy** (overall)
  - **F1-Score** (weighted and per-class)
  - **Precision & Recall** per class
  - **Confusion Matrix** (plotted with `seaborn`)
  - **ROC Curve & AUC** (plotted with `matplotlib`)
  - **Classification Report** (sklearn)
- Outputs metrics as JSON for DVC metrics tracking
- Generates comparison plots (V1 vs V2 as shown in diagram)

#### [NEW] [configs/training_config.yaml](file:///c:/Users/SALMA/Downloads/MLOPS PFA/configs/training_config.yaml)
- YAML config for training hyperparameters (mirrors `params.yaml` but more detailed)
- MLflow tracking URI, W&B project name
- Experiment name and tags

---

### Stage 4 — Continuous Integration & Testing (CI)

Automated testing pipeline using GitHub Actions (and/or GitLab CI) with unit tests for preprocessing and a model validation gate requiring F1-Score > 0.95.

#### [NEW] [tests/\_\_init\_\_.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/tests/__init__.py)
- Empty init file

#### [NEW] [tests/test_preprocessing.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/tests/test_preprocessing.py)
- **Unit tests for the preprocessing pipeline** (pytest):
  - `test_image_resize()` — output is 224×224
  - `test_normalization_range()` — pixel values in expected range after normalization
  - `test_grayscale_to_rgb()` — single-channel images correctly converted to 3 channels
  - `test_augmentation_output_shape()` — augmented images maintain correct dimensions
  - `test_dataloader_batch_shape()` — batches have shape `(B, 3, 224, 224)`
  - `test_label_encoding()` — NORMAL=0, PNEUMONIA=1

#### [NEW] [tests/test_model.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/tests/test_model.py)
- **Unit tests for model architecture**:
  - `test_model_output_shape()` — output shape is `(batch_size, 2)`
  - `test_model_forward_pass()` — no errors on a random input tensor
  - `test_model_num_parameters()` — parameter count matches expected DenseNet121
  - `test_model_save_load()` — save/load roundtrip produces same predictions

#### [NEW] [tests/test_inference.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/tests/test_inference.py)
- **Integration tests for the inference API**:
  - `test_health_endpoint()` — `/health` returns 200
  - `test_predict_endpoint_normal()` — correct prediction on a NORMAL image
  - `test_predict_endpoint_pneumonia()` — correct prediction on a PNEUMONIA image
  - `test_invalid_input()` — returns appropriate error for non-image input

#### [NEW] [tests/test_validation.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/tests/test_validation.py)
- **Model validation gate**:
  - `test_f1_score_threshold()` — asserts F1-Score > 0.95 on the test set
  - `test_accuracy_threshold()` — asserts accuracy > 0.90
  - Loads the trained model and evaluates on the holdout test set
  - **This test acts as the CI gate** — pipeline fails if metrics don't meet threshold

#### [NEW] [.github/workflows/ci_cd.yml](file:///c:/Users/SALMA/Downloads/MLOPS PFA/.github/workflows/ci_cd.yml)
- **GitHub Actions CI/CD pipeline**:
  ```yaml
  name: MLOps CI/CD Pipeline
  
  on:
    push:
      branches: [main, develop]
    pull_request:
      branches: [main]
  
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - name: Set up Python 3.10
          uses: actions/setup-python@v5
          with:
            python-version: "3.10"
        - name: Install dependencies
          run: pip install -r requirements.txt
        - name: Run unit tests (preprocessing)
          run: pytest tests/test_preprocessing.py tests/test_model.py -v
        - name: Pull data with DVC
          run: dvc pull
        - name: Run model validation gate
          run: pytest tests/test_validation.py -v
    
    build-and-push:
      needs: test
      if: github.ref == 'refs/heads/main'
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - name: Build Docker image
          run: docker build -t chest-xray-api:${{ github.sha }} .
        - name: Push to Container Registry
          run: |
            docker tag chest-xray-api:${{ github.sha }} <registry>/chest-xray-api:latest
            docker push <registry>/chest-xray-api:latest
    
    deploy-staging:
      needs: build-and-push
      runs-on: ubuntu-latest
      steps:
        - name: Deploy to Staging
          run: echo "Deploy to staging environment"
    
    deploy-production:
      needs: deploy-staging
      runs-on: ubuntu-latest
      environment: production  # Requires manual approval
      steps:
        - name: Deploy to Production
          run: echo "Deploy to production environment"
  ```

#### [NEW] [.gitlab-ci.yml](file:///c:/Users/SALMA/Downloads/MLOPS PFA/.gitlab-ci.yml)
- **GitLab CI alternative pipeline** (equivalent stages):
  - `test` stage: run pytest suite
  - `validate` stage: model performance gate (F1 > 0.95)
  - `build` stage: Docker build + push to Container Registry
  - `deploy_staging` stage: deploy to staging
  - `deploy_production` stage: manual trigger for production deploy

---

### Stage 5 — Continuous Deployment (CD)

Dockerized inference API using FastAPI and Gradio, pushed to a container registry, deployed through staging → production, with data drift monitoring.

#### [NEW] [api/schemas.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/api/schemas.py)
- **Pydantic models** for API request/response:
  - `PredictionResponse`: `label` (str), `confidence` (float), `probabilities` (dict with NORMAL/PNEUMONIA scores)
  - `HealthResponse`: `status` (str), `model_version` (str)

#### [NEW] [api/app_fastapi.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/api/app_fastapi.py)
- **FastAPI inference application**:
  - `POST /predict` — accepts uploaded chest X-ray image, returns prediction
    - Loads image → applies preprocessing → runs model inference → returns `PredictionResponse`
  - `GET /health` — health check endpoint
  - `GET /model-info` — returns model metadata (version, architecture, metrics)
  - CORS middleware enabled
  - Model loaded once at startup via `@app.on_event("startup")`
  - Supports both file upload and base64-encoded image input

#### [NEW] [api/app_gradio.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/api/app_gradio.py)
- **Gradio web interface** (alternative/complementary UI):
  - Image upload component → prediction display
  - Shows: predicted label, confidence percentage, probability bar chart
  - Displays GradCAM visualization overlay on the X-ray
  - Example images included for quick testing
  - Mobile-friendly interface (matches the phone mockup in the diagram)

#### [NEW] [src/inference/\_\_init\_\_.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/src/inference/__init__.py)
- Empty init file

#### [NEW] [src/inference/predict.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/src/inference/predict.py)
- **`PneumoniaPredictor` class**:
  - `__init__(model_path)` — loads model, sets device
  - `preprocess(image)` — applies inference transforms
  - `predict(image) → dict` — returns `{"label": "PNEUMONIA", "confidence": 0.97, "probabilities": {"NORMAL": 0.03, "PNEUMONIA": 0.97}}`
  - `predict_batch(images)` — batch inference support

#### [NEW] [src/monitoring/\_\_init\_\_.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/src/monitoring/__init__.py)
- Empty init file

#### [NEW] [src/monitoring/drift.py](file:///c:/Users/SALMA/Downloads/MLOPS PFA/src/monitoring/drift.py)
- **Data drift detection** using Evidently AI:
  - `DataDriftDetector` class:
    - Computes image statistics (mean, std, histogram) for reference dataset
    - Compares incoming inference data against reference distribution
    - Detects drift in pixel intensity distributions, image dimensions
  - `generate_drift_report()` — creates an Evidently HTML report
  - **Monitoring metrics** logged:
    - Prediction distribution over time (NORMAL vs PNEUMONIA ratio)
    - Input image quality metrics
    - Model confidence distribution
  - Alerts when drift exceeds configurable threshold

#### [NEW] [Dockerfile](file:///c:/Users/SALMA/Downloads/MLOPS PFA/Dockerfile)
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "api.app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### [NEW] [docker-compose.yml](file:///c:/Users/SALMA/Downloads/MLOPS PFA/docker-compose.yml)
- Multi-container setup:
  - **`api`** service: FastAPI inference API (port 8000)
  - **`gradio`** service: Gradio UI (port 7860)
  - **`mlflow`** service: MLflow tracking server (port 5000)
  - Shared volume for model artifacts
  - Environment variables for configuration

#### [NEW] [requirements.txt](file:///c:/Users/SALMA/Downloads/MLOPS PFA/requirements.txt)
```
# Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# Data & ML
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
Pillow>=10.0.0

# Experiment Tracking
mlflow>=2.8.0
wandb>=0.16.0

# API & Serving
fastapi>=0.104.0
uvicorn>=0.24.0
gradio>=4.0.0
python-multipart>=0.0.6

# Data Versioning
dvc>=3.30.0
dvc-s3>=3.0.0    # OR dvc-gs for GCS

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Monitoring
evidently>=0.4.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0

# Config
pyyaml>=6.0.0

# Utilities
tqdm>=4.66.0
```

#### [NEW] [README.md](file:///c:/Users/SALMA/Downloads/MLOPS PFA/README.md)
- Comprehensive project documentation:
  - Architecture overview with embedded diagram
  - Setup instructions (prerequisites, installation, DVC setup)
  - How to run training, evaluation, and inference
  - API documentation with example curl commands
  - Docker usage guide
  - CI/CD pipeline explanation
  - Monitoring dashboard setup

---

## User Review Required

> [!IMPORTANT]
> **Cloud Storage Choice**: The architecture shows **S3/GCS** for DVC remote storage. Which cloud provider do you prefer?
> - **AWS S3** — requires `dvc-s3` and AWS credentials
> - **Google Cloud Storage (GCS)** — requires `dvc-gs` and GCP credentials
> - **Local/SSH remote** — for development without cloud costs

> [!IMPORTANT]
> **Deep Learning Framework**: The diagram shows **PyTorch/TensorFlow**. This plan uses **PyTorch** as the primary framework. Should we also include a TensorFlow implementation, or is PyTorch alone sufficient?

> [!IMPORTANT]
> **Inference API**: The diagram shows both **Gradio** and **FastAPI**. The plan includes both — FastAPI for production REST API and Gradio for a user-friendly demo UI. Should we keep both, or focus on just one?

> [!WARNING]
> **CI/CD Platform**: The diagram references both **GitHub** and **GitLab** logos. The plan includes configs for both (`.github/workflows/ci_cd.yml` and `.gitlab-ci.yml`). Which platform is your primary choice, or should we keep both?

---

## Open Questions

1. **Container Registry**: Which container registry should we use? Options: Docker Hub, AWS ECR, Google GCR, GitHub Container Registry (GHCR), or GitLab Container Registry.

2. **Deployment Target**: Where will the production deployment live? Options: AWS (EC2/ECS/EKS), GCP (Cloud Run/GKE), Azure, or a self-hosted server.

3. **MLflow vs W&B**: Should we use both experiment trackers simultaneously, or focus on one? MLflow is open-source/self-hosted while W&B is cloud-hosted with a richer UI.

4. **Validation Set**: The current validation set has only 16 images. Should we:
   - (A) Merge val into training and use k-fold cross-validation
   - (B) Split a portion of the training set as validation
   - (C) Keep the current split as-is

5. **GradCAM Visualization**: The diagram's mobile UI shows a heatmap overlay. Should we include GradCAM-based interpretability in the inference API, or defer this to a later phase?

---

## Verification Plan

### Automated Tests
```bash
# Run all unit tests
pytest tests/ -v --cov=src --cov-report=html

# Run preprocessing tests only
pytest tests/test_preprocessing.py -v

# Run model validation gate
pytest tests/test_validation.py -v

# Run the full DVC pipeline
dvc repro
```

### Model Performance Validation
- **F1-Score > 0.95** on the test set (CI gate requirement from diagram)
- **Accuracy tracking** across model versions (V1 vs V2 comparison)
- **Confusion matrix analysis** for false positive/negative rates

### API Validation
```bash
# Build and run Docker container
docker build -t chest-xray-api .
docker run -p 8000:8000 chest-xray-api

# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint
curl -X POST http://localhost:8000/predict -F "file=@test_xray.jpeg"
```

### End-to-End Pipeline
1. `dvc repro` — runs the full pipeline (data → preprocess → train → evaluate)
2. Check MLflow UI at `http://localhost:5000` for logged experiments
3. Verify Docker container builds and serves predictions correctly
4. Validate CI/CD pipeline triggers on push to GitHub/GitLab

### Manual Verification
- Review experiment tracking dashboards (MLflow / W&B)
- Test Gradio UI with sample chest X-ray images
- Verify data drift monitoring reports
- Confirm staging → production deployment flow

---

## Implementation Order

| Phase | Stage | Description | Estimated Effort |
|-------|-------|-------------|-----------------|
| 1 | Setup | Project structure, `requirements.txt`, `.gitignore`, `README.md` | ~1 hour |
| 2 | Stage 1 | DVC initialization, data versioning, remote storage setup | ~1 hour |
| 3 | Stage 2 | Preprocessing pipeline, augmentation, Dataset/DataLoader | ~2 hours |
| 4 | Stage 3 | DenseNet121 model, training loop, MLflow/W&B integration | ~3 hours |
| 5 | Stage 3 | Evaluation metrics, model registry, experiment comparison | ~2 hours |
| 6 | Stage 5 | FastAPI + Gradio inference API, prediction logic | ~2 hours |
| 7 | Stage 4 | Unit tests, validation gate, CI/CD pipelines | ~2 hours |
| 8 | Stage 5 | Dockerfile, docker-compose, container registry setup | ~1 hour |
| 9 | Stage 5 | Data drift monitoring, Evidently integration | ~2 hours |
| 10 | Polish | Documentation, end-to-end testing, final verification | ~2 hours |
