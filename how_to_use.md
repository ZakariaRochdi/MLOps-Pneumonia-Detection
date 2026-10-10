# How to Use: MLOps Chest X-ray Pneumonia Detection

This guide provides step-by-step instructions on how to use every component of the MLOps pipeline, from setting up the environment to tracking experiments and deploying the model.

---

## 1. Initial Setup

Before running anything, you need to set up your Python environment.

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# 3. Install all required dependencies
pip install -r requirements.txt
```

---

## 2. Version Control: Git & DVC

This project uses **Git** to version control the code and **DVC** (Data Version Control) to version control the large datasets and model artifacts.

### Using Git
Git tracks your Python scripts, configuration files, and documentation.
```bash
git add .
git commit -m "Your commit message"
git push origin main
```

### Using DVC for Data
DVC manages the `chest_xray/` dataset, processed data, and trained models. The `.dvc` files and `dvc.lock` act as pointers to your data, and these pointers are committed to Git.

```bash
# 1. Initialize DVC (if not already done)
dvc init

# 2. Add remote storage (e.g., AWS S3 or Google Cloud Storage)
dvc remote add -d myremote s3://your-bucket-name/mlops-pneumonia

# 3. Pull the data from remote storage to your local machine
dvc pull

# 4. If you add new raw images, track them with DVC:
dvc add chest_xray
git add chest_xray.dvc
git commit -m "Added new raw images"
dvc push
```

---

## 3. Training and Evaluation

You have two ways to run the machine learning pipeline: manually step-by-step, or automatically using DVC.

### Method A: Fully Reproducible Pipeline (Recommended)
DVC orchestrates the entire pipeline (`prepare_data` → `train` → `evaluate`) as defined in `dvc.yaml`. It automatically detects if data or code has changed and only reruns what is necessary.

```bash
dvc repro
```

### Method B: Manual Execution
If you prefer to run the scripts manually for debugging:

```bash
# 1. Prepare and preprocess data (copies raw data to data/processed/)
python -m src.data.dataset --prepare

# 2. Train the DenseNet121 model
python -m src.training.train

# 3. Evaluate the model on the test set
python -m src.training.evaluate
```
*Note: You can tweak hyperparameters like batch size, learning rate, and epochs directly in `params.yaml` or `configs/training_config.yaml`.*

---

## 4. Experiment Tracking (MLflow & W&B)

The project automatically logs parameters, metrics (Accuracy, F1-Score, Loss), and model artifacts to tracking servers during training.

### Checking MLflow (Local Tracking)
MLflow runs a local server to visualize all your training runs.

1. Open a new terminal and activate your environment.
2. Start the MLflow UI:
   ```bash
   mlflow server --host 127.0.0.1 --port 5000
   ```
3. Open your browser and go to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**
4. Here you can compare different runs, view the training loss curves, and download the `best_model.pth` artifact.

### Checking Weights & Biases (Cloud Tracking)
W&B provides advanced cloud-based dashboards.
1. Create an account at [wandb.ai](https://wandb.ai) and run `wandb login` in your terminal.
2. Open `configs/training_config.yaml` and set `enabled: true` under `wandb`.
3. Run training (`python -m src.training.train`). A link to your live cloud dashboard will appear in the terminal.

---

## 5. Deployment: Docker, FastAPI, and Gradio

Once you have a trained model (`models/best_model.pth`), you can deploy the inference APIs using Docker.

### Running with Docker Compose
This command will build the containers and launch three services simultaneously:
```bash
docker-compose up --build -d
```

### Accessing the Services:

1. **Gradio Web Interface (Interactive UI)**
   - **URL**: [http://localhost:7860](http://localhost:7860)
   - **Usage**: Open this in your browser. You can drag and drop a chest X-ray image here to get a prediction (Normal/Pneumonia) along with a GradCAM heatmap visualization.

2. **FastAPI (REST API)**
   - **URL**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Usage**: This provides a Swagger UI where you can test the API endpoints programmatically.
   - **Testing via cURL**:
     ```bash
     curl -X POST "http://localhost:8000/predict" \
       -H "accept: application/json" \
       -H "Content-Type: multipart/form-data" \
       -F "file=@test_xray_image.jpeg"
     ```

3. **MLflow inside Docker**
   - **URL**: [http://localhost:5000](http://localhost:5000)

To stop the services, run:
```bash
docker-compose down
```

---

## 6. Testing and CI/CD

Before pushing your code, you should ensure all unit tests and validation gates pass.

```bash
# Run the entire test suite (Preprocessing, Model, Inference)
pytest tests/ -v

# Run ONLY the CI Validation Gate (checks if F1-Score > 0.95)
pytest tests/test_validation.py -v
```

When you push code to GitHub or GitLab, the CI/CD pipelines (defined in `.github/workflows/ci_cd.yml` and `.gitlab-ci.yml`) will automatically run these tests. If the model performance drops below the threshold, the deployment will be blocked automatically.

---

## 7. Data Drift Monitoring

In production, input data can change over time. We use **Evidently AI** to monitor this.

1. **Generate the Baseline**: After training, generate reference statistics for what "normal" data looks like:
   ```python
   from src.monitoring.drift import DataDriftDetector
   DataDriftDetector.compute_reference_stats("data/processed/train")
   ```
2. **Monitor**: The drift detector logic in `src/monitoring/drift.py` tracks the pixel intensity distribution and prediction ratios of incoming API requests. If a significant shift is detected, it will flag a drift alert.
