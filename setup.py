from setuptools import setup, find_packages

setup(
    name="chest-xray-pneumonia",
    version="1.0.0",
    description="MLOps pipeline for Chest X-ray Pneumonia Detection",
    author="MLOps Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "Pillow>=10.0.0",
        "mlflow>=2.8.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "gradio>=4.0.0",
        "python-multipart>=0.0.6",
        "dvc>=3.30.0",
        "pytest>=7.4.0",
        "evidently>=0.4.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0"
    ],
    python_requires=">=3.10",
)
