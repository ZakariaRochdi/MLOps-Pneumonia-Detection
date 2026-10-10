"""
Data Drift Monitoring — Detect distribution shifts in incoming inference data.

Uses Evidently AI to generate drift reports comparing incoming data against
a reference dataset (training distribution). Also tracks prediction distribution
and model confidence over time.

Monitored signals:
  - Pixel intensity distribution shift
  - Prediction class ratio changes
  - Model confidence distribution
  - Input image quality metrics
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

import numpy as np
from PIL import Image

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently import ColumnMapping
    import pandas as pd

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("WARNING: Evidently not installed. Drift monitoring disabled.")
    print("  Install with: pip install evidently")


class DataDriftDetector:
    """
    Monitors data drift by comparing inference inputs against a reference
    dataset (training data distribution).

    Tracks:
      - Pixel statistics (mean, std, histogram) per image
      - Prediction distribution (NORMAL vs PNEUMONIA ratio)
      - Model confidence distribution
      - Alerts when drift exceeds threshold

    Args:
        reference_stats_path: Path to precomputed reference statistics JSON.
        drift_threshold: P-value threshold below which drift is flagged.
        window_size: Number of recent predictions to keep in memory.
        reports_dir: Directory to save drift reports.
    """

    def __init__(
        self,
        reference_stats_path: str = "reports/reference_stats.json",
        drift_threshold: float = 0.05,
        window_size: int = 500,
        reports_dir: str = "reports/monitoring",
    ):
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.reports_dir = reports_dir

        # Rolling window of recent predictions
        self.recent_predictions: deque = deque(maxlen=window_size)
        self.recent_image_stats: deque = deque(maxlen=window_size)
        self.recent_confidences: deque = deque(maxlen=window_size)

        # Load reference statistics
        self.reference_stats = None
        if os.path.exists(reference_stats_path):
            with open(reference_stats_path, "r") as f:
                self.reference_stats = json.load(f)
            print(f"Reference stats loaded from {reference_stats_path}")

    def compute_image_stats(self, image: Image.Image) -> Dict:
        """
        Compute statistical features from an input image.

        Args:
            image: PIL Image.

        Returns:
            Dictionary with pixel statistics.
        """
        img_array = np.array(image.convert("L")).astype(np.float32) / 255.0

        return {
            "mean": float(np.mean(img_array)),
            "std": float(np.std(img_array)),
            "min": float(np.min(img_array)),
            "max": float(np.max(img_array)),
            "median": float(np.median(img_array)),
            "skewness": float(_skewness(img_array.flatten())),
            "width": image.width,
            "height": image.height,
        }

    def log_prediction(
        self,
        image: Image.Image,
        prediction: Dict,
    ):
        """
        Log a prediction for drift monitoring.

        Args:
            image: Input image.
            prediction: Prediction result dict (label, confidence, probabilities).
        """
        # Compute and store image stats
        stats = self.compute_image_stats(image)
        self.recent_image_stats.append(stats)

        # Store prediction info
        self.recent_predictions.append({
            "label": prediction["label"],
            "timestamp": datetime.now().isoformat(),
        })

        self.recent_confidences.append(prediction["confidence"])

    def check_drift(self) -> Dict:
        """
        Check for data drift in the recent predictions.

        Returns:
            Dictionary with drift analysis results:
              - drift_detected: bool
              - prediction_ratio: PNEUMONIA ratio in recent window
              - confidence_stats: mean/std of recent confidences
              - image_stats_drift: comparison of recent vs reference pixel stats
              - alert_message: Human-readable alert (if drift detected)
        """
        result = {
            "drift_detected": False,
            "timestamp": datetime.now().isoformat(),
            "window_size": len(self.recent_predictions),
        }

        if len(self.recent_predictions) < 10:
            result["message"] = "Insufficient data for drift detection"
            return result

        # Prediction distribution
        pneumonia_count = sum(
            1 for p in self.recent_predictions if p["label"] == "PNEUMONIA"
        )
        pneumonia_ratio = pneumonia_count / len(self.recent_predictions)
        result["prediction_ratio"] = {
            "PNEUMONIA": round(pneumonia_ratio, 4),
            "NORMAL": round(1 - pneumonia_ratio, 4),
        }

        # Confidence statistics
        confidences = list(self.recent_confidences)
        result["confidence_stats"] = {
            "mean": round(float(np.mean(confidences)), 4),
            "std": round(float(np.std(confidences)), 4),
            "min": round(float(np.min(confidences)), 4),
        }

        # Image statistics drift (compare recent vs reference)
        if self.reference_stats and len(self.recent_image_stats) > 0:
            recent_means = [s["mean"] for s in self.recent_image_stats]
            recent_stds = [s["std"] for s in self.recent_image_stats]

            ref_mean = self.reference_stats.get("mean_pixel_mean", 0.5)
            ref_std = self.reference_stats.get("mean_pixel_std", 0.2)

            # Simple drift detection: z-test on pixel mean
            current_mean = np.mean(recent_means)
            z_score = abs(current_mean - ref_mean) / (ref_std / np.sqrt(len(recent_means)))

            result["image_stats_drift"] = {
                "current_pixel_mean": round(float(current_mean), 4),
                "reference_pixel_mean": round(float(ref_mean), 4),
                "z_score": round(float(z_score), 4),
                "drift_detected": z_score > 3.0,  # 3-sigma rule
            }

            if z_score > 3.0:
                result["drift_detected"] = True
                result["alert_message"] = (
                    f"Data drift detected! Pixel intensity distribution shifted "
                    f"(z-score: {z_score:.2f}). Current mean: {current_mean:.4f}, "
                    f"reference mean: {ref_mean:.4f}."
                )

        # Confidence drift check
        if result["confidence_stats"]["mean"] < 0.7:
            result["drift_detected"] = True
            result["alert_message"] = result.get("alert_message", "") + (
                f" Low average confidence ({result['confidence_stats']['mean']:.2f}) "
                "may indicate model degradation or OOD data."
            )

        return result

    def generate_drift_report(self, save_path: str = None) -> Optional[str]:
        """
        Generate a comprehensive drift report using Evidently AI.

        Args:
            save_path: Path to save the HTML report. Auto-generated if None.

        Returns:
            Path to the saved report, or None if generation failed.
        """
        if not EVIDENTLY_AVAILABLE:
            print("Evidently not installed, skipping report generation")
            return None

        if len(self.recent_image_stats) < 10:
            print("Insufficient data for Evidently report")
            return None

        try:
            # Build DataFrames for Evidently
            reference_data = pd.DataFrame([self.reference_stats] * 100)
            current_data = pd.DataFrame(list(self.recent_image_stats))

            # Align columns
            common_cols = list(
                set(reference_data.columns) & set(current_data.columns)
            )
            reference_data = reference_data[common_cols]
            current_data = current_data[common_cols]

            # Generate report
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_data, current_data=current_data)

            # Save report
            if save_path is None:
                os.makedirs(self.reports_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(
                    self.reports_dir, f"drift_report_{timestamp}.html"
                )

            report.save_html(save_path)
            print(f"Drift report saved to {save_path}")
            return save_path

        except Exception as e:
            print(f"Failed to generate drift report: {e}")
            return None

    @staticmethod
    def compute_reference_stats(
        data_dir: str,
        save_path: str = "reports/reference_stats.json",
        max_samples: int = 500,
    ):
        """
        Compute reference statistics from the training dataset.

        Should be run once after training to establish the baseline distribution.

        Args:
            data_dir: Path to training data directory.
            save_path: Where to save the reference stats JSON.
            max_samples: Maximum number of images to sample.
        """
        import random

        all_means = []
        all_stds = []

        for class_name in ["NORMAL", "PNEUMONIA"]:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            images = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith((".jpeg", ".jpg", ".png"))
            ]
            sampled = random.sample(images, min(len(images), max_samples // 2))

            for img_name in sampled:
                img_path = os.path.join(class_dir, img_name)
                img = np.array(Image.open(img_path).convert("L")).astype(np.float32) / 255.0
                all_means.append(float(np.mean(img)))
                all_stds.append(float(np.std(img)))

        stats = {
            "mean_pixel_mean": float(np.mean(all_means)),
            "mean_pixel_std": float(np.mean(all_stds)),
            "std_pixel_mean": float(np.std(all_means)),
            "std_pixel_std": float(np.std(all_stds)),
            "num_samples": len(all_means),
            "computed_at": datetime.now().isoformat(),
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"Reference stats saved to {save_path}")
        return stats


def _skewness(data: np.ndarray) -> float:
    """Compute skewness of a 1D array."""
    n = len(data)
    if n < 3:
        return 0.0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return float(np.mean(((data - mean) / std) ** 3))
