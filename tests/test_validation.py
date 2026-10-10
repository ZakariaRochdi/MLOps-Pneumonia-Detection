"""
Test Validation — Model performance validation gate for CI/CD.

These tests act as the CI gate:
  - F1-Score must be > 0.95
  - Accuracy must be > 0.90

If these tests fail, the CI pipeline blocks deployment.
"""

import os
import json
import pytest

from src.training.config import load_config


# ── CI Validation Gate ───────────────────────────────────────────────────────


class TestModelValidationGate:
    """
    Model performance validation gate.

    These tests read the evaluation metrics JSON produced by the evaluate stage
    and assert that they meet the minimum thresholds required for deployment.
    """

    @pytest.fixture
    def config(self):
        """Load project configuration."""
        return load_config()

    @pytest.fixture
    def eval_metrics(self, config):
        """Load evaluation metrics from the reports directory."""
        metrics_path = config.paths.eval_metrics
        if not os.path.exists(metrics_path):
            pytest.skip(
                f"Evaluation metrics not found at {metrics_path}. "
                "Run evaluation first: python -m src.training.evaluate"
            )
        with open(metrics_path, "r") as f:
            return json.load(f)

    def test_f1_score_threshold(self, config, eval_metrics):
        """
        F1-Score must be ≥ threshold (default 0.95).

        This is the primary CI gate. Deployment is blocked if this test fails.
        """
        threshold = config.validation.f1_threshold
        f1 = eval_metrics["f1_weighted"]

        assert f1 >= threshold, (
            f"F1-Score validation FAILED: {f1:.4f} < {threshold}. "
            f"Model does not meet the minimum deployment threshold."
        )

    def test_accuracy_threshold(self, config, eval_metrics):
        """
        Accuracy must be ≥ threshold (default 0.90).

        Secondary validation gate for deployment.
        """
        threshold = config.validation.accuracy_threshold
        accuracy = eval_metrics["accuracy"]

        assert accuracy >= threshold, (
            f"Accuracy validation FAILED: {accuracy:.4f} < {threshold}. "
            f"Model does not meet the minimum deployment threshold."
        )

    def test_per_class_f1_balance(self, eval_metrics):
        """
        Per-class F1-scores should be reasonably balanced.

        Neither class should have an F1-score below 0.85, which would indicate
        the model is failing on one class despite good overall metrics.
        """
        per_class_f1 = eval_metrics.get("f1_per_class", {})
        if not per_class_f1:
            pytest.skip("Per-class F1 not available in metrics")

        for class_name, f1 in per_class_f1.items():
            assert f1 >= 0.85, (
                f"Per-class F1 too low for {class_name}: {f1:.4f} < 0.85. "
                f"Model may be biased against this class."
            )

    def test_roc_auc_threshold(self, eval_metrics):
        """ROC AUC should be ≥ 0.95 for reliable discrimination."""
        roc_auc = eval_metrics.get("roc_auc")
        if roc_auc is None:
            pytest.skip("ROC AUC not available in metrics")

        assert roc_auc >= 0.95, (
            f"ROC AUC too low: {roc_auc:.4f} < 0.95. "
            f"Model discrimination ability is insufficient."
        )

    def test_metrics_completeness(self, eval_metrics):
        """Evaluation metrics should contain all required fields."""
        required_fields = [
            "accuracy",
            "f1_weighted",
            "precision_weighted",
            "recall_weighted",
            "total_samples",
        ]
        for field in required_fields:
            assert field in eval_metrics, (
                f"Missing required metric: {field}"
            )

    def test_sufficient_test_samples(self, eval_metrics):
        """Test set should have a meaningful number of samples."""
        total = eval_metrics.get("total_samples", 0)
        assert total >= 100, (
            f"Too few test samples: {total}. "
            f"Need at least 100 samples for reliable validation."
        )
