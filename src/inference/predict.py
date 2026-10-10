"""
Inference — Prediction logic for the Pneumonia detection model.

Provides:
  - PneumoniaPredictor class for single and batch inference
  - Image preprocessing for raw inputs (file path, PIL Image, bytes)
  - GradCAM visualization overlay
"""

import io
import os
from typing import Dict, List, Union, Optional

import torch
import numpy as np
from PIL import Image

from src.data.preprocessing import get_inference_transforms
from src.data.dataset import CLASS_NAMES
from src.models.utils import load_model, get_device


class PneumoniaPredictor:
    """
    High-level predictor for Chest X-ray Pneumonia detection.

    Wraps model loading, preprocessing, and inference into a simple API.

    Usage:
        predictor = PneumoniaPredictor("models/best_model.pth")
        result = predictor.predict("path/to/xray.jpeg")
        print(result)
        # {"label": "PNEUMONIA", "confidence": 0.97,
        #  "probabilities": {"NORMAL": 0.03, "PNEUMONIA": 0.97}}

    Args:
        model_path: Path to the saved model checkpoint.
        device: Device string (auto-detected if None).
        image_size: Input image size (default 224).
    """

    def __init__(
        self,
        model_path: str = "models/best_model.pth",
        device: str = None,
        image_size: int = 224,
    ):
        self.device = device or get_device()
        self.image_size = image_size
        self.class_names = CLASS_NAMES
        self.transform = get_inference_transforms(image_size)

        # Load model
        self.model = load_model(
            model_path, num_classes=len(self.class_names), device=self.device
        )
        self.model.eval()

    def predict(self, image: Union[str, Image.Image, bytes]) -> Dict:
        """
        Predict class for a single chest X-ray image.

        Args:
            image: Input image as file path, PIL Image, or bytes.

        Returns:
            Dictionary with keys:
              - label: Predicted class name ("NORMAL" or "PNEUMONIA")
              - confidence: Confidence score (0-1)
              - probabilities: Dict mapping class names to probabilities
        """
        img_tensor = self._preprocess(image)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # Add batch dim

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze()

        predicted_idx = probs.argmax().item()
        confidence = probs[predicted_idx].item()

        return {
            "label": self.class_names[predicted_idx],
            "confidence": round(confidence, 4),
            "probabilities": {
                name: round(probs[i].item(), 4)
                for i, name in enumerate(self.class_names)
            },
        }

    def predict_batch(self, images: List[Union[str, Image.Image, bytes]]) -> List[Dict]:
        """
        Predict classes for a batch of images.

        Args:
            images: List of input images.

        Returns:
            List of prediction dictionaries.
        """
        tensors = [self._preprocess(img) for img in images]
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)

        results = []
        for i in range(len(images)):
            p = probs[i]
            predicted_idx = p.argmax().item()
            confidence = p[predicted_idx].item()

            results.append({
                "label": self.class_names[predicted_idx],
                "confidence": round(confidence, 4),
                "probabilities": {
                    name: round(p[j].item(), 4)
                    for j, name in enumerate(self.class_names)
                },
            })

        return results

    def predict_with_gradcam(
        self, image: Union[str, Image.Image, bytes]
    ) -> Dict:
        """
        Predict with GradCAM visualization.

        Returns prediction results plus a GradCAM heatmap as a numpy array.

        Args:
            image: Input image.

        Returns:
            Dictionary with prediction results + "gradcam" key containing
            the heatmap as a numpy array (H, W) with values in [0, 1].
        """
        pil_image = self._to_pil(image)
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Get prediction
        result = self.predict(image)

        # Get GradCAM
        try:
            target_class = self.class_names.index(result["label"])
            heatmap = self.model.get_gradcam(img_tensor, target_class=target_class)
            result["gradcam"] = heatmap.cpu().numpy()
        except Exception as e:
            result["gradcam"] = None
            result["gradcam_error"] = str(e)

        return result

    def _preprocess(self, image: Union[str, Image.Image, bytes]) -> torch.Tensor:
        """
        Preprocess an image for model input.

        Handles:
          - File path string
          - PIL Image object
          - Raw bytes

        Args:
            image: Input image in any supported format.

        Returns:
            Preprocessed tensor of shape (3, 224, 224).
        """
        pil_image = self._to_pil(image)
        return self.transform(pil_image)

    def _to_pil(self, image: Union[str, Image.Image, bytes]) -> Image.Image:
        """Convert input to PIL Image."""
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, bytes):
            pil_image = Image.open(io.BytesIO(image))
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError(
                f"Unsupported image type: {type(image)}. "
                "Expected str (path), PIL.Image, or bytes."
            )
        return pil_image.convert("RGB")
