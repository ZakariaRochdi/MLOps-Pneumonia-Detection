import os
import matplotlib
import numpy as np
import gradio as gr
from matplotlib import cm
from PIL import Image
from src.data.dataset import CLASS_NAMES
from src.inference.predict import PneumoniaPredictor

"""
Gradio Application — Interactive web UI for Pneumonia detection.

Features:
  - Drag-and-drop image upload
  - Prediction display with confidence scores
  - Probability bar chart
  - GradCAM heatmap overlay visualization
  - Example images for quick testing
  - Mobile-friendly interface
"""

matplotlib.use("Agg")

# ── Global predictor ─────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pth")
predictor = None


def load_predictor():
    """Load the model predictor (lazy initialization)."""
    global predictor
    if predictor is None:
        try:
            predictor = PneumoniaPredictor(model_path=MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    return predictor


def predict_image(image):
    """
    Predict pneumonia from an uploaded chest X-ray image.

    Args:
        image: PIL Image or numpy array from Gradio input.

    Returns:
        Tuple of (label_text, confidences_dict, gradcam_overlay).
    """
    pred = load_predictor()
    if pred is None:
        return (
            "⚠️ Model not loaded. Please ensure model file exists.",
            {},
            None,
        )

    if image is None:
        return "⚠️ Please upload an image.", {}, None

    # Convert numpy to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Get prediction
    result = pred.predict(image)

    label = result["label"]
    confidence = result["confidence"]
    probs = result["probabilities"]

    # Format output label
    emoji = "🔴" if label == "PNEUMONIA" else "🟢"
    label_text = f"{emoji} {label} ({confidence * 100:.1f}%)"

    # Confidences for Gradio Label component
    confidences = {name: probs[name] for name in CLASS_NAMES}

    # GradCAM overlay
    gradcam_image = None
    try:
        gradcam_result = pred.predict_with_gradcam(image)
        if gradcam_result.get("gradcam") is not None:
            gradcam_image = create_gradcam_overlay(image, gradcam_result["gradcam"])
    except Exception:
        pass  # GradCAM is optional

    return label_text, confidences, gradcam_image


def create_gradcam_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Create a GradCAM heatmap overlay on the original image.

    Args:
        original_image: Original PIL Image.
        heatmap: GradCAM heatmap (H, W) with values in [0, 1].
        alpha: Overlay transparency (0=original, 1=heatmap).

    Returns:
        Numpy array of the overlaid image.
    """
    # Resize original to match heatmap display size
    img = original_image.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0

    # Apply colormap to heatmap
    colormap = cm.jet(heatmap)[:, :, :3]  # Remove alpha channel

    # Blend
    overlay = (1 - alpha) * img_array + alpha * colormap
    overlay = np.clip(overlay, 0, 1)

    return (overlay * 255).astype(np.uint8)

# ── Gradio Interface ─────────────────────────────────────────────────────────


def build_interface():
    """Build and return the Gradio interface."""

    with gr.Blocks(
        title="Chest X-ray Pneumonia Detection",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
        ),
    ) as demo:
        gr.Markdown(
            """
            # 🫁 Chest X-ray Pneumonia Detection
            ### Powered by DenseNet121 Deep Learning Model

            Upload a chest X-ray image to detect whether it shows signs of
            **pneumonia** or is **normal**. The model uses a DenseNet121 architecture
            trained on thousands of labeled X-ray images.

            ---
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Chest X-ray",
                    type="pil",
                    height=350,
                )
                predict_btn = gr.Button(
                    "🔍 Analyze X-ray",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                output_label = gr.Textbox(
                    label="Diagnosis",
                    interactive=False,
                )
                output_confidences = gr.Label(
                    label="Class Probabilities",
                    num_top_classes=2,
                )
                output_gradcam = gr.Image(
                    label="GradCAM Visualization",
                    height=250,
                )

        # Connect predict button
        predict_btn.click(
            fn=predict_image,
            inputs=[input_image],
            outputs=[output_label, output_confidences, output_gradcam],
        )

        # Also predict on image upload
        input_image.change(
            fn=predict_image,
            inputs=[input_image],
            outputs=[output_label, output_confidences, output_gradcam],
        )

        gr.Markdown(
            """
            ---
            > **⚠️ Disclaimer**: This tool is for educational and research
            > purposes only. It is NOT a substitute for professional medical
            > diagnosis. Always consult a qualified healthcare provider.

            **Model**: DenseNet121 (transfer learning from ImageNet)
            | **Input**: 224×224 RGB chest X-ray
            | **Output**: NORMAL / PNEUMONIA with confidence score
            """
        )

    return demo


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
