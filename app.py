"""Interactive Gradio presentation app for RoadSense traffic-sign models."""
import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from src.corruptions import CORRUPTIONS, apply_corruption  # noqa: E402
from src.explainability import (  # noqa: E402
    GradCAM,
    load_model_for_explainability,
    overlay_heatmap,
    preprocess_pil_image,
)
from src.utils import get_class_name  # noqa: E402
from ui.results import (  # noqa: E402
    dataset_figures,
    explainability_figures,
    model_metrics_view,
    robustness_view,
)

MODEL_CHOICES = ["baseline_cnn", "improved_cnn", "resnet18", "mobilenet_v2"]
TRANSFER_ARCHS = {"resnet18", "mobilenet_v2"}
MODEL_LABELS = {
    "baseline_cnn": "Baseline CNN",
    "improved_cnn": "Improved CNN",
    "resnet18": "ResNet-18",
    "mobilenet_v2": "MobileNetV2",
}
WORK_SIZE = 64      # uploads are shrunk to a GTSRB-like size BEFORE corrupting, so the
                    # corruption strengths mean the same thing as in the robustness bench
DISPLAY_SIZE = 256  # heatmaps are drawn this big so they are easy to see
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
APP_CSS = """
.gradio-container { max-width: 1440px !important; }
h1, h2, h3 { font-family: 'Avenir Next', Georgia, serif !important; }
.roadsense-intro { max-width: 960px; }
@media (max-width: 700px) { .gradio-container { padding: 12px !important; } }
"""
APP_THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="orange", neutral_hue="slate")

_model_cache = {}


def get_model(model_name):
    """Load a model once and reuse its cached model and checkpoint state."""
    if model_name not in _model_cache:
        _model_cache[model_name] = load_model_for_explainability(model_name, DEVICE)
    return _model_cache[model_name]


def explain(model, model_type, img_size, pil_img):
    """Return a Grad-CAM overlay, top-five scores, predicted class, and confidence."""
    tensor = preprocess_pil_image(
        pil_img, img_size=img_size, use_imagenet_norm=model_type in TRANSFER_ARCHS, device=DEVICE
    )
    with GradCAM(model) as cam:
        heatmap, pred, conf, probs = cam.generate(tensor)

    shown = pil_img.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.BILINEAR)
    overlay = overlay_heatmap(shown, heatmap, alpha=0.45)

    top5 = np.argsort(probs)[::-1][:5]
    top5_dict = {f"{int(i)}: {get_class_name(int(i))}": float(probs[i]) for i in top5}
    return overlay, top5_dict, int(pred), float(conf)


def predict(model, model_type, img_size, pil_img):
    """Run an inference-only prediction without retaining gradient state."""
    tensor = preprocess_pil_image(
        pil_img, img_size=img_size, use_imagenet_norm=model_type in TRANSFER_ARCHS, device=DEVICE
    )
    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    pred = int(np.argmax(probs))
    return pred, float(probs[pred])


def run(image, model_name, corruption, severity, seed):
    """Compare one model's clean and corrupted predictions with Grad-CAM overlays."""
    if image is None:
        raise gr.Error("Please upload a traffic-sign image first.")

    try:
        model, model_type, img_size, loaded = get_model(model_name)
        base = image.convert("RGB").resize((WORK_SIZE, WORK_SIZE), Image.Resampling.LANCZOS)
        if corruption == "none":
            corrupted = base.copy()
        else:
            corrupted = apply_corruption(base, corruption, severity=int(severity), seed=int(seed))
        clean_overlay, clean_top5, clean_pred, clean_conf = explain(
            model, model_type, img_size, base
        )
        corr_overlay, corr_top5, corr_pred, corr_conf = explain(
            model, model_type, img_size, corrupted
        )
    except Exception as exc:
        raise gr.Error(f"Could not run {model_name}: {exc}") from exc

    lines = []
    if not loaded:
        lines.append(
            f"**WARNING: no checkpoint found for `{model_name}`** "
            f"(expected `checkpoints/best_{model_name}.pth`). The model is untrained, "
            "so predictions below are meaningless."
        )
    lines.append(
        f"**Model:** `{model_name}` ({img_size}x{img_size} input)  |  "
        f"**Clean:** {get_class_name(clean_pred)} ({clean_conf:.1%})"
    )
    if corruption == "none":
        lines.append("No corruption applied: prediction unchanged; confidence change +0.0%.")
    else:
        verdict = "UNCHANGED" if corr_pred == clean_pred else "CHANGED"
        lines.append(
            f"**{corruption}, severity {int(severity)}:** {get_class_name(corr_pred)} "
            f"({corr_conf:.1%})  |  prediction **{verdict}**, "
            f"confidence change {corr_conf - clean_conf:+.1%}"
        )
    return (
        "\n\n".join(lines), base, clean_overlay, clean_top5,
        corrupted, corr_overlay, corr_top5,
    )


def compare_all_models(image, corruption, severity, seed):
    """Compare clean and corrupted predictions for each model with a checkpoint."""
    if image is None:
        return "Upload an image before comparing models.", pd.DataFrame()

    base = image.convert("RGB").resize((WORK_SIZE, WORK_SIZE), Image.Resampling.LANCZOS)
    if corruption == "none":
        corrupted = base.copy()
    else:
        corrupted = apply_corruption(base, corruption, severity=int(severity), seed=int(seed))

    rows = []
    problems = []
    missing = []
    for model_name in MODEL_CHOICES:
        checkpoint = CHECKPOINT_DIR / f"best_{model_name}.pth"
        if not checkpoint.is_file():
            missing.append(model_name)
            continue
        try:
            model, model_type, img_size, _ = get_model(model_name)
            clean_pred, clean_conf = predict(model, model_type, img_size, base)
            corr_pred, corr_conf = predict(model, model_type, img_size, corrupted)
            rows.append([
                MODEL_LABELS[model_name], get_class_name(clean_pred), f"{clean_conf:.1%}",
                get_class_name(corr_pred), f"{corr_conf:.1%}",
            ])
        except Exception as exc:
            problems.append(f"{model_name}: {exc}")

    if not rows:
        message = "No trained checkpoints found. Add best_<model>.pth files under checkpoints/."
    else:
        message = f"Compared {len(rows)} trained model(s)."
    if missing:
        message += " Untrained models skipped: " + ", ".join(missing) + "."
    if problems:
        message += " Errors: " + "; ".join(problems)
    table = pd.DataFrame(rows, columns=[
        "Model", "Clean prediction", "Clean confidence",
        "Corrupted prediction", "Corrupted confidence",
    ])
    return message, table


def update_robustness(model_name):
    """Refresh the selected model's corruption-by-severity accuracy chart."""
    return robustness_view(RESULTS_DIR, model_name)


def update_explainability(model_name):
    """Refresh the Grad-CAM figure gallery for the selected model."""
    return explainability_figures(RESULTS_DIR, model_name)


def build_demo():
    """Create all presentation tabs without loading model weights at startup."""
    with gr.Blocks(title="RoadSense | Traffic Sign Recognition") as demo:
        gr.Markdown(
            "# RoadSense: Robust Traffic Sign Recognition\n"
            "Atria University · Group 4 · GTSRB, 43 classes",
            elem_classes="roadsense-intro",
        )
        missing_checkpoints = [
            name for name in MODEL_CHOICES
            if not (CHECKPOINT_DIR / f"best_{name}.pth").is_file()
        ]
        if missing_checkpoints:
            gr.Markdown(
                "**Untrained model warning:** no checkpoint found for "
                f"{', '.join(MODEL_LABELS[name] for name in missing_checkpoints)}. "
                "Add `best_<model>.pth` files under `checkpoints/` before interpreting predictions."
            )

        with gr.Tab("Predict"):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    image_input = gr.Image(type="pil", label="Traffic sign image", height=260)
                    model_input = gr.Dropdown(
                        MODEL_CHOICES, value="baseline_cnn", label="Model",
                        info="Models without checkpoints run untrained and are marked clearly.",
                    )
                    corruption_input = gr.Dropdown(
                        ["none"] + list(CORRUPTIONS), value="none", label="Corruption"
                    )
                    severity_input = gr.Radio([1, 2, 3], value=2, label="Severity")
                    seed_input = gr.Number(value=42, precision=0, label="Random seed")
                    predict_button = gr.Button("Classify image", variant="primary")
                    compare_button = gr.Button("Compare all trained models")
                with gr.Column(scale=2, min_width=320):
                    prediction_status = gr.Markdown()
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Clean")
                            clean_image = gr.Image(label="Input", height=190)
                            clean_cam = gr.Image(label="Grad-CAM overlay", height=256)
                            clean_labels = gr.Label(num_top_classes=5, label="Top-5 predictions")
                        with gr.Column():
                            gr.Markdown("### Corrupted")
                            corrupted_image = gr.Image(label="Input", height=190)
                            corrupted_cam = gr.Image(label="Grad-CAM overlay", height=256)
                            corrupted_labels = gr.Label(num_top_classes=5, label="Top-5 predictions")
                    comparison_status = gr.Markdown()
                    comparison_table = gr.Dataframe(
                        headers=[
                            "Model", "Clean prediction", "Clean confidence",
                            "Corrupted prediction", "Corrupted confidence",
                        ],
                        datatype=["str", "str", "str", "str", "str"],
                        interactive=False,
                        label="Model predictions",
                    )
            predict_button.click(
                run,
                inputs=[image_input, model_input, corruption_input, severity_input, seed_input],
                outputs=[
                    prediction_status, clean_image, clean_cam, clean_labels,
                    corrupted_image, corrupted_cam, corrupted_labels,
                ],
            )
            compare_button.click(
                compare_all_models,
                inputs=[image_input, corruption_input, severity_input, seed_input],
                outputs=[comparison_status, comparison_table],
            )

        with gr.Tab("Robustness"):
            gr.Markdown("## Accuracy under visual corruption")
            initial_plot, initial_status = update_robustness("baseline_cnn")
            robustness_model = gr.Dropdown(MODEL_CHOICES, value="baseline_cnn", label="Model")
            robustness_status = gr.Markdown(initial_status)
            robustness_plot = gr.Plot(
                value=initial_plot, label="Accuracy by corruption and severity"
            )
            robustness_model.change(
                update_robustness, inputs=robustness_model,
                outputs=[robustness_plot, robustness_status],
            )

        with gr.Tab("Compare"):
            gr.Markdown("## Clean test-set performance")
            metrics_table, metrics_plot, metrics_status = model_metrics_view(
                RESULTS_DIR, MODEL_CHOICES, MODEL_LABELS
            )
            gr.Markdown(metrics_status)
            gr.Dataframe(value=metrics_table, interactive=False, label="Model metrics")
            gr.Plot(value=metrics_plot, label="Accuracy and macro-F1")

        with gr.Tab("EDA"):
            gr.Markdown("## GTSRB dataset overview")
            eda_items, eda_status = dataset_figures(RESULTS_DIR)
            gr.Markdown(eda_status)
            gr.Gallery(
                value=eda_items, label="Dataset and EDA figures", columns=2,
                height="auto", object_fit="contain",
            )

        with gr.Tab("Explainability gallery"):
            gr.Markdown("## Saved Grad-CAM analyses")
            initial_items, initial_message = update_explainability("baseline_cnn")
            explainability_model = gr.Dropdown(
                MODEL_CHOICES, value="baseline_cnn", label="Model"
            )
            explainability_status = gr.Markdown(initial_message)
            explainability_gallery = gr.Gallery(
                value=initial_items, label="Grad-CAM figures", columns=2,
                height="auto", object_fit="contain",
            )
            explainability_model.change(
                update_explainability, inputs=explainability_model,
                outputs=[explainability_gallery, explainability_status],
            )

        with gr.Tab("About"):
            gr.Markdown(
                "## RoadSense\n"
                "**Atria University · Group 4**\n\n"
                "**Team:** Raju, Manoj, Sohail, and Aseem (Team Leader).\n\n"
                "### Project phases\n"
                "1. Dataset acquisition and verification\n"
                "2. Exploratory analysis and leakage-free splitting\n"
                "3. Baseline CNN design and evaluation\n"
                "4. Robustness testing under visual corruptions\n"
                "5. Label-safe augmentation and transfer learning\n"
                "6. Grad-CAM explainability and real-world evaluation\n\n"
                "The models were trained on German GTSRB traffic signs. Signs from other "
                "countries may be misread because their designs differ from the training data."
            )
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_demo().launch(
        share=args.share, server_port=args.port, theme=APP_THEME, css=APP_CSS
    )
