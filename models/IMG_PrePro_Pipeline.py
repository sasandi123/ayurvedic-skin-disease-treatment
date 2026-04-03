import os
import json
import time
import csv
from typing import Tuple, Dict, Any

import numpy as np
import cv2
import gradio as gr
from PIL import Image

# Global configuration for standardized model input
IMG_SIZE: Tuple[int, int] = (224, 224)
OUT_ROOT = os.path.abspath("./user_preprocessed_outputs")
os.makedirs(OUT_ROOT, exist_ok=True)
FEATURES_CSV = os.path.join(OUT_ROOT, "features_log.csv")


def validate_and_convert(img: Image.Image) -> Image.Image:
    """Verifies image integrity and enforces minimum resolution constraints."""
    if img is None:
        return None
    img = img.convert("RGB")
    w, h = img.size
    if w < 80 or h < 80:
        raise gr.Error(f"Image resolution too low ({w}x{h}). Please provide a clearer sample.")
    return img


def safe_name(name: str) -> str:
    """Sanitizes user input for filesystem compatibility."""
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (name or "").strip())
    return safe if safe else "auto_processed"


def laplacian_sharpness(rgb_np: np.ndarray) -> float:
    """Calculates image focus using the variance of the Laplacian operator."""
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def make_edge_preview(rgb: np.ndarray) -> Image.Image:
    """Generates a Canny edge map to visualize structural morphology."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


def center_crop_square(rgb: np.ndarray, crop_ratio: float) -> np.ndarray:
    """Executes a proportional center crop to standardize aspect ratio."""
    h, w = rgb.shape[:2]
    side = int(min(h, w) * float(crop_ratio))
    y0, x0 = (h - side) // 2, (w - side) // 2
    return rgb[y0:y0 + side, x0:x0 + side]


def denoise_nlmeans(rgb: np.ndarray, strength: int) -> np.ndarray:
    """Reduces Gaussian noise while preserving critical edge detail."""
    if strength <= 0: return rgb
    return cv2.fastNlMeansDenoisingColored(rgb, None, strength, strength, 7, 21)


def apply_clahe_rgb(rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Performs local contrast enhancement via LAB color space transformation."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
    lab2 = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def gamma_correction(rgb: np.ndarray, gamma: float) -> np.ndarray:
    """Adjusts luminance non-linearly to improve visibility in low-light regions."""
    if abs(gamma - 1.0) < 1e-6: return rgb
    table = (np.array([(i / 255.0) ** (1.0 / gamma) for i in range(256)]) * 255).astype("uint8")
    return cv2.LUT(rgb, table)


def gray_world_white_balance(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Normalizes color temperature by balancing RGB channel means."""
    if strength <= 0: return rgb
    img_f = rgb.astype(np.float32) + 1e-6
    scale = img_f.reshape(-1, 3).mean(axis=0).mean() / img_f.reshape(-1, 3).mean(axis=0)
    wb = np.clip(img_f * scale, 0, 255).astype(np.uint8)
    return cv2.addWeighted(rgb, 1 - strength, wb, strength, 0)


def unsharp_mask(rgb: np.ndarray, amount: float, radius: int) -> np.ndarray:
    """Enhances high-frequency spatial components for increased perceived clarity."""
    if amount <= 0: return rgb
    blurred = cv2.GaussianBlur(rgb, (2 * radius + 1, 2 * radius + 1), 0)
    return cv2.addWeighted(rgb, 1.0 + amount, blurred, -amount, 0)


def extract_features(rgb: np.ndarray) -> Dict[str, Any]:
    """Quantifies primary visual metrics for metadata logging."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "brightness_mean": float(np.mean(gray)),
        "contrast_std": float(np.std(gray)),
        "sharpness_laplacian": laplacian_sharpness(rgb),
        "rgb_mean": [float(np.mean(rgb[:, :, i])) for i in range(3)],
    }


def append_features_csv(out_name: str, meta: Dict[str, Any]) -> None:
    """Persists extracted feature vectors to a centralized CSV registry."""
    row = {"file": out_name, "timestamp": meta.get("timestamp"), **meta.get("features", {})}
    exists = os.path.exists(FEATURES_CSV)
    with open(FEATURES_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists: writer.writeheader()
        writer.writerow(row)


def process_auto(img: Image.Image, out_name: str):
    """Orchestrates the end-to-end automated preprocessing workflow."""
    if img is None: return None, None, {}

    # Geometry normalization
    img = validate_and_convert(img)
    rgb = center_crop_square(np.array(img), 0.95)
    rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # Enhancement sequence
    rgb = gray_world_white_balance(rgb, 0.6)
    rgb = denoise_nlmeans(rgb, 5)
    rgb = apply_clahe_rgb(rgb, 2.0)
    rgb = gamma_correction(rgb, 0.95)
    rgb = unsharp_mask(rgb, 0.5, 2)

    # Output generation and storage
    out_img = Image.fromarray(rgb)
    meta = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "features": extract_features(rgb)}

    name = safe_name(out_name)
    out_img.save(os.path.join(OUT_ROOT, f"{name}.png"))
    append_features_csv(name, meta)

    return out_img, make_edge_preview(rgb), meta


def main():
    """Initializes the Gradio event-driven user interface."""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧬 Automated Dermoscopic Analysis")
        gr.Markdown("Upload image to initiate autonomous preprocessing and feature logging.")

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Source Image")
                filename = gr.Textbox(label="Export Alias", value="processed_sample")
            with gr.Column():
                output_main = gr.Image(label="Standardized Output (224x224)")
                output_edge = gr.Image(label="Structural Edge Map")

        with gr.Accordion("Quantitative Metadata", open=True):
            output_json = gr.JSON()

        # Reactive trigger linked to the image input state
        input_img.change(
            fn=process_auto,
            inputs=[input_img, filename],
            outputs=[output_main, output_edge, output_json]
        )

    demo.launch()


if __name__ == "__main__":
    main()