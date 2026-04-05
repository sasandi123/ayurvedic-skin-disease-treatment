import os
import json
import time
import csv
from typing import Tuple, Dict, Any

import numpy as np
import cv2
import gradio as gr
from PIL import Image

# ---------------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------------

def validate_and_convert(img: Image.Image) -> Image.Image:
    if img is None:
        raise ValueError("Invalid image")
    img = img.convert("RGB")
    w, h = img.size
    if w < 80 or h < 80:
        raise ValueError(f"Image resolution too low ({w}x{h})")
    return img


def center_crop_square(rgb: np.ndarray, crop_ratio: float) -> np.ndarray:
    h, w  = rgb.shape[:2]
    side  = int(min(h, w) * float(crop_ratio))
    y0    = (h - side) // 2
    x0    = (w - side) // 2
    return rgb[y0:y0 + side, x0:x0 + side]


def denoise_nlmeans(rgb: np.ndarray, strength: int) -> np.ndarray:
    if strength <= 0:
        return rgb
    return cv2.fastNlMeansDenoisingColored(rgb, None, strength, strength, 7, 21)


def apply_clahe_rgb(rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    lab        = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b    = cv2.split(lab)
    clahe      = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
    lab2       = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def gamma_correction(rgb: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-6:
        return rgb
    inv_gamma = 1.0 / gamma
    table     = (np.array([(i / 255.0) ** inv_gamma for i in range(256)]) * 255).astype("uint8")
    return cv2.LUT(rgb, table)


def gray_world_white_balance(rgb: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return rgb
    img_f = rgb.astype(np.float32) + 1e-6
    scale = img_f.reshape(-1, 3).mean(axis=0).mean() / img_f.reshape(-1, 3).mean(axis=0)
    wb    = np.clip(img_f * scale, 0, 255).astype(np.uint8)
    return cv2.addWeighted(rgb, 1 - strength, wb, strength, 0)


def unsharp_mask(rgb: np.ndarray, amount: float, radius: int) -> np.ndarray:
    if amount <= 0:
        return rgb
    blurred = cv2.GaussianBlur(rgb, (2 * radius + 1, 2 * radius + 1), 0)
    return cv2.addWeighted(rgb, 1.0 + amount, blurred, -amount, 0)


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = validate_and_convert(img)
    rgb = center_crop_square(np.array(img), 0.95)
    rgb = cv2.resize(rgb, PREPROCESS_SIZE, interpolation=cv2.INTER_AREA)
    rgb = gray_world_white_balance(rgb, 0.6)
    rgb = denoise_nlmeans(rgb, 5)
    rgb = apply_clahe_rgb(rgb, 2.0)
    rgb = gamma_correction(rgb, 0.95)
    rgb = unsharp_mask(rgb, 0.5, 2)
    return rgb


def prepare_for_disease_model(preprocessed_rgb: np.ndarray) -> np.ndarray:
    """Raw 0-255 values. No division by 255."""
    img_pil     = Image.fromarray(preprocessed_rgb.astype('uint8'))
    img_resized = img_pil.resize(DISEASE_MODEL_SIZE, Image.Resampling.LANCZOS)
    arr         = np.array(img_resized, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def prepare_for_severity_model(preprocessed_rgb: np.ndarray) -> np.ndarray:
    """Raw 0-255 values. No division by 255."""
    img_pil     = Image.fromarray(preprocessed_rgb.astype('uint8'))
    img_resized = img_pil.resize(SEVERITY_MODEL_SIZE, Image.Resampling.LANCZOS)
    arr         = np.array(img_resized, dtype=np.float32)
    return np.expand_dims(arr, axis=0)