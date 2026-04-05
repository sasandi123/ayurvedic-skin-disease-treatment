import os
import uuid

import numpy as np
import torch
from flask import Flask, request, render_template_string
from PIL import Image
from torchvision import transforms
from werkzeug.utils import secure_filename

from train import (
    CLASS_NAMES,
    MEAN,
    STD,
    SquarePad,
    build_model,
    default_thresholds,
    forward_class_probs,
    load_checkpoint,
    predict_from_scores,
)


app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_PATH = "severity_model_pytorch.h5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
class_names = list(CLASS_NAMES)
img_size = 320
dropout = 0.35
thresholds = default_thresholds()
zoom_margin = 24
mean = MEAN
std = STD
load_error = None


def load_model():
    global model, class_names, img_size, dropout, thresholds, mean, std, load_error
    if not os.path.exists(MODEL_PATH):
        load_error = f"{MODEL_PATH} not found. Train the model first."
        return

    checkpoint = load_checkpoint(MODEL_PATH, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        load_error = "Unsupported checkpoint format. Retrain with the updated train.py script."
        return

    class_names = checkpoint.get("class_names", list(CLASS_NAMES))
    img_size = int(checkpoint.get("img_size", 320))
    dropout = float(checkpoint.get("dropout", 0.35))
    thresholds = checkpoint.get("ordinal_thresholds", default_thresholds())
    mean = checkpoint.get("mean", MEAN)
    std = checkpoint.get("std", STD)

    model_name = checkpoint.get("model_name", "efficientnet_v2_m")
    model_instance = build_model(model_name, dropout, device, pretrained=False)
    model_instance.load_state_dict(checkpoint["state_dict"])
    model_instance.to(device)
    model_instance.eval()
    model = model_instance


load_model()


def inference_transform():
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Severity Classification</title>
<style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #16324f 0%, #28587b 100%);
        color: #333;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        margin: 0;
        padding: 24px;
        box-sizing: border-box;
    }

    .card {
        background: rgba(255, 255, 255, 0.96);
        padding: 36px;
        border-radius: 22px;
        box-shadow: 0 18px 40px rgba(0,0,0,0.22);
        text-align: center;
        width: 100%;
        max-width: 520px;
    }

    h1 { color: #16324f; margin-bottom: 8px; font-size: 28px; }
    h2 { color: #607d8b; font-size: 16px; margin-bottom: 28px; font-weight: normal; }

    .upload-section {
        border: 2px dashed #9fb3c8;
        padding: 22px;
        border-radius: 14px;
        margin-bottom: 18px;
        transition: border-color 0.3s;
    }

    .upload-section:hover { border-color: #16324f; }

    button {
        background: #16324f;
        color: white;
        border: none;
        padding: 13px 30px;
        border-radius: 999px;
        cursor: pointer;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: transform 0.2s, background 0.2s;
    }

    button:hover {
        background: #1f466c;
        transform: translateY(-1px);
    }

    img {
        margin-top: 22px;
        max-width: 100%;
        height: auto;
        border-radius: 14px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }

    .result {
        margin-top: 22px;
        padding: 18px;
        border-radius: 14px;
        font-size: 24px;
        font-weight: bold;
    }

    .meta {
        margin-top: 8px;
        font-size: 14px;
        font-weight: normal;
    }

    .severity-Mild { background: #e8f5e9; color: #2e7d32; }
    .severity-Moderate { background: #fff3e0; color: #ef6c00; }
    .severity-Severe { background: #ffebee; color: #c62828; }
    .error { background: #fff4e5; color: #8a4b08; }

    .bars {
        margin-top: 18px;
        text-align: left;
    }

    .bar-row {
        margin-bottom: 10px;
    }

    .bar-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
        font-size: 14px;
        color: #37474f;
    }

    .bar-track {
        width: 100%;
        height: 10px;
        background: #dbe7f0;
        border-radius: 999px;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #2a7bbd 0%, #16324f 100%);
    }
</style>
</head>
<body>
<div class="card">
    <h1>Skin Disease Detection</h1>
    <h2>Ordinal Severity Classification</h2>

    <form method="POST" enctype="multipart/form-data">
        <div class="upload-section">
            <input type="file" name="file" accept="image/*" required>
        </div>
        <button type="submit">Analyze Image</button>
    </form>

    {% if image_path %}
    <img src="/{{ image_path }}">
    {% endif %}

    {% if message %}
    <div class="result {{ result_class }}">
        {{ message }}
        {% if confidence is not none %}
        <div class="meta">Confidence: {{ "%.1f"|format(confidence * 100) }}%</div>
        {% endif %}
    </div>
    {% endif %}

    {% if probabilities %}
    <div class="bars">
        {% for label, value in probabilities %}
        <div class="bar-row">
            <div class="bar-label">
                <span>{{ label }}</span>
                <span>{{ "%.1f"|format(value * 100) }}%</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width: {{ value * 100 }}%"></div>
            </div>
        </div>
        {% endfor %}
    </div>
    {% endif %}
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    message = None
    result_class = ""
    confidence = None
    probabilities = None
    image_display_path = None

    if load_error:
        message = load_error
        result_class = "error"

    if request.method == "POST" and model is not None:
        file = request.files.get("file")
        if file and file.filename:
            original_name = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{original_name or 'upload.jpg'}"
            filepath = os.path.join(UPLOAD_FOLDER, unique_name)
            file.save(filepath)
            image_display_path = filepath.replace("\\", "/")

            try:
                image = Image.open(filepath).convert("RGB")
                tensor = inference_transform()(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    class_probs = forward_class_probs(model, tensor, use_tta=True, zoom_margin=zoom_margin)

                prob_vector = class_probs[0].cpu().numpy()
                score = np.array([prob_vector @ np.arange(len(class_names), dtype=np.float32)], dtype=np.float32)
                pred_idx = int(predict_from_scores(score, thresholds)[0])
                pred_idx = max(0, min(pred_idx, len(class_names) - 1))

                message = f"Result: {class_names[pred_idx]}"
                result_class = f"severity-{class_names[pred_idx]}"
                confidence = float(prob_vector[pred_idx])
                probabilities = list(zip(class_names, prob_vector.tolist()))
            except Exception as exc:
                message = f"Error: {exc}"
                result_class = "error"

    return render_template_string(
        HTML_TEMPLATE,
        message=message,
        result_class=result_class,
        confidence=confidence,
        probabilities=probabilities,
        image_path=image_display_path,
    )


if __name__ == "__main__":
    app.run(debug=False, port=5000)
