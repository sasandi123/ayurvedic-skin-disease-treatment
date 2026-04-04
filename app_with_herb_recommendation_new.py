import os
import sys
from pathlib import Path
import pickle
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
import cv2
import h5py
import json
import torch
import torchvision.models as tv_models
from PIL import Image
from flask import Flask, request, jsonify, render_template, redirect, url_for, make_response
from tensorflow import keras
import tensorflow as tf
import joblib

# ---------------------------------------------------------------
# FLASK APP SETUP
# ---------------------------------------------------------------

BASE_DIR     = Path(__file__).parent.absolute()
TEMPLATE_DIR = BASE_DIR / 'templates'

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------

PREPROCESS_SIZE     = (224, 224)
DISEASE_MODEL_SIZE  = (380, 380)
SEVERITY_MODEL_SIZE = (320, 320)
CLASS_NAMES         = ['acne', 'eczema', 'ringworm']
DATABASE_FILE       = 'ayurvedic_treatment_database_medical_validated.xlsx'

# Global models and data
disease_model       = None
severity_model      = None
severity_thresholds = [0.170583575963974, 1.3933634757995605]  # default fallback
herb_rf_model       = None
label_encoders      = None
mlb_herbs           = None
treatment_database  = None


# ---------------------------------------------------------------
# CORAL ORDINAL HEAD
# Matches saved keys exactly:
#   classifier.1.fc.weight  shape (1, 1280)  — Linear without bias
#   classifier.1.bias       shape (2,)        — separate per-threshold bias
# ---------------------------------------------------------------

class CoralOrdinal(torch.nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        # fc has NO bias — bias is stored separately
        self.fc   = torch.nn.Linear(in_features, 1, bias=False)
        # one bias per ordinal threshold  (num_classes - 1 = 2)
        self.bias = torch.nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # broadcast: (B,1) + (num_classes-1,) → (B, num_classes-1)
        return self.fc(x) + self.bias


# ---------------------------------------------------------------
# PATIENT PROFILE MAPPING
# ---------------------------------------------------------------

def map_age_to_group(age: int) -> str:
    if age <= 12:   return 'child'
    elif age <= 19: return 'teen'
    elif age <= 60: return 'adult'
    else:           return 'elderly'


def map_skin_type(skin_type: str) -> str:
    s = str(skin_type).lower().strip()
    if s == 'oily': return 'oily'
    if s == 'dry':  return 'dry'
    return 'normal'


def map_sensitivity(skin_sensitivity: str) -> str:
    s = str(skin_sensitivity).lower().strip()
    if s in ['sensitive', 'high']: return 'sensitive'
    return 'normal'


# ---------------------------------------------------------------
# OOD DETECTION
# ---------------------------------------------------------------

class EnhancedOODDetector:
    """
    Minimal pre-check. Only rejects images that are technically
    unusable regardless of content.
    """
    def detect_non_skin_image(self, rgb_array: np.ndarray) -> Tuple[bool, str]:
        brightness = float(np.mean(rgb_array)) / 255.0
        if brightness < 0.05:
            return True, "Image is too dark to analyse"
        if brightness > 0.98:
            return True, "Image is overexposed"
        if float(np.std(rgb_array)) < 5.0:
            return True, "Image appears blank or corrupted"
        return False, "Basic checks passed"


class ModelBasedOODDetector:

    def __init__(self):
        self.min_confidence = 0.48
        self.max_entropy    = 0.76

    def check_prediction(self, probs: np.ndarray,
                          disease: str,
                          confidence: float) -> Tuple[bool, str]:
        entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
        print(f"   [OOD] confidence={confidence:.3f}, entropy={entropy:.3f}")

        if confidence < self.min_confidence:
            return True, (
                "System is unable to identify the image. Please upload a clearer image."
            )
        if entropy > self.max_entropy:
            return True, (
                "System is unable to identify the image. Please upload a clearer image."
            )
        return False, f"Valid: {disease} ({confidence:.1%})"

    def get_confidence_level(self, confidence: float) -> str:
        if confidence >= 0.85:   return "High"
        elif confidence >= 0.70: return "Moderate"
        else:                    return "Low"


# Instantiate globally
preprocessing_ood = EnhancedOODDetector()
model_ood         = ModelBasedOODDetector()


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


# ---------------------------------------------------------------
# HDF5 STATE DICT READER
# ---------------------------------------------------------------

def read_hdf5_state_dict(h5_group) -> dict:
    """Recursively read nested HDF5 groups into a flat PyTorch state_dict."""
    result = {}
    for key, item in h5_group.items():
        if isinstance(item, h5py.Group):
            sub = read_hdf5_state_dict(item)
            for subkey, val in sub.items():
                result[f"{key}.{subkey}"] = val
        else:
            arr = np.array(item)
            result[key] = torch.tensor(arr.astype(np.float32))
    return result


# ---------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------

def load_resources():
    global disease_model, severity_model, severity_thresholds
    global herb_rf_model, label_encoders, mlb_herbs, treatment_database

    print("\n[LOADING MODELS]")

    # ── 1. Disease model (Keras / TensorFlow) ──────────────────
    disease_model = keras.models.load_model(
        str(BASE_DIR / 'best_production_model.h5'), compile=False)
    _ = disease_model.predict(
        np.zeros((1,) + DISEASE_MODEL_SIZE + (3,), dtype=np.float32), verbose=0)
    print("disease_model loaded")

    # ── 2. Severity model (PyTorch EfficientNetV2-M + CORAL) ───
    sev_path = str(BASE_DIR / 'severity_model_pytorch.h5')

    with h5py.File(sev_path, 'r') as f:
        metadata   = json.loads(f.attrs['metadata_json'])
        state_dict = read_hdf5_state_dict(f['state_dict'])

    print(f"   Loaded {len(state_dict)} tensors — "
          f"sample keys: {list(state_dict.keys())[:3]}")

    dropout_val         = float(metadata.get('dropout', 0.35))
    num_classes         = int(metadata.get('num_classes', 3))
    severity_thresholds = metadata.get(
        'ordinal_thresholds', [0.170583575963974, 1.3933634757995605])

    # Rebuild EfficientNetV2-M with CoralOrdinal head.
    # Saved keys:
    #   classifier.1.fc.weight  shape (1, 1280)
    #   classifier.1.bias       shape (2,)
    base        = tv_models.efficientnet_v2_m(weights=None)
    in_features = base.classifier[1].in_features          # 1280
    base.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=dropout_val),
        CoralOrdinal(in_features, num_classes)
    )
    base.load_state_dict(state_dict)
    base.eval()
    severity_model = base
    print("severity_model loaded")

    # ── 3. Herb RF model ───────────────────────────────────────
    herb_rf_model = joblib.load(str(BASE_DIR / 'herb_recommendation_rf_model.pkl'))
    print("herb_rf_model loaded")

    # ── 4. Label encoders ──────────────────────────────────────
    with open(BASE_DIR / 'label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    print("label_encoders loaded:", list(label_encoders.keys()))

    # ── 5. Multi-label binarizer ───────────────────────────────
    with open(BASE_DIR / 'mlb_herbs.pkl', 'rb') as f:
        mlb_herbs = pickle.load(f)
    print(f"mlb_herbs loaded — {len(mlb_herbs.classes_)} herbs: {list(mlb_herbs.classes_)}")

    # ── 6. Treatment database ──────────────────────────────────
    treatment_database = pd.read_excel(str(BASE_DIR / DATABASE_FILE))
    treatment_database.columns = treatment_database.columns.str.strip().str.lower()
    print(f"treatment_database loaded — {len(treatment_database)} rows")


# ---------------------------------------------------------------
# STEP 1 — DISEASE CLASSIFICATION
# ---------------------------------------------------------------

def predict_disease_with_improved_ood(img_array: np.ndarray,
                                       preprocessed_rgb: np.ndarray) -> dict:
    # Stage 1: minimal technical check
    is_bad, reason = preprocessing_ood.detect_non_skin_image(preprocessed_rgb)
    if is_bad:
        print(f"[STAGE 1 REJECT] {reason}")
        return {
            'success':        False,
            'is_ood':         True,
            'ood_stage':      'preprocessing',
            'error':          f'Image rejected: {reason}',
            'suggestion':     'Please upload a clear photo of a skin condition.',
            'rejection_type': 'unusable_image'
        }
    print(f"[STAGE 1 PASS] {reason}")

    # Stage 2: model prediction
    probs      = disease_model.predict(img_array, verbose=0)[0]
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    disease    = CLASS_NAMES[pred_idx]
    print(f"   Prediction: {disease} ({confidence:.1%})")
    print(f"   All probs: acne={probs[0]:.3f}, eczema={probs[1]:.3f}, ringworm={probs[2]:.3f}")

    # Stage 2: confidence/entropy check
    is_ood, ood_reason = model_ood.check_prediction(probs, disease, confidence)
    if is_ood:
        print(f"[STAGE 2 REJECT] {ood_reason}")
        return {
            'success':        False,
            'is_ood':         True,
            'ood_stage':      'model',
            'error':          ood_reason,
            'suggestion':     'Please upload a clear, well-lit photo of the affected skin area.',
            'rejection_type': 'low_confidence'
        }
    print(f"[STAGE 2 PASS] {ood_reason}")

    return {
        'success':          True,
        'is_ood':           False,
        'disease':          disease,
        'confidence':       confidence * 100,
        'confidence_level': model_ood.get_confidence_level(confidence),
        'probabilities':    {
            'acne':     float(probs[0]),
            'eczema':   float(probs[1]),
            'ringworm': float(probs[2])
        }
    }


# ---------------------------------------------------------------
# STEP 2 — SEVERITY ASSESSMENT
# ---------------------------------------------------------------

def predict_severity(severity_img_array: np.ndarray,
                     disease: str,
                     confidence: float) -> str:
    if severity_model is not None:
        try:
            # ImageNet normalisation (matches training)
            mean_val = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std_val  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            img    = severity_img_array[0].astype(np.float32) / 255.0
            img    = (img - mean_val) / std_val

            # HWC -> CHW, add batch dim
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                # CoralOrdinal output: raw logits shape [1, 2]
                logits    = severity_model(tensor)[0].cpu()       # shape [2]
                coral_out = torch.sigmoid(logits).numpy()         # [2] probs
                p0        = float(coral_out[0])
                p1        = float(coral_out[1])

            # Standard CORAL 3-class conversion:
            # P(Mild)     = 1 - p0
            # P(Moderate) = p0 - p1
            # P(Severe)   = p1
            class_probs = np.array([
                1.0 - p0,
                p0 - p1,
                p1
            ], dtype=np.float32)
            class_probs = np.clip(class_probs, 0.0, 1.0)

            # Ordinal score -> threshold lookup
            score = float(class_probs @ np.arange(3, dtype=np.float32))
            t     = severity_thresholds          # [0.1705, 1.3933]
            if score < t[0]:
                pred_idx = 0
            elif score < t[1]:
                pred_idx = 1
            else:
                pred_idx = 2

            severity_map = ['mild', 'moderate', 'severe']
            result       = severity_map[pred_idx]
            print(f"   [SEVERITY] {result} | p0={p0:.3f} p1={p1:.3f} | "
                  f"score={score:.3f} | probs={class_probs.round(3)}")
            return result

        except Exception as e:
            print(f"Severity model error: {e}")
            import traceback
            traceback.print_exc()

    # Fallback based on disease-model confidence
    if confidence >= 0.85:   return "mild"
    elif confidence >= 0.70: return "moderate"
    else:                    return "severe"


# ---------------------------------------------------------------
# STEP 3 — HERB RECOMMENDATION via RF MODEL
# ---------------------------------------------------------------

def recommend_herbs_ranked(disease: str, severity: str,
                            age: int, skin_type: str,
                            skin_sensitivity: str) -> Tuple[List[str], List[float]]:
    if herb_rf_model is None or label_encoders is None or mlb_herbs is None:
        return [], []

    try:
        age_group     = map_age_to_group(age)
        skin_oiliness = map_skin_type(skin_type)
        skin_sens     = map_sensitivity(skin_sensitivity)

        input_df = pd.DataFrame([{
            'disease':          disease.lower().strip(),
            'severity':         severity.lower().strip(),
            'age_group':        age_group,
            'skin_oiliness':    skin_oiliness,
            'skin_sensitivity': skin_sens
        }])

        print(f"   [RF] disease={disease}, severity={severity}, "
              f"age_group={age_group}, oiliness={skin_oiliness}, sensitivity={skin_sens}")

        for col in ['disease', 'severity', 'age_group', 'skin_oiliness', 'skin_sensitivity']:
            try:
                input_df[col] = label_encoders[col].transform(input_df[col])
            except ValueError:
                fallback = label_encoders[col].classes_[0]
                print(f"   [RF WARN] Unknown value for '{col}', using '{fallback}'")
                input_df[col] = label_encoders[col].transform([fallback])

        herb_probs = np.array([
            est.predict_proba(input_df)[0][1]
            for est in herb_rf_model.estimators_
        ])

        sorted_idx   = np.argsort(herb_probs)[::-1]
        ranked_herbs = np.array(mlb_herbs.classes_)[sorted_idx].tolist()
        ranked_probs = herb_probs[sorted_idx].tolist()

        THRESHOLD = 0.25
        filtered = [(h, p) for h, p in zip(ranked_herbs, ranked_probs) if p >= THRESHOLD]
        if len(filtered) < 3:
            filtered = list(zip(ranked_herbs[:3], ranked_probs[:3]))

        result_herbs = [h for h, p in filtered]
        result_probs = [p for h, p in filtered]

        print(f"   [RF] Ranked: {[(h, round(p, 2)) for h, p in zip(result_herbs, result_probs)]}")
        return result_herbs, result_probs

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"   [RF ERROR] {e}")
        return [], []


# ---------------------------------------------------------------
# STEP 4 — TREATMENT LOOKUP
# ---------------------------------------------------------------

def get_treatment(disease: str, severity: str,
                  age: int, skin_type: str, skin_sensitivity: str,
                  ranked_herbs: List[str]) -> Dict:
    if treatment_database is None or not ranked_herbs:
        return _fallback_treatment(disease, severity, ranked_herbs)

    age_group     = map_age_to_group(age)
    skin_oiliness = map_skin_type(skin_type)
    skin_sens     = map_sensitivity(skin_sensitivity)
    db            = treatment_database
    base          = (
        (db['disease'].str.lower()  == disease.lower()) &
        (db['severity'].str.lower() == severity.lower())
    )

    for herb in ranked_herbs:
        h_low     = herb.lower()
        herb_mask = base & (db['herb_name_english'].str.lower() == h_low)

        # 1. Full profile match
        subset = db[
            herb_mask &
            (db['age_group'].str.lower()        == age_group) &
            (db['skin_oiliness'].str.lower()    == skin_oiliness) &
            (db['skin_sensitivity'].str.lower() == skin_sens)
        ]
        # 2. Age group only
        if subset.empty:
            subset = db[herb_mask & (db['age_group'].str.lower() == age_group)]
        # 3. Herb only
        if subset.empty:
            subset = db[herb_mask]

        if not subset.empty:
            record = subset.iloc[0]
            print(f"   [DB] Matched: herb={herb}, "
                  f"age={record.get('age_group', '?')}, "
                  f"oiliness={record.get('skin_oiliness', '?')}, "
                  f"sensitivity={record.get('skin_sensitivity', '?')}")
            return _build_treatment_dict(record, herb, ranked_herbs)

    # 4. First available row for disease/severity
    fallback_rows = db[base]
    if not fallback_rows.empty:
        record = fallback_rows.iloc[0]
        herb   = str(record.get('herb_name_english', ranked_herbs[0]))
        print(f"   [DB] Last resort row: herb={herb}")
        return _build_treatment_dict(record, herb, ranked_herbs)

    return _fallback_treatment(disease, severity, ranked_herbs)


def _build_treatment_dict(record, selected_herb: str,
                           ranked_herbs: List[str]) -> Dict:
    def val(key):
        v = record.get(key, 'N/A')
        return str(v) if pd.notna(v) else 'N/A'

    return {
        'selected_herb':      selected_herb,
        'herb_english':       val('herb_name_english'),
        'herb_sinhala':       val('herb_name_sinhala'),
        'herb_scientific':    val('herb_botanical_name'),
        'herb_part_used':     val('herb_part_used'),
        'preparation_type':   val('preparation_type'),
        'preparation':        val('preparation_method'),
        'application':        val('application_instructions'),
        'duration':           val('treatment_duration'),
        'frequency':          val('frequency_per_day'),
        'dosage':             f"{val('dosage_g')}g / {val('dosage_ml')}ml",
        'storage':            val('storage_instructions'),
        'precautions':        val('contraindications'),
        'side_effects':       val('possible_side_effects'),
        'when_to_stop':       val('when_to_stop'),
        'when_to_see_doctor': val('when_to_see_doctor'),
        'patch_test':         val('patch_test_required'),
        'recommended_herbs':  ', '.join(ranked_herbs)
    }


def _fallback_treatment(disease: str, severity: str, herbs: List[str]) -> Dict:
    treatments = {
        'acne': {
            'herb_english':    'Neem',
            'herb_sinhala':    'kohomba',
            'herb_scientific': 'Azadirachta indica',
            'preparation':     'Crush 10-15 fresh neem leaves into a paste.',
            'application':     'Apply twice daily for 15-20 minutes, rinse with lukewarm water.',
            'duration':        '4-6 weeks',
            'precautions':     'Patch test before use.'
        },
        'eczema': {
            'herb_english':    'Coconut Oil',
            'herb_sinhala':    'pol thel',
            'herb_scientific': 'Cocos nucifera',
            'preparation':     'Use pure virgin coconut oil.',
            'application':     'Massage gently onto affected areas 2-3 times daily.',
            'duration':        '6-8 weeks',
            'precautions':     'Avoid hot showers.'
        },
        'ringworm': {
            'herb_english':    'Neem Oil',
            'herb_sinhala':    'kohomba thel',
            'herb_scientific': 'Azadirachta indica',
            'preparation':     'Use pure neem oil or dilute with a carrier oil.',
            'application':     'Apply 2-3 times daily to the affected area.',
            'duration':        '3-4 weeks',
            'precautions':     'Keep the area clean and dry. Wash hands after application.'
        }
    }
    base = treatments.get(disease.lower(), treatments['acne']).copy()
    base['selected_herb']     = base['herb_english']
    base['recommended_herbs'] = ', '.join(herbs) if herbs else 'N/A'
    for k in ['herb_part_used', 'preparation_type', 'frequency', 'dosage',
              'storage', 'side_effects', 'when_to_stop', 'when_to_see_doctor', 'patch_test']:
        base.setdefault(k, 'N/A')
    return base


# ---------------------------------------------------------------
# INTEGRATED PIPELINE
# ---------------------------------------------------------------

def integrated_pipeline(file_stream, patient_data: Dict) -> Dict:
    img_pil          = Image.open(file_stream)
    preprocessed_rgb = preprocess_image(img_pil)
    disease_input    = prepare_for_disease_model(preprocessed_rgb)

    # Step 1: disease classification + OOD
    result = predict_disease_with_improved_ood(disease_input, preprocessed_rgb)
    if result.get('is_ood'):
        return result

    disease    = result['disease']
    confidence = result['confidence'] / 100

    # Step 2: severity
    severity_input = prepare_for_severity_model(preprocessed_rgb)
    severity       = predict_severity(severity_input, disease, confidence)

    # Step 3: RF model ranks herbs for this patient
    age              = int(patient_data.get('age', 25))
    skin_type        = patient_data.get('skinType', 'normal')
    skin_sensitivity = patient_data.get('skinSensitivity', 'normal')

    ranked_herbs, herb_probs = recommend_herbs_ranked(
        disease, severity, age, skin_type, skin_sensitivity
    )

    # Step 4: fetch matching DB row
    treatment = get_treatment(
        disease, severity, age, skin_type, skin_sensitivity, ranked_herbs
    )

    return {
        'success':            True,
        'disease':            disease,
        'confidence':         confidence * 100,
        'confidence_level':   result.get('confidence_level', 'Moderate'),
        'severity':           severity,
        'is_ood':             False,
        'probabilities':      result['probabilities'],
        'recommended_herbs':  ranked_herbs,
        'herb_probabilities': [round(p, 3) for p in herb_probs],
        'treatment':          treatment,
        'patient_data':       patient_data
    }


# ---------------------------------------------------------------
# FLASK ROUTES
# ---------------------------------------------------------------

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/diagnosis')
def diagnosis():
    return render_template('diagnosis.html')


@app.route('/results')
def results():
    return render_template('results.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/guidelines')
def guidelines():
    return render_template('guidelines.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    patient_data = {
        'age':             request.form.get('age', '25'),
        'skinType':        request.form.get('skinType', 'normal'),
        'skinSensitivity': request.form.get('skinSensitivity', 'normal')
    }

    try:
        result = integrated_pipeline(file.stream, patient_data)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Analysis failed. Please try again.'}), 500


# ---------------------------------------------------------------
# PDF DOWNLOAD ROUTE
# ---------------------------------------------------------------

from pdf_generator import generate_diagnosis_pdf


@app.route('/api/download-pdf', methods=['POST'])
def download_pdf():
    """
    generates a formatted PDF report and streams it as a download.
    """
    try:
        result_data = request.get_json(force=True)

        if not result_data or not result_data.get('disease'):
            return jsonify({'success': False,
                            'error': 'No diagnosis data provided'}), 400

        pdf_bytes = generate_diagnosis_pdf(result_data)

        # Build a safe filename from disease name
        disease_slug = str(result_data.get('disease', 'report')).lower().replace(' ', '_')
        filename = f"ayurderma_{disease_slug}_report.pdf"

        response = make_response(pdf_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        response.headers['Content-Length'] = len(pdf_bytes)
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False,
                        'error': f'PDF generation failed: {str(e)}'}), 500


@app.errorhandler(404)
def not_found(e):
    return redirect(url_for('home'))



# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print(" " * 20 + "SYSTEM INITIALIZATION")
    print("=" * 70)

    if not TEMPLATE_DIR.exists():
        print(f"Template folder not found: {TEMPLATE_DIR}")
        sys.exit(1)

    try:
        load_resources()
    except Exception as e:
        print(f"\nFailed to load resources: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 70)
    print("SYSTEM READY")
    print("=" * 70)
    print("\nOpen: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
