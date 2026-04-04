# AyurDerma — Disease Classification & System Integration

## My Contribution

This branch covers two main responsibilities in the AyurDerma project:
1. **Skin Disease Classification Model** — training and evaluating the deep learning model
2. **Full System Integration** — connecting all models into a working Flask web application

---

## Disease Classification Model

The classification model identifies three skin conditions from a photo:
- Acne
- Eczema
- Ringworm

### Architecture
- **Base Model:** EfficientNetB4 with ImageNet pre-trained weights (Transfer Learning)
- **Approach:** Fine-tuned the top layers on a curated skin disease dataset
- **Framework:** Keras / TensorFlow
- **Input Size:** 380 × 380 px
- **Output:** Softmax over 3 classes

### Preprocessing Pipeline
Before inference, each image goes through a custom preprocessing pipeline:
- Center crop (95%)
- Gray world white balance
- NL-means denoising
- CLAHE contrast enhancement
- Gamma correction
- Unsharp masking

### Out-of-Distribution (OOD) Detection
To prevent the model from making predictions on irrelevant images, two OOD checks are applied:
- **Stage 1 — Image quality check:** Rejects images that are too dark, overexposed, or blank
- **Stage 2 — Confidence/entropy check:** Rejects predictions with low confidence (< 0.48) or high entropy (> 0.76)

### Results
| Metric | Value |
|---|---|
| Accuracy | **87.56%** |
| Classes | Acne, Eczema, Ringworm |
| Framework | Keras / TensorFlow |

---

## System Integration

The Flask application (`app_with_herb_recommendation_new.py`) integrates all three models into a single end-to-end pipeline:

```
Image Upload → Disease Classification → Severity Assessment → Herb Recommendation → Treatment Lookup
```

### Integration Responsibilities
- Built all Flask routes (`/`, `/diagnosis`, `/results`, `/about`, `/guidelines`)
- Implemented the `/api/predict` endpoint that runs the full pipeline
- Connected the disease model, severity model (PyTorch) and herb recommendation model (Scikit-learn) into one unified flow
- Handled model loading, error handling, and OOD rejection responses
- Implemented the `/api/download-pdf` endpoint for generating downloadable diagnosis reports
- Built all frontend HTML templates (home, diagnosis, results, about, guidelines)

### PDF Report Generation
Added a PDF download feature using ReportLab that generates a formatted diagnosis report containing:
- Detected condition, severity and confidence
- Patient profile
- Full Ayurvedic treatment details
- Medical disclaimer

---

## Tech Stack

| Component | Technology |
|---|---|
| Web Framework | Flask |
| Classification Model | EfficientNetB4 (Keras/TensorFlow) |
| PDF Generation | ReportLab |
| Frontend | HTML, CSS, JavaScript |

---

## How to Run

```bash
pip install -r requirements.txt
python app_with_herb_recommendation_new.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

*Sasandi Abeywickrama*