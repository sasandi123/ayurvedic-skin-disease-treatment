# Skin Disease Severity Classification (Ordinal Deep Learning)

This project implements a state-of-the-art deep learning system for classifying skin disease images into three severity levels: **Mild**, **Moderate**, and **Severe**. It uses a rank-consistent ordinal regression (CORAL) approach to respect the natural ordering of severity.

## Overview
- **Architecture**: EfficientNet-V2 (Medium/Small)
- **Frameworks**: PyTorch, Torchvision, Flask
- **Classification Strategy**: **Ordinal Regression (CORAL)** — Treats the problem as a sequence (Mild < Moderate < Severe) for better boundary precision.
- **Key Features**:
    - **Data Cleaning**: Automatically detects and handles duplicate images or conflicting labels.
    - **Square Padding**: Preserves the full image context without distorted stretching or aggressive cropping.
    - **TTA (Test-Time Augmentation)**: Averages predictions from multiple views (Flip, Zoom) for robust inference.
    - **EMA (Exponential Moving Average)**: Uses smoothed model weights for superior generalization.
    - **HDF5 Checkpoints**: Portable and metadata-rich model storage (`.h5`).

## Project Structure
- `train.py`: Advanced training script with duplicate cleaning, ordinal loss, and OneCycleLR.
- `app.py`: Flask-based web application for real-time analysis with confidence bars.
- `Mild/`, `Moderate/`, `Severe/`: Dataset directories sorted by severity.
- `severity_model_pytorch.h5`: Trained model weights and metadata.
- `requirements.txt`: Python dependencies.

## Installation & Setup

### 1. Environment Setup
Create a virtual environment and activate it:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/macOS
```

### 2. Install Dependencies
Install the required libraries:
```bash
pip install -r requirements.txt
```

**Note for PyTorch CUDA Support:**
For systems with an NVIDIA GPU, install the CUDA-enabled version of PyTorch for significantly faster training:
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
Example (CUDA 12.1):
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

## Usage

### Training
Run this from the VS Code terminal in the project root:
```powershell
.\venv\Scripts\python.exe -u train.py --batch-size 8 --accumulation-steps 2 --num-workers 2 --head-epochs 3 --finetune-epochs 18 --patience 6 --img-size 320
```

What this command does:
1. Scans the dataset and removes any exact duplicate conflicts during the training pipeline.
2. Runs Phase 1 head warmup for `3` epochs.
3. Runs Phase 2 full fine-tuning for up to `18` epochs with early stopping patience of `6`.
4. Saves the best checkpoint as `severity_model_pytorch.h5`.
5. Writes the latest metrics to `training_metrics.json`.
6. Updates the training curve image `training_curves.png`.

If you also want the full console output saved to log files:
```powershell
.\venv\Scripts\python.exe -u train.py --batch-size 8 --accumulation-steps 2 --num-workers 2 --head-epochs 3 --finetune-epochs 18 --patience 6 --img-size 320 1> training_manual.log 2> training_manual.err
```

Use this version when you want to inspect the raw training output later without keeping the terminal open.

### Report Generation
After training finishes, generate the visual evaluation reports with:
```powershell
.\venv\Scripts\python.exe generate_reports.py
```

This command rebuilds the main report artifacts from the latest saved checkpoint, including:
- `training_curves.png`
- `dataset_distribution.png`
- `confusion_matrix_val_tta.png`
- `confusion_matrix_test_tta.png`
- `per_class_metrics_test_tta.png`
- `misclassified_examples_test_tta.png`
- `test_predictions_tta.csv`

### Inference (Web App)
To start the classification server:
```bash
python app.py
```
Open `http://localhost:5000` in your browser. Upload an image to see the detected severity and the probability distribution across all classes.

## Current Performance

Latest saved checkpoint: `severity_model_pytorch.h5`

Best checkpoint results on the current cleaned and relabeled dataset:
- **Best Epoch**: `19`
- **Validation Accuracy (single)**: `75.78%`
- **Validation Accuracy (TTA)**: `75.99%`
- **Validation QWK (TTA)**: `0.7674`
- **Test Accuracy (single)**: `74.92%`
- **Test Accuracy (TTA)**: `75.72%`
- **Test QWK (TTA)**: `0.7373`

Dataset state used for this run:
- **Total Images**: `6219`
- **Exact Duplicate Images Remaining**: `0`
- **Cross-label Exact Duplicates Remaining**: `0`
- **Class Counts**:
  - `Mild`: `1695`
  - `Moderate`: `2561`
  - `Severe`: `1963`

## Predictions

The latest `app.py` was tested with the current `.h5` checkpoint and loaded successfully on CUDA.

Verified app behavior:
- Model load status: `OK`
- Device used by the app: `cuda`
- Flask upload route test: `200 OK`
- Test-time augmentation used in the app: `Yes`

The app was evaluated on the held-out test split used by the training pipeline (`622` images). There is no separate top-level `test/` folder in the project root, so this review uses the saved split logic from `train.py`.

Exact `app.py` inference-path performance on the held-out test split:
- **Accuracy**: `75.40%`
- **Quadratic Weighted Kappa**: `0.7367`
- **Correct Predictions**: `469 / 622`
- **Incorrect Predictions**: `153 / 622`
- **Average Prediction Confidence**: `67.72%`
- **Median Prediction Confidence**: `63.18%`
- **Average Confidence on Correct Predictions**: `70.92%`
- **Average Confidence on Incorrect Predictions**: `57.92%`

Per-class app inference performance:
- **Mild**: precision `78.17%`, recall `65.29%`, F1 `71.15%`
- **Moderate**: precision `69.55%`, recall `78.52%`, F1 `73.76%`
- **Severe**: precision `82.20%`, recall `80.10%`, F1 `81.14%`

Review summary:
- `Severe` is the strongest class.
- The main remaining confusion is `Mild` vs `Moderate`.
- The app-path accuracy is slightly below the checkpoint evaluator, so matching app preprocessing exactly with training-time evaluation can still improve consistency.

Saved prediction review files:
- `app_http_test_review.json`
- `app_http_test_predictions.csv`

## Technical Details
- **Training Resolution**: 320x320 with `SquarePad` preprocessing.
- **Augmentation**: RandAugment, ColorJitter, RandomErasing, and RandomRotation.
- **Loss**: Binary Cross Entropy with Logits (per-rank thresholding).
- **Checkpoint Format**: HDF5 (`.h5`) with model metadata for reproducible loading in `app.py`.
