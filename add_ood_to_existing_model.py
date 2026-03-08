"""
OOD Calibration - Softmax Confidence + Entropy Only
====================================================
WHY WE CHANGED APPROACH:
  Your model maps ALL images (dogs, cars, skin) into a compact Euclidean
  cluster because it uses ImageNet-pretrained weights. Distance-based OOD
  cannot separate them — a dog scores 2.9 and acne scores 3.2.

  The only reliable signal is SOFTMAX OUTPUT:
  - In-distribution images  → high confidence (your val mean = 75.7%)
  - OOD images              → uncertain, spread across classes (low max prob)

  We use TWO combined signals:
    1. Max softmax probability  (confidence)
    2. Prediction entropy       (uncertainty)

  Both thresholds are calibrated from YOUR validation data.

REQUIREMENTS:
  - Just your existing model, no OOD images needed.
  - Run this once, then run app.py.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ================== CONFIG ==================
MODEL_PATH  = 'best_production_model.h5'
VAL_DIR     = r"C:\Users\asus\Downloads\cleaned_dataset\validation"  # <- change if needed
IMG_SIZE    = (380, 380)
NUM_CLASSES = 3

# Percentile of in-distribution scores used as threshold.
# Lower = stricter (rejects more). Start at 5 = only reject if confidence
# is in the bottom 5% of your validation set.
CONFIDENCE_PERCENTILE = 10   # reject if confidence < P10 of val set
ENTROPY_PERCENTILE    = 90   # reject if entropy   > P90 of val set

print("=" * 60)
print("  OOD Calibration — Softmax Confidence + Entropy")
print("=" * 60)

# Load model
print("\n[1/3] Loading model...")
model = keras.models.load_model(MODEL_PATH, compile=False)
dummy = np.zeros((1,) + IMG_SIZE + (3,), dtype=np.float32)
_ = model.predict(dummy, verbose=0)
print("      Done")

# Extract softmax probabilities from validation set
print("\n[2/3] Running model on validation set...")
val_gen = ImageDataGenerator().flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=16,
    class_mode='categorical', shuffle=False
)

all_probs = []
collected = 0
for imgs, _ in val_gen:
    all_probs.append(model.predict(imgs, verbose=0))
    collected += len(imgs)
    print(f"      {collected}/{val_gen.samples}...", end='\r')
    if collected >= val_gen.samples:
        break

all_probs = np.concatenate(all_probs)[:val_gen.samples]
print(f"\n      {len(all_probs)} predictions done")

idx_to_class = {v: k for k, v in val_gen.class_indices.items()}
class_names  = [idx_to_class[i] for i in range(NUM_CLASSES)]
print(f"      Classes: {class_names}")

# Compute confidence and entropy for every val sample
max_probs = all_probs.max(axis=1)
entropy   = -np.sum(all_probs * np.log(all_probs + 1e-8), axis=1)
max_entropy = float(np.log(NUM_CLASSES))  # theoretical max

print(f"\n[3/3] Computing thresholds...")
print(f"\n      CONFIDENCE stats (higher = more certain):")
print(f"        Mean : {max_probs.mean():.4f}")
print(f"        Min  : {max_probs.min():.4f}")
print(f"        P5   : {np.percentile(max_probs, 5):.4f}")
print(f"        P10  : {np.percentile(max_probs, 10):.4f}")
print(f"        P25  : {np.percentile(max_probs, 25):.4f}")

print(f"\n      ENTROPY stats (lower = more certain, max={max_entropy:.3f}):")
print(f"        Mean : {entropy.mean():.4f}")
print(f"        Max  : {entropy.max():.4f}")
print(f"        P75  : {np.percentile(entropy, 75):.4f}")
print(f"        P90  : {np.percentile(entropy, 90):.4f}")
print(f"        P95  : {np.percentile(entropy, 95):.4f}")

conf_threshold    = float(np.percentile(max_probs, CONFIDENCE_PERCENTILE))
entropy_threshold = float(np.percentile(entropy,   ENTROPY_PERCENTILE))

print(f"\n      Confidence threshold (P{CONFIDENCE_PERCENTILE}) : {conf_threshold:.4f}")
print(f"      Entropy    threshold (P{ENTROPY_PERCENTILE}) : {entropy_threshold:.4f}")
print(f"\n      An OOD image (dog, car, building) will typically have:")
print(f"        confidence ~ 0.33-0.40  (vs your val mean {max_probs.mean():.2f})")
print(f"        entropy    ~ 1.0+       (vs your val mean {entropy.mean():.2f})")

# Save
np.savez(
    'ood_params.npz',
    conf_threshold=np.array([conf_threshold]),
    entropy_threshold=np.array([entropy_threshold]),
    class_names=np.array(class_names),
    max_entropy=np.array([max_entropy])
)

print(f"\n      Saved -> ood_params.npz")
print("\n" + "=" * 60)
print("  Calibration complete!")
print(f"    Reject if confidence < {conf_threshold:.4f}  (P{CONFIDENCE_PERCENTILE})")
print(f"    Reject if entropy    > {entropy_threshold:.4f}  (P{ENTROPY_PERCENTILE})")
print(f"\n  TUNING (edit values at top of this file and re-run):")
print(f"    OOD images accepted  -> lower CONFIDENCE_PERCENTILE or")
print(f"                            lower ENTROPY_PERCENTILE")
print(f"    Valid images rejected -> raise CONFIDENCE_PERCENTILE or")
print(f"                            raise ENTROPY_PERCENTILE")
print("=" * 60)