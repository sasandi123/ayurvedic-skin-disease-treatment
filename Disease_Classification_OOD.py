import os, json, hashlib, shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.covariance import EmpiricalCovariance

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint, Callback)
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = r"C:\Users\asus\Downloads\balanced_3diseases_dataset"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
CLASSES    = ['acne', 'eczema', 'ringworm']

EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 40

# OOD thresholds (all auto-calibrated on validation set)
MC_DROPOUT_PASSES        = 20
OOD_CONFIDENCE_THRESHOLD = 0.55
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
class DataCleaner:
    def __init__(self, data_dir, classes):
        self.data_dir    = Path(data_dir)
        self.classes     = classes
        self.cleaned_dir = self.data_dir.parent / 'cleaned_dataset'

    def _hash(self, path):
        img = cv2.imread(str(path))
        if img is None: return None
        return hashlib.md5(cv2.resize(img, (64, 64)).tobytes()).hexdigest()

    def clean(self):
        print("\n" + "="*70)
        print("STEP 1 – DATA CLEANING")
        print("="*70)

        splits = {
            'train':      self.data_dir / 'train',
            'validation': self.data_dir / 'validation',
            'test':       self.data_dir / 'test',
        }
        bucket = defaultdict(list)
        total, bad = 0, 0

        for split, sdir in splits.items():
            print(f"  Scanning {split}…")
            for cls in self.classes:
                cdir = sdir / cls
                if not cdir.exists():
                    print(f"  Missing: {cdir}"); continue
                for p in cdir.glob('*.*'):
                    if p.suffix.lower() not in {'.jpg','.jpeg','.png'}: continue
                    total += 1
                    h = self._hash(p)
                    if h is None: bad += 1; continue
                    bucket[split].append({'path': p, 'hash': h, 'class': cls})

        print(f"\n  Found {total} images | {bad} corrupt")

        if self.cleaned_dir.exists():
            shutil.rmtree(self.cleaned_dir)
        for sp in ['train','validation','test']:
            for cls in self.classes:
                (self.cleaned_dir / sp / cls).mkdir(parents=True, exist_ok=True)

        seen, dups, copied = set(), 0, 0
        counts = {sp: {c: 0 for c in self.classes} for sp in splits}
        for sp in ['train','validation','test']:
            for info in bucket[sp]:
                if info['hash'] in seen: dups += 1; continue
                seen.add(info['hash'])
                dest = self.cleaned_dir / sp / info['class'] / info['path'].name
                shutil.copy2(info['path'], dest)
                counts[sp][info['class']] += 1
                copied += 1

        print(f"  Removed {dups} duplicates → {copied} unique images\n")
        for sp in ['train','validation','test']:
            print(f"  {sp.capitalize()}:")
            for cls in self.classes:
                print(f"    {cls:10s}: {counts[sp][cls]:4d}")
        return self.cleaned_dir, counts


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATORS
# ══════════════════════════════════════════════════════════════════════════════
def make_generators(cleaned_dir, img_size, batch_size, classes):
    print("\n" + "="*70)
    print("STEP 2 – DATA GENERATORS")
    print("="*70)

    train_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.20,
        height_shift_range=0.20,
        shear_range=0.20,
        zoom_range=0.20,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    eval_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

    common = dict(
        target_size=img_size, batch_size=batch_size,
        class_mode='categorical', classes=classes
    )
    train_gen = train_aug.flow_from_directory(
        str(cleaned_dir/'train'), shuffle=True, seed=42, **common)
    val_gen   = eval_aug.flow_from_directory(
        str(cleaned_dir/'validation'), shuffle=False, **common)
    test_gen  = eval_aug.flow_from_directory(
        str(cleaned_dir/'test'), shuffle=False, **common)

    assert train_gen.class_indices == {c: i for i,c in enumerate(classes)}, \
        f"Class order wrong: {train_gen.class_indices}"

    mapping = {'classes': classes, 'class_indices': train_gen.class_indices}
    with open('class_mapping.json','w') as f: json.dump(mapping, f, indent=2)
    print(f"  ✓ Class order: {train_gen.class_indices}")

    cnts  = Counter(train_gen.classes)
    total = sum(cnts.values())
    weights = {i: total/(len(cnts)*cnts[i]) for i in sorted(cnts)}
    print("  Weights:", {classes[k]: round(v,3) for k,v in weights.items()})
    print(f"  Train {train_gen.samples} | Val {val_gen.samples} | Test {test_gen.samples}")
    return train_gen, val_gen, test_gen, weights


# ══════════════════════════════════════════════════════════════════════════════
# MODEL  — returns BOTH full model AND a feature extractor
# ══════════════════════════════════════════════════════════════════════════════
def build_model(img_size, num_classes):
    print("\n" + "="*70)
    print("STEP 3 – MODEL")
    print("="*70)

    base = EfficientNetB4(include_top=False, weights='imagenet',
                          input_shape=(*img_size, 3))
    base.trainable = False

    inp = keras.Input(shape=(*img_size, 3))
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(0.005))(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.4)(x)
    # ── This is the feature layer we will use for OOD ─────────────────────────
    features = layers.Dense(256, activation='relu', name='feature_layer',
                             kernel_regularizer=regularizers.l2(0.005))(x)
    x   = layers.Dropout(0.3)(features)
    out = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model          = keras.Model(inp, out,      name='classifier')
    feature_model  = keras.Model(inp, features, name='feature_extractor')

    print(f"  Classifier  : input → EfficientNetB4 → GAP → Dense(512) "
          f"→ Dense(256)[feature_layer] → Dense({num_classes})")
    print(f"  Feature extractor : same model, output at 'feature_layer' (256-d)")
    return model, feature_model, base


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════
class TrainingMonitor(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        va, ta = logs.get('val_accuracy',0), logs.get('accuracy',0)
        if epoch==4 and va < 1/len(CLASSES)+0.05:
            print(f"\n  val_acc={va:.3f} barely above random – check data")
        if ta - va > 0.25:
            print(f"\n Overfitting gap: train={ta:.3f} val={va:.3f}")


def get_cbs(stage):
    ckpt = f'best_model_stage{stage}.keras'
    return [
        TrainingMonitor(),
        EarlyStopping(monitor='val_loss', patience=8 if stage==1 else 15,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=4,
                          min_lr=1e-7, verbose=1),
        ModelCheckpoint(ckpt, monitor='val_accuracy', save_best_only=True,
                        mode='max', verbose=1),
    ], ckpt


def train(model, base, train_gen, val_gen, weights):
    # Stage 1 – frozen
    print("\n" + "="*70)
    print("STAGE 1 – FROZEN BASE  (head warm-up)")
    print("="*70)
    model.compile(optimizer=Adam(1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    cbs, _ = get_cbs(1)
    h1 = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_STAGE1,
                   callbacks=cbs, class_weight=weights, verbose=1)
    best1 = max(h1.history['val_accuracy'])
    print(f"\n  Stage 1 best val_accuracy: {best1:.4f} ({best1*100:.1f}%)")

    # Stage 2 – fine-tune last 100 layers
    print("\n" + "="*70)
    print("STAGE 2 – FINE-TUNING  (last 100 layers)")
    print("="*70)
    base.trainable = True
    for layer in base.layers[:-100]:
        layer.trainable = False
    print(f"  Unfrozen {sum(1 for l in base.layers if l.trainable)} layers")
    model.compile(optimizer=Adam(1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    cbs2, best_ckpt = get_cbs(2)
    h2 = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_STAGE2,
                   callbacks=cbs2, class_weight=weights, verbose=1)
    best2 = max(h2.history['val_accuracy'])
    print(f"\n  Stage 2 best val_accuracy: {best2:.4f} ({best2*100:.1f}%)")
    return h1, h2, best_ckpt


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE-BASED OOD  (Mahalanobis Distance)
# ══════════════════════════════════════════════════════════════════════════════
class MahalanobisOOD:
    def __init__(self, num_classes, feature_dim=256):
        self.num_classes  = num_classes
        self.feature_dim  = feature_dim
        self.centroids    = None   # shape (num_classes, feature_dim)
        self.precision    = None   # shape (feature_dim, feature_dim)
        self.threshold    = None
        self.fitted       = False

    def fit(self, feature_model, train_gen, classes):
        """
        Extract features from ALL training images and compute:
         - Per-class centroid
         - Shared precision matrix (inverse covariance)
        """
        print("\n" + "="*70)
        print("MAHALANOBIS OOD – COMPUTING CLASS CENTROIDS")
        print("="*70)
        print("  Extracting features from training set…")

        # Reset generator to start
        train_gen.reset()

        class_features = {i: [] for i in range(self.num_classes)}
        total_batches  = len(train_gen)

        for batch_idx in range(total_batches):
            imgs, labels = train_gen[batch_idx]
            feats = feature_model.predict(imgs, verbose=0)
            for feat, lbl in zip(feats, labels):
                class_idx = int(np.argmax(lbl))
                class_features[class_idx].append(feat)

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx+1}/{total_batches}…")

        # Compute centroids
        self.centroids = np.zeros((self.num_classes, self.feature_dim))
        all_features   = []
        all_labels     = []

        print("\n  Class centroids:")
        for cls_idx, feats in class_features.items():
            feats_arr = np.array(feats)
            self.centroids[cls_idx] = feats_arr.mean(axis=0)
            all_features.append(feats_arr)
            all_labels.extend([cls_idx] * len(feats))
            print(f"    {classes[cls_idx]:10s}: {len(feats):4d} samples | "
                  f"centroid norm = {np.linalg.norm(self.centroids[cls_idx]):.2f}")

        # Compute shared precision matrix (inverse covariance)
        # Using all samples centered by their class mean
        print("\n  Computing shared precision matrix…")
        centered = []
        for cls_idx, feats in class_features.items():
            feats_arr = np.array(feats)
            centered.append(feats_arr - self.centroids[cls_idx])

        centered_all = np.vstack(centered)

        # Fit covariance and compute precision
        cov_estimator = EmpiricalCovariance(assume_centered=True)
        cov_estimator.fit(centered_all)
        self.precision = cov_estimator.precision_

        print(f"  Precision matrix shape: {self.precision.shape}")
        self.fitted = True
        print("  ✓ Mahalanobis OOD fitted")

    def mahalanobis_distance(self, feature_vector):
        """
        Compute Mahalanobis distance from feature_vector to each class centroid.
        Returns min distance (closest centroid = most likely class).
        """
        distances = []
        for centroid in self.centroids:
            diff = feature_vector - centroid
            # Mahalanobis = sqrt(diff^T * Precision * diff)
            dist = float(np.sqrt(diff @ self.precision @ diff))
            distances.append(dist)
        return distances, min(distances)

    def calibrate_threshold(self, feature_model, val_gen, percentile=95):
        """
        Compute OOD threshold from validation set (in-distribution data).
        Use the 95th percentile of min-distances as the threshold.
        Images farther than this are likely OOD.
        """
        print("\n  Calibrating Mahalanobis threshold on validation set…")
        val_gen.reset()

        min_distances = []
        for batch_idx in range(len(val_gen)):
            imgs, _ = val_gen[batch_idx]
            feats   = feature_model.predict(imgs, verbose=0)
            for feat in feats:
                _, min_d = self.mahalanobis_distance(feat)
                min_distances.append(min_d)

        min_distances = np.array(min_distances)
        self.threshold = float(np.percentile(min_distances, percentile))

        print(f"  Min-distance stats (validation):")
        print(f"    Mean  : {min_distances.mean():.4f}")
        print(f"    Std   : {min_distances.std():.4f}")
        print(f"    {percentile}th pct : {self.threshold:.4f}  ← OOD threshold")
        print(f"  (Images with min-distance > {self.threshold:.4f} → rejected as OOD)")

        return self.threshold

    def is_ood(self, feature_vector):
        """Return (is_ood: bool, min_distance: float, distances: list)"""
        if not self.fitted or self.threshold is None:
            return False, 0.0, []
        distances, min_d = self.mahalanobis_distance(feature_vector)
        return min_d > self.threshold, min_d, distances

    def save(self, path='mahalanobis_ood.npz'):
        """Save centroids, precision matrix, and threshold."""
        np.savez(path,
                 centroids=self.centroids,
                 precision=self.precision,
                 threshold=np.array([self.threshold]))
        print(f"  ✓ Mahalanobis OOD saved to {path}")

    @classmethod
    def load(cls, path='mahalanobis_ood.npz', num_classes=3, feature_dim=256):
        """Load from saved file."""
        data = np.load(path)
        ood  = cls(num_classes, feature_dim)
        ood.centroids = data['centroids']
        ood.precision = data['precision']
        ood.threshold = float(data['threshold'][0])
        ood.fitted    = True
        return ood


# ══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL OOD HELPERS (MC-Dropout + Entropy as backup layers)
# ══════════════════════════════════════════════════════════════════════════════
def softmax_entropy(probs):
    p = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(p * np.log(p)))


def mc_predict(model, img_array, n=MC_DROPOUT_PASSES):
    preds = np.stack(
        [model(img_array, training=True).numpy() for _ in range(n)], axis=0)
    return preds.mean(axis=0)[0], float(preds.std(axis=0)[0].mean())


# ══════════════════════════════════════════════════════════════════════════════
# FULL PREDICTION PIPELINE  (used in Flask)
# ══════════════════════════════════════════════════════════════════════════════
def predict_with_ood(model, feature_model, img_array,
                     classes, mahal_ood, entropy_threshold,
                     conf_threshold=OOD_CONFIDENCE_THRESHOLD):
    """
    Three-layer OOD defence:
      Layer 1 — Mahalanobis distance in feature space  (primary)
      Layer 2 — MC-Dropout epistemic uncertainty        (secondary)
      Layer 3 — Softmax entropy + confidence threshold  (backup)

    """
    # Extract features
    feature_vec = feature_model.predict(img_array, verbose=0)[0]

    # Layer 1: Mahalanobis OOD
    is_mahal_ood, min_dist, all_dists = mahal_ood.is_ood(feature_vec)

    # Layer 2: MC-Dropout
    mean_probs, mc_std = mc_predict(model, img_array)

    # Layer 3: Entropy + confidence
    entropy    = softmax_entropy(mean_probs)
    confidence = float(mean_probs.max())
    pred_idx   = int(mean_probs.argmax())

    ood, reasons = False, []

    if is_mahal_ood:
        ood = True
        reasons.append(
            f"Feature distance too large: {min_dist:.2f} > threshold {mahal_ood.threshold:.2f}"
        )

    if mc_std > 0.10:
        ood = True
        reasons.append(
            f"High MC-Dropout uncertainty: std={mc_std:.3f} > 0.10"
        )

    if entropy > entropy_threshold:
        ood = True
        reasons.append(
            f"High softmax entropy: {entropy:.3f} > threshold {entropy_threshold:.3f}"
        )

    if confidence < conf_threshold:
        ood = True
        reasons.append(
            f"Low confidence: {confidence*100:.1f}% < {conf_threshold*100:.0f}%"
        )

    return {
        "predicted_class"    : "Uncertain – Not Recognised" if ood else classes[pred_idx],
        "predicted_index"    : -1 if ood else pred_idx,
        "confidence_pct"     : round(confidence * 100, 2),
        "is_ood"             : ood,
        "rejection_reasons"  : reasons,
        "all_probabilities"  : {c: round(float(p)*100,2) for c,p in zip(classes, mean_probs)},
        # Detailed OOD scores
        "mahalanobis_dist"   : round(min_dist, 4),
        "mahal_threshold"    : round(mahal_ood.threshold, 4),
        "mahal_class_dists"  : {c: round(d,4) for c,d in zip(classes, all_dists)},
        "entropy"            : round(entropy, 4),
        "mc_uncertainty_std" : round(mc_std, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model, test_gen, classes):
    print("\n" + "="*70)
    print("STEP 6 – EVALUATION")
    print("="*70)

    probs  = model.predict(test_gen, verbose=1)
    y_pred = probs.argmax(axis=1)
    y_true = test_gen.classes
    acc    = accuracy_score(y_true, y_pred)

    print(f"\n  Test Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    Path('training_outputs').mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (fmt, title, data) in zip(axes, [
        ('d',   'Counts',     cm),
        ('.2%', 'Normalized', cm.astype(float)/cm.sum(axis=1, keepdims=True)),
    ]):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(f'Confusion Matrix ({title})')
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')

    plt.tight_layout()
    plt.savefig('training_outputs/confusion_matrix.png', dpi=200)
    plt.close()
    print("  Per-class accuracy:")
    for i, cls in enumerate(classes):
        pa = cm[i,i]/cm[i].sum() if cm[i].sum() else 0
        print(f"    {cls:10s}: {pa:.4f} ({pa*100:.1f}%)")
    return acc


def plot_history(h1, h2):
    Path('training_outputs').mkdir(exist_ok=True)
    e1, e2 = len(h1.history['accuracy']), len(h2.history['accuracy'])
    r1, r2 = range(1,e1+1), range(e1+1, e1+e2+1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, m, t in zip(axes, ['accuracy','loss'], ['Accuracy','Loss']):
        ax.plot(r1, h1.history[m],       'b-',  label='S1 Train', lw=2)
        ax.plot(r1, h1.history[f'val_{m}'],'b--',label='S1 Val',   lw=2)
        ax.plot(r2, h2.history[m],       'r-',  label='S2 Train', lw=2)
        ax.plot(r2, h2.history[f'val_{m}'],'r--',label='S2 Val',   lw=2)
        ax.axvline(x=e1, color='gray', linestyle=':', lw=1.5, label='Fine-tune start')
        ax.set_title(t); ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle('Training History', y=1.01)
    plt.tight_layout()
    plt.savefig('training_outputs/training_history.png', dpi=200, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*70)
    print("GROUP 13 – DISEASE CLASSIFICATION (FINAL + FEATURE OOD)")
    print("="*70)

    # 1. Clean
    cleaner = DataCleaner(DATA_DIR, CLASSES)
    clean_dir, _ = cleaner.clean()

    # 2. Generators
    train_gen, val_gen, test_gen, weights = make_generators(
        clean_dir, IMG_SIZE, BATCH_SIZE, CLASSES)

    # 3. Build
    model, feature_model, base = build_model(IMG_SIZE, len(CLASSES))

    # 4. Train
    h1, h2, best_ckpt = train(model, base, train_gen, val_gen, weights)

    # Reload best weights
    print(f"\n  Loading best checkpoint: {best_ckpt}")
    model = keras.models.load_model(best_ckpt)
    # Rebuild feature model from reloaded model weights
    feature_model = keras.Model(
        model.input,
        model.get_layer('feature_layer').output,
        name='feature_extractor'
    )

    # 5. Fit Mahalanobis OOD
    mahal_ood = MahalanobisOOD(num_classes=len(CLASSES), feature_dim=256)
    mahal_ood.fit(feature_model, train_gen, CLASSES)
    mahal_ood.calibrate_threshold(feature_model, val_gen, percentile=95)
    mahal_ood.save('mahalanobis_ood.npz')

    # 6. Calibrate entropy threshold
    print("\n  Calibrating entropy threshold on validation set…")
    val_gen.reset()
    all_probs     = model.predict(val_gen, verbose=0)
    all_entropies = np.array([softmax_entropy(p) for p in all_probs])
    entropy_thresh = float(np.percentile(all_entropies, 95))
    print(f"  Entropy 95th pct: {entropy_thresh:.4f}")

    # 7. Plot + evaluate
    plot_history(h1, h2)
    test_acc = evaluate(model, test_gen, CLASSES)

    # 8. Save everything
    print("\n" + "="*70)
    print("SAVING")
    print("="*70)
    model.save('disease_classifier_final.keras')
    model.save('disease_classifier_final.h5')
    feature_model.save('feature_extractor.keras')
    feature_model.save('feature_extractor.h5')

    ood_cfg = {
        'entropy_threshold'    : entropy_thresh,
        'confidence_threshold' : OOD_CONFIDENCE_THRESHOLD,
        'mc_dropout_passes'    : MC_DROPOUT_PASSES,
        'classes'              : CLASSES,
        'img_size'             : list(IMG_SIZE),
        'ood_method'           : 'mahalanobis_primary + mc_dropout + entropy',
        'mahal_ood_file'       : 'mahalanobis_ood.npz',
        'feature_layer'        : 'feature_layer',
        'feature_dim'          : 256,
    }
    with open('ood_config.json','w') as f: json.dump(ood_cfg, f, indent=2)

    print("  ✓ disease_classifier_final.h5")
    print("  ✓ feature_extractor.h5")
    print("  ✓ mahalanobis_ood.npz   ← centroids + precision + threshold")
    print("  ✓ class_mapping.json")
    print("  ✓ ood_config.json")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Test Accuracy          : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Mahalanobis threshold  : {mahal_ood.threshold:.4f}")
    print(f"  Entropy threshold      : {entropy_thresh:.4f}")
    print(f"  Class order            : {CLASSES}")
    print("\n  OOD layers active:")
    print("    1. Mahalanobis distance (feature space)  ← PRIMARY")
    print("    2. MC-Dropout uncertainty                ← SECONDARY")
    print("    3. Softmax entropy                       ← BACKUP")
    print("    4. Confidence floor                      ← BACKUP")
    print("="*70)


if __name__ == '__main__':
    main()