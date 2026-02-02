"""
Disease Classification - FIXED VERSION

This fixed version addresses the low accuracy issue by:
- Removing rescale=1./255 from data generators (since EfficientNetB3 includes internal rescaling expecting [0,255] inputs)
- Increasing initial learning rate to 5e-5 for faster convergence
- Unfreezing more base layers (50 instead of 30) for better feature adaptation while keeping anti-overfitting measures
- Keeping aggressive augmentation and stronger regularization

This should resolve the "stuck" training with low accuracy.
Expected: Validation accuracy should start increasing from the first epochs, reaching 85-95% with low overfitting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import hashlib
import shutil
from collections import defaultdict, Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)


class FixedDiseaseClassificationPipeline:
    """Fixed pipeline with input range correction and adjustments"""

    def __init__(self, data_dir, img_size=(299, 299), batch_size=32):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.classes = ['acne', 'eczema', 'ringworm']

        self.train_dir = self.data_dir / 'train'
        self.val_dir = self.data_dir / 'validation'
        self.test_dir = self.data_dir / 'test'

        self.cleaned_dir = self.data_dir.parent / 'cleaned_dataset'
        self.cleaned_train_dir = self.cleaned_dir / 'train'
        self.cleaned_val_dir = self.cleaned_dir / 'validation'
        self.cleaned_test_dir = self.cleaned_dir / 'test'

        self.model = None
        self.class_weights = None
        self.stats = {
            'duplicates_removed': 0,
            'corrupt_removed': 0,
            'cross_split_removed': 0,
            'original_count': 0,
            'final_count': 0,
            'class_distribution': {}
        }

    def compute_image_hash(self, img_path):
        """Compute hash of image content"""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            img_resized = cv2.resize(img, (64, 64))
            return hashlib.md5(img_resized.tobytes()).hexdigest()
        except:
            return None

    def clean_dataset_automatically(self):
        """Automatically clean dataset"""
        print("=" * 80)
        print("STEP 1: AUTOMATIC DATA CLEANING")
        print("=" * 80)

        all_images = {}
        split_images = defaultdict(list)

        splits = {
            'train': self.train_dir,
            'validation': self.val_dir,
            'test': self.test_dir
        }

        original_count = 0
        corrupt_count = 0

        for split_name, split_dir in splits.items():
            print(f"\n Processing {split_name} set...")

            for class_name in self.classes:
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    continue

                for img_path in class_dir.glob('*.*'):
                    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                        continue

                    original_count += 1

                    img = cv2.imread(str(img_path))
                    if img is None:
                        corrupt_count += 1
                        continue

                    img_hash = self.compute_image_hash(img_path)
                    if img_hash is None:
                        corrupt_count += 1
                        continue

                    split_images[split_name].append({
                        'path': img_path,
                        'hash': img_hash,
                        'class': class_name,
                        'split': split_name
                    })

                    if img_hash not in all_images:
                        all_images[img_hash] = {
                            'first_path': img_path,
                            'first_split': split_name,
                            'class': class_name,
                            'count': 1
                        }
                    else:
                        all_images[img_hash]['count'] += 1

        print(f"\n Initial scan complete:")
        print(f"   Total images found: {original_count}")
        print(f"   Corrupt images: {corrupt_count}")

        cross_split_leakage = []
        for img_hash, info in all_images.items():
            splits_with_hash = set()
            for split_name, images in split_images.items():
                if any(img['hash'] == img_hash for img in images):
                    splits_with_hash.add(split_name)

            if len(splits_with_hash) > 1:
                cross_split_leakage.append({
                    'hash': img_hash,
                    'splits': splits_with_hash,
                    'class': info['class']
                })

        if cross_split_leakage:
            print(f"  Found {len(cross_split_leakage)} images in multiple splits")

        if self.cleaned_dir.exists():
            shutil.rmtree(self.cleaned_dir)

        for split_name in ['train', 'validation', 'test']:
            for class_name in self.classes:
                (self.cleaned_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)

        seen_hashes = set()
        duplicates_removed = 0
        copied_count = 0
        class_counts = {split: {cls: 0 for cls in self.classes} for split in ['train', 'validation', 'test']}

        priority_order = ['train', 'validation', 'test']

        for split_name in priority_order:
            for img_info in split_images[split_name]:
                img_hash = img_info['hash']
                img_path = img_info['path']
                class_name = img_info['class']

                if img_hash in seen_hashes:
                    duplicates_removed += 1
                    continue

                is_cross_split = img_hash in [leak['hash'] for leak in cross_split_leakage]

                if is_cross_split:
                    first_occurrence_split = all_images[img_hash]['first_split']
                    if split_name != first_occurrence_split:
                        continue

                dest_path = self.cleaned_dir / split_name / class_name / img_path.name
                shutil.copy2(img_path, dest_path)
                seen_hashes.add(img_hash)
                copied_count += 1
                class_counts[split_name][class_name] += 1

        self.stats['original_count'] = original_count
        self.stats['final_count'] = copied_count
        self.stats['duplicates_removed'] = duplicates_removed
        self.stats['corrupt_removed'] = corrupt_count
        self.stats['class_distribution'] = class_counts

        print(f"\n✓ Cleaning complete:")
        print(f"   Copied: {copied_count} images")
        print(f"   Removed: {original_count - copied_count} images")

        print("\n Class Distribution After Cleaning:")
        print("-" * 80)
        for split_name in ['train', 'validation', 'test']:
            print(f"\n {split_name.upper()}:")
            total = sum(class_counts[split_name].values())
            for class_name in self.classes:
                count = class_counts[split_name][class_name]
                pct = (count / total * 100) if total > 0 else 0
                print(f"   {class_name:10s}: {count:4d} ({pct:5.2f}%)")

        train_counts = [class_counts['train'][cls] for cls in self.classes]
        if train_counts:
            imbalance_ratio = max(train_counts) / min(train_counts) if min(train_counts) > 0 else 0
            print(f"\n Class Imbalance Ratio: {imbalance_ratio:.2f}:1")

            if imbalance_ratio > 1.5:
                print("  ⚠ Significant class imbalance detected!")
                print("    → Using class weights")
            else:
                print("  ✓ Classes are relatively balanced")

        return self.stats

    def create_data_generators(self):
        """Create data generators with aggressive augmentation but NO RESCALE (fixed for EfficientNet)"""
        print("\n" + "=" * 80)
        print("STEP 2: DATA GENERATORS")
        print("=" * 80)

        # Aggressive augmentation, but NO rescale (EfficientNet handles internally)
        train_datagen = ImageDataGenerator(
            # rescale removed
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            channel_shift_range=20.0,
            fill_mode='nearest'
        )

        test_datagen = ImageDataGenerator(
            # rescale removed
        )

        train_generator = train_datagen.flow_from_directory(
            self.cleaned_train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )

        val_generator = test_datagen.flow_from_directory(
            self.cleaned_val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        test_generator = test_datagen.flow_from_directory(
            self.cleaned_test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        # Calculate class weights
        class_counts = Counter(train_generator.classes)
        total_samples = sum(class_counts.values())

        self.class_weights = {}
        print("\n Class Weights:")
        print("-" * 80)
        for class_idx in sorted(class_counts.keys()):
            class_name = list(train_generator.class_indices.keys())[class_idx]
            weight = total_samples / (len(class_counts) * class_counts[class_idx])
            self.class_weights[class_idx] = weight
            print(f"   {class_name:10s}: {weight:.4f} (samples: {class_counts[class_idx]})")

        print("\n✓ Data generators created")
        print(f"   Training samples: {train_generator.samples}")
        print(f"   Validation samples: {val_generator.samples}")
        print(f"   Test samples: {test_generator.samples}")
        print(f"   Aggressive augmentation enabled")
        print(f"   IMPORTANT FIX: No rescaling (inputs in [0,255] as expected by EfficientNet)")

        return train_generator, val_generator, test_generator

    def build_model(self):
        """Build model with more trainable layers"""
        print("\n" + "=" * 80)
        print("STEP 3: BUILDING MODEL")
        print("=" * 80)

        base_model = EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=(*self.img_size, 3)
        )

        # Unfreeze more layers (50 instead of 30)
        for layer in base_model.layers[:-50]:
            layer.trainable = False

        print(f"\n Base Model: EfficientNetB3")
        print(f"   Total layers: {len(base_model.layers)}")
        print(f"   Trainable layers: {sum([1 for l in base_model.layers if l.trainable])}")

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),

            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(
                256,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.005),
                kernel_initializer='he_normal'
            ),

            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(
                128,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.005),
                kernel_initializer='he_normal'
            ),

            layers.Dense(len(self.classes), activation='softmax')
        ])

        self.model = model

        print("\n Model Architecture:")
        print("-" * 80)
        print("   1. EfficientNetB3 (50 trainable layers)")
        print("   2. GlobalAveragePooling2D")
        print("   3. Dense(256) + BatchNorm + Dropout(0.5) + L2(0.005)")
        print("   4. Dense(128) + BatchNorm + Dropout(0.5) + L2(0.005)")
        print("   5. Output(3) + Softmax")
        print("\n✓ Model built")

        return model

    def compile_model(self, learning_rate=5e-5):
        """Compile with higher LR"""
        print("\n" + "=" * 80)
        print("STEP 4: COMPILING MODEL")
        print("=" * 80)

        optimizer = Adam(learning_rate=learning_rate)

        loss = 'categorical_crossentropy'

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )

        print(f"\n Loss Function: Categorical Cross-Entropy")
        print(f"   Optimizer: Adam")
        print(f"   Learning Rate: {learning_rate} (increased for faster convergence)")
        print(f"   Class Weights: Enabled")
        print("\n✓ Model compiled")

    def train_model(self, train_generator, val_generator, epochs=50):
        """Train model"""
        print("\n" + "=" * 80)
        print("STEP 5: TRAINING MODEL")
        print("=" * 80)

        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=25,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.3,
                patience=10,
                min_lr=1e-7,
                verbose=1,
                mode='max'
            ),
            ModelCheckpoint(
                'best_disease_model_fixed.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]

        print("\n Training Configuration:")
        print("-" * 80)
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Steps per epoch: {len(train_generator)}")
        print("   Callbacks:")
        print("     • EarlyStopping (patience=25)")
        print("     • ReduceLROnPlateau (factor=0.3, patience=10)")
        print("     • ModelCheckpoint")
        print("\n" + "-" * 80)

        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )

        print("\n" + "-" * 80)
        print("✓ Training completed!")

        return history

    def plot_training_history(self, history):
        """Plot training metrics"""
        print("\n" + "=" * 80)
        print("STEP 6: TRAINING ANALYSIS")
        print("=" * 80)

        output_dir = Path('training_outputs')
        output_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        metrics = [
            ('accuracy', 'Accuracy'),
            ('loss', 'Loss'),
            ('precision', 'Precision'),
            ('recall', 'Recall'),
            ('auc', 'AUC')
        ]

        for idx, (metric, title) in enumerate(metrics):
            row = idx // 3
            col = idx % 3

            if metric in history.history:
                axes[row, col].plot(history.history[metric], label='Train', linewidth=2)
                axes[row, col].plot(history.history[f'val_{metric}'], label='Val', linewidth=2)
                axes[row, col].set_title(f'{title}', fontsize=14, fontweight='bold')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel(title)
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'training_history_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()

        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        gap = abs(final_train_acc - final_val_acc)

        print(f"\n Training Summary:")
        print("-" * 80)
        print(f"   Final Train Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
        print(f"   Final Val Accuracy:   {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        print(f"   Best Val Accuracy:    {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        print(f"   Best at Epoch:        {best_epoch}")
        print(f"   Overfitting Gap:      {gap:.4f} ({gap*100:.2f}%)")

        if gap < 0.05:
            print("\n   ✓✓ EXCELLENT generalization! Gap < 5%")
        elif gap < 0.08:
            print("\n   ✓ Good generalization. Gap < 8%")
        else:
            print("\n    Moderate overfitting. Gap > 8%")

        print("\n✓ Training plots saved to 'training_outputs/'")

    def evaluate_model_with_tta(self, test_generator, tta_steps=5):
        """Evaluate with TTA - fixed no rescale"""
        print("\n" + "=" * 80)
        print("STEP 7: MODEL EVALUATION")
        print("=" * 80)

        # Standard evaluation
        print("\n Standard Evaluation:")
        y_pred_probs = self.model.predict(test_generator, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_generator.classes
        accuracy = accuracy_score(y_true, y_pred)
        print(f"   Base Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # TTA - no rescale
        print(f"\n Test-Time Augmentation ({tta_steps} steps):")
        tta_datagen = ImageDataGenerator(
            # rescale removed
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True
        )

        tta_generator = tta_datagen.flow_from_directory(
            self.cleaned_test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        tta_predictions = []
        for i in range(tta_steps):
            tta_generator.reset()
            preds = self.model.predict(tta_generator, verbose=0)
            tta_predictions.append(preds)
            print(f"   TTA step {i+1}/{tta_steps} completed")

        y_pred_tta = np.mean(tta_predictions, axis=0)
        y_pred_tta_classes = np.argmax(y_pred_tta, axis=1)
        tta_accuracy = accuracy_score(y_true, y_pred_tta_classes)

        improvement = (tta_accuracy - accuracy) * 100
        print(f"\n   TTA Accuracy:  {tta_accuracy:.4f} ({tta_accuracy*100:.2f}%)")
        print(f"   Improvement:   +{improvement:.2f}%")

        print("\n" + "=" * 80)
        print("FINAL TEST RESULTS")
        print("=" * 80)

        class_names = list(test_generator.class_indices.keys())
        report = classification_report(y_true, y_pred_tta_classes, target_names=class_names, digits=4)
        print(f"\nOverall Accuracy: {tta_accuracy:.4f} ({tta_accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print("-" * 80)
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_tta_classes)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')

        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')

        plt.tight_layout()
        plt.savefig(Path('training_outputs') / 'confusion_matrix_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Per-class performance
        print("\nPer-Class Performance:")
        print("-" * 80)
        for idx, class_name in enumerate(class_names):
            class_acc = cm[idx, idx] / cm[idx].sum() if cm[idx].sum() > 0 else 0
            class_total = cm[idx].sum()
            class_correct = cm[idx, idx]
            print(f"{class_name.capitalize():10s}: {class_acc:.4f} ({class_acc*100:.2f}%) - {class_correct}/{class_total} correct")

        print("\n✓ Evaluation complete")
        print("✓ Results saved to 'training_outputs/'")

        return tta_accuracy, report, cm


def main():
    """Main execution - FIXED VERSION"""

    print("\n" + "=" * 80)
    print(" " * 12 + "DISEASE CLASSIFICATION - FIXED VERSION")
    print(" " * 15 + "Input Range Correction + Adjustments")
    print("=" * 80)

    DATA_DIR = r"C:\Users\asus\Downloads\balanced_3diseases_dataset"
    IMG_SIZE = (299, 299)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 5e-5  # Increased

    print("\n Configuration:")
    print(f"   Dataset: {DATA_DIR}")
    print(f"   Image Size: {IMG_SIZE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Max Epochs: {EPOCHS}")
    print(f"   Learning Rate: {LEARNING_RATE}")

    pipeline = FixedDiseaseClassificationPipeline(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    try:
        stats = pipeline.clean_dataset_automatically()

        train_gen, val_gen, test_gen = pipeline.create_data_generators()

        model = pipeline.build_model()

        pipeline.compile_model(learning_rate=LEARNING_RATE)

        history = pipeline.train_model(train_gen, val_gen, epochs=EPOCHS)

        pipeline.plot_training_history(history)

        accuracy, report, cm = pipeline.evaluate_model_with_tta(test_gen, tta_steps=5)

        print("\n" + "=" * 80)
        print("SAVING MODEL")
        print("=" * 80)
        pipeline.model.save('final_disease_model_fixed.keras')
        print("✓ Model saved as 'final_disease_model_fixed.keras'")

        print("\n" + "=" * 80)
        print(" " * 32 + "COMPLETE!")
        print("=" * 80)

        print("\n Generated Files:")
        print("   • Cleaned dataset: cleaned_dataset/")
        print("   • Best model: best_disease_model_fixed.h5")
        print("   • Final model: final_disease_model_fixed.keras")
        print("   • Training plots: training_outputs/training_history_fixed.png")
        print("   • Confusion matrix: training_outputs/confusion_matrix_fixed.png")

        print("\n Final Statistics:")
        print(f"   Original images: {stats['original_count']}")
        print(f"   Cleaned images: {stats['final_count']}")
        print(f"   Removed images: {stats['original_count'] - stats['final_count']}")
        print(f"   Test Accuracy (with TTA): {accuracy:.4f} ({accuracy*100:.2f}%)")

        print("\n" + "=" * 80)
        print("FIXES IN THIS VERSION:")
        print("=" * 80)
        print("  ✓ Removed rescale=1./255 (EfficientNet expects [0,255] and handles scaling internally)")
        print("  ✓ Increased learning rate to 5e-5 for better convergence")
        print("  ✓ Unfroze more base layers (50) for improved learning while maintaining regularization")
        print("  ✓ Kept aggressive augmentation and high dropout/L2")

    except Exception as e:
        print(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()