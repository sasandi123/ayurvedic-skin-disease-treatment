"""
Skin Disease Severity Classification - Ordinal PyTorch Trainer

Key upgrades over the previous trainer:
- Cleans exact duplicate images and drops exact duplicates with conflicting labels.
- Treats Mild < Moderate < Severe as an ordinal problem instead of plain 3-way softmax.
- Preserves the full image with square padding instead of aggressive random cropping.
- Saves checkpoint metadata so inference can rebuild the correct model automatically.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


CLASS_NAMES = ["Mild", "Moderate", "Severe"]
NUM_CLASSES = len(CLASS_NAMES)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@dataclass
class Config:
    seed: int = 42
    model_name: str = "efficientnet_v2_m"
    img_size: int = 320
    batch_size: int = 4
    accumulation_steps: int = 4
    num_workers: int = 2
    head_epochs: int = 3
    finetune_epochs: int = 18
    patience: int = 6
    head_lr: float = 3e-4
    backbone_lr: float = 1.5e-5
    classifier_lr: float = 2.5e-4
    weight_decay: float = 2e-4
    dropout: float = 0.35
    val_ratio: float = 0.15
    test_ratio: float = 0.10
    ema_decay: float = 0.999
    tta_zoom_margin: int = 24
    output_name: str = "severity_model_pytorch.h5"
    metrics_name: str = "training_metrics.json"
    curves_name: str = "training_curves.png"
    dataset_distribution_name: str = "dataset_distribution.png"
    val_confusion_name: str = "confusion_matrix_val_tta.png"
    test_confusion_name: str = "confusion_matrix_test_tta.png"
    per_class_metrics_name: str = "per_class_metrics_test_tta.png"
    misclassified_examples_name: str = "misclassified_examples_test_tta.png"
    predictions_csv_name: str = "test_predictions_tta.csv"


@dataclass
class Sample:
    path: str
    label: int
    class_name: str
    sha1: str
    levels: Tuple[float, ...] | None = None


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train an ordinal skin severity classifier.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="efficientnet_v2_m")
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1.5e-5)
    parser.add_argument("--classifier-lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--tta-zoom-margin", type=int, default=24)
    args = parser.parse_args()
    return Config(**vars(args))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SquarePad:
    """Pads images to a square canvas to preserve the full lesion context."""

    def __init__(self, fill: Tuple[int, int, int] = (0, 0, 0)) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        pad_left = (side - width) // 2
        pad_top = (side - height) // 2
        pad_right = side - width - pad_left
        pad_bottom = side - height - pad_top
        return ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)


def get_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    resize_size = int(round(img_size * 1.08))
    train_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((resize_size, resize_size), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=12, interpolation=InterpolationMode.BILINEAR)],
                p=0.35,
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.14, contrast=0.14, saturation=0.08, hue=0.02)],
                p=0.65,
            ),
            transforms.RandomAutocontrast(p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.10, scale=(0.02, 0.10), ratio=(0.5, 2.5)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_transform, eval_transform


def sha1_of_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hard_levels_tuple(label: int) -> Tuple[float, ...]:
    return tuple(float(label > threshold) for threshold in range(NUM_CLASSES - 1))


def empirical_levels(labels: Sequence[int]) -> Tuple[float, ...]:
    labels_array = np.asarray(labels, dtype=np.int64)
    levels = []
    for threshold in range(NUM_CLASSES - 1):
        levels.append(float(np.mean(labels_array > threshold)))
    return tuple(levels)


def build_clean_samples(data_dir: Path) -> Tuple[List[Sample], List[Sample], Dict[str, object]]:
    raw_items: List[Sample] = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                raw_items.append(
                    Sample(
                        path=str(path),
                        label=label,
                        class_name=class_name,
                        sha1=sha1_of_file(path),
                    )
                )

    grouped: Dict[str, List[Sample]] = defaultdict(list)
    for sample in raw_items:
        grouped[sample.sha1].append(sample)

    cleaned: List[Sample] = []
    soft_conflicts: List[Sample] = []
    conflicts: List[Dict[str, object]] = []
    same_label_duplicates = 0
    for sha1, items in grouped.items():
        labels = sorted({item.class_name for item in items})
        if len(labels) > 1:
            labels_int = [item.label for item in items]
            mean_label = float(np.mean(labels_int))
            display_label = int(np.clip(round(mean_label), 0, NUM_CLASSES - 1))
            canonical_item = sorted(items, key=lambda item: item.path)[0]
            soft_conflicts.append(
                Sample(
                    path=canonical_item.path,
                    label=display_label,
                    class_name=CLASS_NAMES[display_label],
                    sha1=sha1,
                    levels=empirical_levels(labels_int),
                )
            )
            conflicts.append(
                {
                    "sha1": sha1,
                    "labels": labels,
                    "files": [item.path for item in items],
                }
            )
            continue

        items = sorted(items, key=lambda item: item.path)
        kept_item = items[0]
        cleaned.append(
            Sample(
                path=kept_item.path,
                label=kept_item.label,
                class_name=kept_item.class_name,
                sha1=kept_item.sha1,
                levels=hard_levels_tuple(kept_item.label),
            )
        )
        if len(items) > 1:
            same_label_duplicates += len(items) - 1

    cleaned = sorted(cleaned, key=lambda item: item.path)
    soft_conflicts = sorted(soft_conflicts, key=lambda item: item.path)
    report = {
        "raw_images": len(raw_items),
        "unique_hashes": len(grouped),
        "kept_unique_images": len(cleaned),
        "dropped_conflicting_hash_groups": len(conflicts),
        "dropped_conflicting_images": sum(len(conflict["files"]) for conflict in conflicts),
        "soft_conflict_training_hash_groups": len(soft_conflicts),
        "removed_same_label_duplicate_images": same_label_duplicates,
        "class_counts_after_cleanup": dict(Counter(item.class_name for item in cleaned)),
        "conflicts_preview": conflicts[:25],
    }
    return cleaned, soft_conflicts, report


def split_samples(samples: Sequence[Sample], config: Config) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    indices = list(range(len(samples)))
    labels = [sample.label for sample in samples]
    holdout_ratio = config.val_ratio + config.test_ratio

    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=holdout_ratio,
        stratify=labels,
        random_state=config.seed,
    )

    holdout_labels = [labels[idx] for idx in holdout_idx]
    test_fraction_of_holdout = config.test_ratio / holdout_ratio
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=test_fraction_of_holdout,
        stratify=holdout_labels,
        random_state=config.seed,
    )

    train_samples = [samples[idx] for idx in train_idx]
    val_samples = [samples[idx] for idx in val_idx]
    test_samples = [samples[idx] for idx in test_idx]
    return train_samples, val_samples, test_samples


class SkinSeverityDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform: transforms.Compose) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if sample.levels is None:
            levels = torch.tensor(hard_levels_tuple(sample.label), dtype=torch.float32)
        else:
            levels = torch.tensor(sample.levels, dtype=torch.float32)
        return self.transform(image), sample.label, levels


def build_dataloaders(
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    test_samples: Sequence[Sample],
    config: Config,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform, eval_transform = get_transforms(config.img_size)
    train_dataset = SkinSeverityDataset(train_samples, train_transform)
    val_dataset = SkinSeverityDataset(val_samples, eval_transform)
    test_dataset = SkinSeverityDataset(test_samples, eval_transform)

    loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader


class CoralLayer(nn.Module):
    """Rank-consistent ordinal head for severity levels."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("CORAL requires at least 2 classes.")
        self.fc = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.linspace(1.5, -1.5, steps=num_classes - 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shared_logits = self.fc(inputs)
        return shared_logits + self.bias


def model_spec(model_name: str):
    if model_name == "efficientnet_v2_m":
        return models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
    if model_name == "efficientnet_v2_s":
        return models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    if model_name == "convnext_base":
        return models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_model(
    model_name: str,
    dropout: float,
    device: torch.device,
    pretrained: bool = True,
) -> nn.Module:
    ctor, weights = model_spec(model_name)
    model = ctor(weights=weights if pretrained else None)

    if model_name.startswith("efficientnet"):
        in_features = model.classifier[1].in_features
        model.classifier[0] = nn.Dropout(p=dropout, inplace=True)
        model.classifier[1] = CoralLayer(in_features, NUM_CLASSES)
    elif model_name.startswith("convnext"):
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout),
            CoralLayer(in_features, NUM_CLASSES),
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model.to(device)


def backbone_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for name, param in model.named_parameters():
        if not name.startswith("classifier."):
            yield param


def classifier_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    yield from model.classifier.parameters()


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for param in backbone_parameters(model):
        param.requires_grad = trainable


def ordinal_levels(labels: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    thresholds = torch.arange(num_classes - 1, device=labels.device)
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def cumulative_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return torch.cummin(probs, dim=1).values


def class_probs_from_cumulative(cumulative_probs: torch.Tensor) -> torch.Tensor:
    if cumulative_probs.size(1) != NUM_CLASSES - 1:
        raise ValueError("Unexpected ordinal output shape.")
    first = 1.0 - cumulative_probs[:, :1]
    middle = cumulative_probs[:, :-1] - cumulative_probs[:, 1:]
    last = cumulative_probs[:, -1:]
    class_probs = torch.cat([first, middle, last], dim=1).clamp(min=0.0)
    return class_probs / class_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)


def default_thresholds() -> List[float]:
    return [0.5, 1.5]


def predict_from_scores(scores: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    preds = np.zeros_like(scores, dtype=np.int64)
    preds[scores >= thresholds[0]] = 1
    preds[scores >= thresholds[1]] = 2
    return preds


def tune_thresholds(scores: np.ndarray, labels: np.ndarray) -> Tuple[List[float], float]:
    unique_scores = np.unique(np.sort(scores))
    if unique_scores.size < 2:
        preds = np.zeros_like(labels)
        return default_thresholds(), accuracy_score(labels, preds)

    candidates = [0.0]
    candidates.extend(((unique_scores[:-1] + unique_scores[1:]) / 2.0).tolist())
    candidates.append(float(NUM_CLASSES - 1))

    best_thresholds = default_thresholds()
    best_acc = -1.0
    for left_idx in range(len(candidates)):
        for right_idx in range(left_idx + 1, len(candidates)):
            thresholds = [candidates[left_idx], candidates[right_idx]]
            preds = predict_from_scores(scores, thresholds)
            acc = accuracy_score(labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_thresholds = thresholds
    return best_thresholds, best_acc


def compute_pos_weight(samples: Sequence[Sample], device: torch.device) -> torch.Tensor:
    levels_matrix = np.asarray(
        [
            sample.levels if sample.levels is not None else hard_levels_tuple(sample.label)
            for sample in samples
        ],
        dtype=np.float32,
    )
    positives = torch.tensor(levels_matrix.sum(axis=0), dtype=torch.float32)
    negatives = len(samples) - positives
    pos_weight = negatives / positives.clamp_min(1.0)
    return pos_weight.to(device)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach()

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}


def zoom_view(images: torch.Tensor, margin: int) -> torch.Tensor:
    if margin <= 0:
        return images
    height = images.size(2)
    width = images.size(3)
    margin = min(margin, height // 4, width // 4)
    if margin <= 0:
        return images
    cropped = images[:, :, margin : height - margin, margin : width - margin]
    return torch.nn.functional.interpolate(
        cropped,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )


@torch.no_grad()
def forward_class_probs(
    model: nn.Module,
    images: torch.Tensor,
    use_tta: bool,
    zoom_margin: int,
) -> torch.Tensor:
    if not use_tta:
        logits = model(images)
        cumulative = cumulative_probs_from_logits(logits)
        return class_probs_from_cumulative(cumulative)

    views = [
        images,
        torch.flip(images, dims=[3]),
        zoom_view(images, zoom_margin),
    ]
    class_prob_views = []
    for view in views:
        logits = model(view)
        cumulative = cumulative_probs_from_logits(logits)
        class_prob_views.append(class_probs_from_cumulative(cumulative))
    return torch.stack(class_prob_views, dim=0).mean(dim=0)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: Sequence[float],
    use_tta: bool,
    zoom_margin: int,
) -> Dict[str, np.ndarray]:
    model.eval()
    labels_all: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []

    for images, labels, _levels in loader:
        images = images.to(device, non_blocking=True)
        class_probs = forward_class_probs(model, images, use_tta=use_tta, zoom_margin=zoom_margin)
        probs_all.append(class_probs.cpu().numpy())
        labels_all.append(labels.numpy())

    class_probs = np.concatenate(probs_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    scores = class_probs @ np.arange(NUM_CLASSES, dtype=np.float32)
    preds = predict_from_scores(scores, thresholds)
    return {
        "labels": labels,
        "preds": preds,
        "class_probs": class_probs,
        "scores": scores,
    }


def evaluate_single_crop(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels, levels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            levels = levels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, levels)
            cumulative = cumulative_probs_from_logits(logits)
            class_probs = class_probs_from_cumulative(cumulative)

            total_loss += loss.item() * images.size(0)
            total_items += images.size(0)
            probs_all.append(class_probs.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    class_probs = np.concatenate(probs_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    return total_loss / max(1, total_items), labels, class_probs


def summarize_predictions(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "qwk": float(cohen_kappa_score(labels, preds, weights="quadratic")),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    scheduler=None,
    accumulation_steps: int = 1,
    ema: EMA | None = None,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    total_items = 0
    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, total=len(loader), desc="  Train", leave=False)
    for step, (images, labels, levels) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        levels = levels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, levels)
            scaled_loss = loss / accumulation_steps

        scaler.scale(scaled_loss).backward()

        should_step = (step % accumulation_steps == 0) or (step == len(loader))
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)

        running_loss += loss.item() * images.size(0)
        total_items += images.size(0)

        with torch.no_grad():
            cumulative = cumulative_probs_from_logits(logits.detach())
            class_probs = class_probs_from_cumulative(cumulative)
            probs_all.append(class_probs.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
            scores = class_probs.cpu().numpy() @ np.arange(NUM_CLASSES, dtype=np.float32)
            preds = predict_from_scores(scores, default_thresholds())
            batch_acc = accuracy_score(labels.cpu().numpy(), preds)

        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}")

    class_probs = np.concatenate(probs_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    scores = class_probs @ np.arange(NUM_CLASSES, dtype=np.float32)
    preds = predict_from_scores(scores, default_thresholds())
    epoch_acc = accuracy_score(labels, preds)
    return running_loss / max(1, total_items), float(epoch_acc)


def _write_dataset_recursive(group: h5py.Group, key: str, value: np.ndarray) -> None:
    parts = key.split(".")
    current = group
    for part in parts[:-1]:
        current = current.require_group(part)
    current.create_dataset(parts[-1], data=value)


def save_checkpoint(path: Path, checkpoint: Dict[str, object]) -> None:
    path = Path(path)
    if path.suffix.lower() == ".h5":
        metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
        with h5py.File(path, "w") as handle:
            handle.attrs["metadata_json"] = json.dumps(metadata)
            state_group = handle.create_group("state_dict")
            for key, tensor in checkpoint["state_dict"].items():
                _write_dataset_recursive(state_group, key, tensor.detach().cpu().numpy())
        return

    torch.save(checkpoint, path)


def load_checkpoint(path: Path | str, map_location=None) -> Dict[str, object]:
    path = Path(path)
    if path.suffix.lower() == ".h5":
        with h5py.File(path, "r") as handle:
            metadata = json.loads(handle.attrs["metadata_json"])
            state_dict = OrderedDict()

            def visitor(name: str, obj) -> None:
                if isinstance(obj, h5py.Dataset):
                    tensor = torch.from_numpy(np.array(obj))
                    key = name.replace("/", ".")
                    state_dict[key] = tensor

            handle["state_dict"].visititems(visitor)
        metadata["state_dict"] = state_dict
        return metadata

    return torch.load(path, map_location=map_location)


def checkpoint_payload(
    model: nn.Module,
    config: Config,
    cleanup_report: Dict[str, object],
    thresholds: Sequence[float],
    best_epoch: int,
    best_val_acc: float,
) -> Dict[str, object]:
    return {
        "state_dict": copy.deepcopy(model.state_dict()),
        "model_name": config.model_name,
        "img_size": config.img_size,
        "class_names": CLASS_NAMES,
        "dropout": config.dropout,
        "mean": MEAN,
        "std": STD,
        "ordinal_thresholds": list(map(float, thresholds)),
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val_acc),
        "cleanup_report": cleanup_report,
        "trainer": "ordinal_coral",
        "num_classes": NUM_CLASSES,
    }


def plot_history(history: Dict[str, List[float]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", color="#1565C0")
    axes[0].plot(history["val_loss"], label="Val Loss", color="#D84315")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Acc", color="#1565C0")
    axes[1].plot(history["val_acc"], label="Val Acc", color="#D84315")
    axes[1].plot(history["val_qwk"], label="Val QWK", color="#2E7D32")
    axes[1].axhline(y=0.80, color="#6A1B9A", linestyle="--")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_dataset_distribution(
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    test_samples: Sequence[Sample],
    output_path: Path,
) -> None:
    split_samples = {
        "Train": train_samples,
        "Validation": val_samples,
        "Test": test_samples,
    }
    x = np.arange(len(CLASS_NAMES))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (split_name, samples) in enumerate(split_samples.items()):
        counts = [sum(sample.label == class_idx for sample in samples) for class_idx in range(NUM_CLASSES)]
        ax.bar(x + (idx - 1) * width, counts, width=width, label=split_name)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Images")
    ax.set_title("Dataset Class Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix_image(
    cm: np.ndarray,
    class_names: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    cm = np.asarray(cm)
    normalized = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = normalized[row, col]
            text_color = "white" if value > 0.5 else "black"
            ax.text(
                col,
                row,
                f"{cm[row, col]}\n{value * 100:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_per_class_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    output_path: Path,
) -> None:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    x = np.arange(len(CLASS_NAMES))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision, width=width, label="Precision")
    ax.bar(x, recall, width=width, label="Recall")
    ax.bar(x + width, f1, width=width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name}\n(n={int(support[idx])})" for idx, name in enumerate(CLASS_NAMES)])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics on Test Set (TTA)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_misclassified_examples(
    samples: Sequence[Sample],
    labels: np.ndarray,
    preds: np.ndarray,
    class_probs: np.ndarray,
    output_path: Path,
    max_items: int = 12,
) -> None:
    misclassified = []
    for idx, (label, pred) in enumerate(zip(labels, preds)):
        if int(label) == int(pred):
            continue
        confidence = float(class_probs[idx, int(pred)])
        misclassified.append((confidence, idx))

    if not misclassified:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No misclassified examples in this split.", ha="center", va="center", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close(fig)
        return

    misclassified.sort(reverse=True)
    chosen = misclassified[:max_items]
    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.8 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for plot_idx, (confidence, sample_idx) in enumerate(chosen):
        ax = axes.flat[plot_idx]
        sample = samples[sample_idx]
        image = Image.open(sample.path).convert("RGB")
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"True: {CLASS_NAMES[int(labels[sample_idx])]}\n"
            f"Pred: {CLASS_NAMES[int(preds[sample_idx])]} ({confidence * 100:.1f}%)",
            fontsize=10,
        )

    fig.suptitle("Most Confident Test Misclassifications (TTA)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close(fig)


def write_predictions_csv(
    samples: Sequence[Sample],
    labels: np.ndarray,
    preds: np.ndarray,
    class_probs: np.ndarray,
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "path",
            "true_label",
            "pred_label",
            "pred_confidence",
            "prob_mild",
            "prob_moderate",
            "prob_severe",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, sample in enumerate(samples):
            pred_label = int(preds[idx])
            writer.writerow(
                {
                    "path": sample.path,
                    "true_label": CLASS_NAMES[int(labels[idx])],
                    "pred_label": CLASS_NAMES[pred_label],
                    "pred_confidence": float(class_probs[idx, pred_label]),
                    "prob_mild": float(class_probs[idx, 0]),
                    "prob_moderate": float(class_probs[idx, 1]),
                    "prob_severe": float(class_probs[idx, 2]),
                }
            )


def save_report_artifacts(
    data_dir: Path,
    config: Config,
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    test_samples: Sequence[Sample],
    val_tta: Dict[str, np.ndarray],
    test_tta: Dict[str, np.ndarray],
) -> Dict[str, str]:
    dataset_distribution_path = data_dir / config.dataset_distribution_name
    val_confusion_path = data_dir / config.val_confusion_name
    test_confusion_path = data_dir / config.test_confusion_name
    per_class_metrics_path = data_dir / config.per_class_metrics_name
    misclassified_examples_path = data_dir / config.misclassified_examples_name
    predictions_csv_path = data_dir / config.predictions_csv_name

    plot_dataset_distribution(train_samples, val_samples, test_samples, dataset_distribution_path)
    plot_confusion_matrix_image(
        confusion_matrix(val_tta["labels"], val_tta["preds"]),
        CLASS_NAMES,
        "Validation Confusion Matrix (TTA)",
        val_confusion_path,
    )
    plot_confusion_matrix_image(
        confusion_matrix(test_tta["labels"], test_tta["preds"]),
        CLASS_NAMES,
        "Test Confusion Matrix (TTA)",
        test_confusion_path,
    )
    plot_per_class_metrics(test_tta["labels"], test_tta["preds"], per_class_metrics_path)
    plot_misclassified_examples(
        test_samples,
        test_tta["labels"],
        test_tta["preds"],
        test_tta["class_probs"],
        misclassified_examples_path,
    )
    write_predictions_csv(
        test_samples,
        test_tta["labels"],
        test_tta["preds"],
        test_tta["class_probs"],
        predictions_csv_path,
    )

    return {
        "dataset_distribution_image": str(dataset_distribution_path.name),
        "validation_confusion_image": str(val_confusion_path.name),
        "test_confusion_image": str(test_confusion_path.name),
        "per_class_metrics_image": str(per_class_metrics_path.name),
        "misclassified_examples_image": str(misclassified_examples_path.name),
        "test_predictions_csv": str(predictions_csv_path.name),
    }


def print_split_counts(name: str, samples: Sequence[Sample]) -> None:
    counts = Counter(CLASS_NAMES[sample.label] for sample in samples)
    print(f"{name}: {len(samples)} -> {dict(counts)}")


def main() -> None:
    config = parse_args()
    data_dir = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(config.seed)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nScanning dataset and cleaning exact duplicate conflicts...")
    samples, soft_conflict_samples, cleanup_report = build_clean_samples(data_dir)
    print(json.dumps({k: v for k, v in cleanup_report.items() if k != "conflicts_preview"}, indent=2))

    train_samples, val_samples, test_samples = split_samples(samples, config)
    print_split_counts("Train", train_samples)
    print_split_counts("Val", val_samples)
    print_split_counts("Test", test_samples)
    print(f"Extra soft-label conflict samples added to train only: {len(soft_conflict_samples)}")

    train_samples_for_loader = list(train_samples) + list(soft_conflict_samples)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_samples_for_loader,
        val_samples,
        test_samples,
        config,
        device,
    )

    print(f"\nBuilding model: {config.model_name}")
    model = build_model(config.model_name, config.dropout, device)

    pos_weight = compute_pos_weight(train_samples_for_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_qwk": [],
    }

    best_val_acc = 0.0
    best_epoch = -1
    best_payload = None

    print("\n" + "=" * 64)
    print(f"PHASE 1: classifier warmup for {config.head_epochs} epochs")
    print("=" * 64)
    set_backbone_trainable(model, False)
    optimizer = optim.AdamW(classifier_parameters(model), lr=config.head_lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.head_epochs))

    for epoch in range(config.head_epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            scheduler=scheduler,
            accumulation_steps=config.accumulation_steps,
            ema=None,
        )

        val_loss, val_labels, val_class_probs = evaluate_single_crop(model, val_loader, criterion, device)
        val_scores = val_class_probs @ np.arange(NUM_CLASSES, dtype=np.float32)
        tuned_thresholds, tuned_acc = tune_thresholds(val_scores, val_labels)
        val_preds = predict_from_scores(val_scores, tuned_thresholds)
        val_qwk = cohen_kappa_score(val_labels, val_preds, weights="quadratic")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(float(tuned_acc))
        history["val_qwk"].append(float(val_qwk))

        print(
            f"P1 Epoch {epoch + 1}/{config.head_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Acc: {tuned_acc:.4f} | Val QWK: {val_qwk:.4f}"
        )

        if tuned_acc > best_val_acc:
            best_val_acc = float(tuned_acc)
            best_epoch = epoch + 1
            best_payload = checkpoint_payload(
                model,
                config,
                cleanup_report,
                tuned_thresholds,
                best_epoch,
                best_val_acc,
            )
            save_checkpoint(data_dir / config.output_name, best_payload)

    print(f"\nPhase 1 best val acc: {best_val_acc:.4f}")

    print("\n" + "=" * 64)
    print(f"PHASE 2: full fine-tuning for {config.finetune_epochs} epochs")
    print("=" * 64)
    set_backbone_trainable(model, True)
    optimizer = optim.AdamW(
        [
            {"params": list(backbone_parameters(model)), "lr": config.backbone_lr},
            {"params": list(classifier_parameters(model)), "lr": config.classifier_lr},
        ],
        weight_decay=config.weight_decay,
    )
    total_optimizer_steps = max(
        1,
        int(np.ceil(len(train_loader) / config.accumulation_steps)) * config.finetune_epochs,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[config.backbone_lr, config.classifier_lr],
        total_steps=total_optimizer_steps,
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=50.0,
    )
    ema = EMA(model, decay=config.ema_decay)
    no_improve = 0

    for epoch in range(config.finetune_epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            scheduler=scheduler,
            accumulation_steps=config.accumulation_steps,
            ema=ema,
        )

        ema.apply_shadow(model)
        val_loss, val_labels, val_class_probs = evaluate_single_crop(model, val_loader, criterion, device)
        ema.restore(model)

        val_scores = val_class_probs @ np.arange(NUM_CLASSES, dtype=np.float32)
        tuned_thresholds, tuned_acc = tune_thresholds(val_scores, val_labels)
        val_preds = predict_from_scores(val_scores, tuned_thresholds)
        val_qwk = cohen_kappa_score(val_labels, val_preds, weights="quadratic")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(float(tuned_acc))
        history["val_qwk"].append(float(val_qwk))

        print(
            f"P2 Epoch {epoch + 1}/{config.finetune_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Acc: {tuned_acc:.4f} | Val QWK: {val_qwk:.4f}"
        )

        if tuned_acc > best_val_acc:
            best_val_acc = float(tuned_acc)
            best_epoch = config.head_epochs + epoch + 1
            no_improve = 0

            ema.apply_shadow(model)
            best_payload = checkpoint_payload(
                model,
                config,
                cleanup_report,
                tuned_thresholds,
                best_epoch,
                best_val_acc,
            )
            save_checkpoint(data_dir / config.output_name, best_payload)
            ema.restore(model)
            print(f"  New best model saved at val acc {best_val_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= config.patience:
                print(f"\nEarly stopping: no validation improvement for {config.patience} epochs.")
                break

    if best_payload is None:
        raise RuntimeError("No checkpoint was produced during training.")

    checkpoint = load_checkpoint(data_dir / config.output_name, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    print("\n" + "=" * 64)
    print("FINAL EVALUATION")
    print("=" * 64)

    val_single = collect_predictions(
        model,
        val_loader,
        device,
        thresholds=checkpoint.get("ordinal_thresholds", default_thresholds()),
        use_tta=False,
        zoom_margin=config.tta_zoom_margin,
    )
    retuned_thresholds, retuned_val_acc = tune_thresholds(val_single["scores"], val_single["labels"])
    checkpoint["ordinal_thresholds"] = list(map(float, retuned_thresholds))
    checkpoint["best_val_accuracy"] = float(retuned_val_acc)
    save_checkpoint(data_dir / config.output_name, checkpoint)

    val_single = collect_predictions(
        model,
        val_loader,
        device,
        thresholds=retuned_thresholds,
        use_tta=False,
        zoom_margin=config.tta_zoom_margin,
    )
    val_tta = collect_predictions(
        model,
        val_loader,
        device,
        thresholds=retuned_thresholds,
        use_tta=True,
        zoom_margin=config.tta_zoom_margin,
    )
    test_single = collect_predictions(
        model,
        test_loader,
        device,
        thresholds=retuned_thresholds,
        use_tta=False,
        zoom_margin=config.tta_zoom_margin,
    )
    test_tta = collect_predictions(
        model,
        test_loader,
        device,
        thresholds=retuned_thresholds,
        use_tta=True,
        zoom_margin=config.tta_zoom_margin,
    )

    val_single_metrics = summarize_predictions(val_single["labels"], val_single["preds"])
    val_tta_metrics = summarize_predictions(val_tta["labels"], val_tta["preds"])
    test_single_metrics = summarize_predictions(test_single["labels"], test_single["preds"])
    test_tta_metrics = summarize_predictions(test_tta["labels"], test_tta["preds"])

    print(f"Validation Accuracy (single): {val_single_metrics['accuracy'] * 100:.2f}%")
    print(f"Validation Accuracy (TTA):    {val_tta_metrics['accuracy'] * 100:.2f}%")
    print(f"Test Accuracy (single):       {test_single_metrics['accuracy'] * 100:.2f}%")
    print(f"Test Accuracy (TTA):          {test_tta_metrics['accuracy'] * 100:.2f}%")

    print("\nClassification Report (test, TTA):")
    print(classification_report(test_tta["labels"], test_tta["preds"], target_names=CLASS_NAMES))

    cm = confusion_matrix(test_tta["labels"], test_tta["preds"])
    print("Confusion Matrix (test, TTA):")
    print(cm)

    metrics_payload = {
        "config": asdict(config),
        "cleanup_report": cleanup_report,
        "best_epoch": checkpoint.get("best_epoch", best_epoch),
        "best_val_accuracy": checkpoint.get("best_val_accuracy", best_val_acc),
        "ordinal_thresholds": checkpoint["ordinal_thresholds"],
        "validation_single": val_single_metrics,
        "validation_tta": val_tta_metrics,
        "test_single": test_single_metrics,
        "test_tta": test_tta_metrics,
        "test_tta_confusion_matrix": cm.tolist(),
    }
    artifact_paths = save_report_artifacts(
        data_dir,
        config,
        train_samples,
        val_samples,
        test_samples,
        val_tta,
        test_tta,
    )
    metrics_payload["artifacts"] = artifact_paths
    with (data_dir / config.metrics_name).open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    plot_history(history, data_dir / config.curves_name)


if __name__ == "__main__":
    main()
