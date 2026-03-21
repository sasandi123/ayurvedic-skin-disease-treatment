"""
Generate visual evaluation artifacts from the current checkpoint without retraining.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from train import (
    Config,
    build_clean_samples,
    build_dataloaders,
    build_model,
    collect_predictions,
    default_thresholds,
    load_checkpoint,
    save_report_artifacts,
    set_seed,
    split_samples,
)


def load_config(data_dir: Path) -> Config:
    metrics_path = data_dir / "training_metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        if "config" in metrics:
            return Config(**metrics["config"])
    return Config()


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = data_dir / "severity_model_pytorch.h5"
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)

    config = load_config(data_dir)
    config.model_name = checkpoint.get("model_name", config.model_name)
    config.img_size = int(checkpoint.get("img_size", config.img_size))
    config.dropout = float(checkpoint.get("dropout", config.dropout))

    set_seed(config.seed)
    samples, soft_conflicts, _cleanup_report = build_clean_samples(data_dir)
    train_samples, val_samples, test_samples = split_samples(samples, config)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_samples + soft_conflicts,
        val_samples,
        test_samples,
        config,
        device,
    )

    model = build_model(config.model_name, config.dropout, device, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    thresholds = checkpoint.get("ordinal_thresholds", default_thresholds())
    val_tta = collect_predictions(model, val_loader, device, thresholds, use_tta=True, zoom_margin=config.tta_zoom_margin)
    test_tta = collect_predictions(model, test_loader, device, thresholds, use_tta=True, zoom_margin=config.tta_zoom_margin)
    artifacts = save_report_artifacts(
        data_dir,
        config,
        train_samples,
        val_samples,
        test_samples,
        val_tta,
        test_tta,
    )
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
