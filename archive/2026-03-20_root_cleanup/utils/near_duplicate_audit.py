"""
Audit visually similar images using perceptual dHash and generate review artifacts.

These are candidate near-duplicates, not guaranteed duplicates.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CLASS_NAMES = ["Mild", "Moderate", "Severe"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageRecord:
    path: Path
    class_name: str
    dhash: int


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    image = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = 0
    for value in diff.flatten():
        bits = (bits << 1) | int(value)
    return bits


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class BKTree:
    def __init__(self) -> None:
        self.root: Tuple[int, int, Dict[int, object]] | None = None

    def add(self, key: int, index: int) -> None:
        node = (key, index, {})
        if self.root is None:
            self.root = node
            return

        current = self.root
        while True:
            distance = hamming_distance(key, current[0])
            children = current[2]
            if distance in children:
                current = children[distance]
            else:
                children[distance] = node
                return

    def query(self, key: int, max_distance: int) -> List[int]:
        matches: List[int] = []
        if self.root is None:
            return matches

        stack = [self.root]
        while stack:
            current = stack.pop()
            distance = hamming_distance(key, current[0])
            if distance <= max_distance:
                matches.append(current[1])
            lower = distance - max_distance
            upper = distance + max_distance
            for edge_distance, child in current[2].items():
                if lower <= edge_distance <= upper:
                    stack.append(child)
        return matches


def load_records(data_dir: Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            try:
                image = Image.open(path).convert("RGB")
                records.append(ImageRecord(path=path, class_name=class_name, dhash=dhash(image)))
            except Exception:
                continue
    return records


def find_candidate_pairs(records: List[ImageRecord], max_distance: int = 4) -> List[Dict[str, object]]:
    tree = BKTree()
    pairs: List[Dict[str, object]] = []
    seen = set()
    for idx, record in enumerate(records):
        matches = tree.query(record.dhash, max_distance=max_distance)
        for other_idx in matches:
            left, right = sorted((idx, other_idx))
            if left == right or (left, right) in seen:
                continue
            seen.add((left, right))
            other = records[other_idx]
            distance = hamming_distance(record.dhash, other.dhash)
            pairs.append(
                {
                    "left_path": str(records[left].path),
                    "right_path": str(records[right].path),
                    "left_class": records[left].class_name,
                    "right_class": records[right].class_name,
                    "distance": distance,
                    "cross_label": records[left].class_name != records[right].class_name,
                }
            )
        tree.add(record.dhash, idx)

    pairs.sort(key=lambda item: (item["distance"], not item["cross_label"], item["left_path"], item["right_path"]))
    return pairs


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["left_path", "right_path", "left_class", "right_class", "distance", "cross_label"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_montage(rows: List[Dict[str, object]], output_path: Path, max_pairs: int = 8) -> None:
    chosen = rows[:max_pairs]
    if not chosen:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No near-duplicate candidates found at this threshold.", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close(fig)
        return

    fig, axes = plt.subplots(len(chosen), 2, figsize=(8, 3 * len(chosen)))
    if len(chosen) == 1:
        axes = np.array([axes])

    for row_idx, pair in enumerate(chosen):
        left = Image.open(pair["left_path"]).convert("RGB")
        right = Image.open(pair["right_path"]).convert("RGB")
        for col_idx, image in enumerate([left, right]):
            ax = axes[row_idx, col_idx]
            ax.imshow(image)
            ax.axis("off")
        axes[row_idx, 0].set_title(
            f"{pair['left_class']}\n{Path(pair['left_path']).name}",
            fontsize=10,
        )
        axes[row_idx, 1].set_title(
            f"{pair['right_class']}\n{Path(pair['right_path']).name}",
            fontsize=10,
        )
        axes[row_idx, 0].text(
            0.0,
            -0.12,
            f"dHash distance: {pair['distance']} | cross-label: {pair['cross_label']}",
            transform=axes[row_idx, 0].transAxes,
            fontsize=9,
        )

    fig.suptitle("Near-Duplicate Candidate Pairs", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    csv_path = data_dir / "near_duplicate_candidates.csv"
    image_path = data_dir / "near_duplicate_candidates.png"

    records = load_records(data_dir)
    pairs = find_candidate_pairs(records, max_distance=4)
    write_csv(pairs, csv_path)
    create_montage(pairs, image_path)

    cross_label = sum(1 for row in pairs if row["cross_label"])
    print(f"Scanned images: {len(records)}")
    print(f"Candidate near-duplicate pairs: {len(pairs)}")
    print(f"Cross-label candidate pairs: {cross_label}")
    print(f"CSV: {csv_path}")
    print(f"Image: {image_path}")


if __name__ == "__main__":
    main()
