"""
Relabel high-confidence cross-label conflicts caused by misplaced augmentations.

Default behavior is a dry run:
- scans the Mild / Moderate / Severe folders
- groups files by a normalized basename
- if a cross-label group has exactly one non-augmented "original" file,
  treats that original file's class as canonical
- moves only the conflicting augmented files into that canonical class folder

Safety defaults:
- skips ambiguous groups with zero or multiple non-augmented originals
- skips any move whose destination filename already exists
- does not modify files unless --apply is provided
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


CLASS_NAMES = ["Mild", "Moderate", "Severe"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NORMALIZE_PREFIX_RE = re.compile(
    r"^(?:mendeley_aug_\d+_|mendeley_\d+_|aug_\d+_|mendeley_|aug_)",
    re.IGNORECASE,
)
AUGMENTED_PREFIX_RE = re.compile(r"^(?:mendeley_aug_\d+_|aug_\d+_)", re.IGNORECASE)


@dataclass
class FileRecord:
    path: Path
    class_name: str
    normalized_key: str
    is_augmented: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relabel high-confidence augmented cross-label conflicts.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing Mild/Moderate/Severe folders.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files. Without this flag the script only prints a dry run.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("relabel_report.csv"),
        help="CSV report path.",
    )
    return parser.parse_args()


def iter_dataset_files(data_dir: Path) -> Iterable[Path]:
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                yield path


def normalize_basename(path: Path) -> str:
    stem = path.stem.lower()
    stem = NORMALIZE_PREFIX_RE.sub("", stem)
    stem = stem.replace(" ", "-").replace("_", "-")
    stem = re.sub(r"-+", "-", stem).strip("-")
    return stem


def is_augmented_name(path: Path) -> bool:
    return bool(AUGMENTED_PREFIX_RE.match(path.name))


def collect_records(data_dir: Path) -> List[FileRecord]:
    records: List[FileRecord] = []
    for path in iter_dataset_files(data_dir):
        records.append(
            FileRecord(
                path=path,
                class_name=path.parent.name,
                normalized_key=normalize_basename(path),
                is_augmented=is_augmented_name(path),
            )
        )
    return records


def write_report(rows: List[Dict[str, str]], report_path: Path) -> None:
    fieldnames = [
        "normalized_key",
        "reason",
        "original_class",
        "original_path",
        "source_class",
        "source_path",
        "target_class",
        "target_path",
        "action",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()

    records = collect_records(data_dir)
    grouped: Dict[str, List[FileRecord]] = defaultdict(list)
    for record in records:
        grouped[record.normalized_key].append(record)

    cross_label_groups = 0
    canonical_groups = 0
    ambiguous_groups = 0
    destination_conflicts = 0
    moved_files = 0
    rows: List[Dict[str, str]] = []

    for normalized_key, group in sorted(grouped.items()):
        class_names = sorted({record.class_name for record in group})
        if len(class_names) <= 1:
            continue

        cross_label_groups += 1
        originals = [record for record in group if not record.is_augmented]
        if len(originals) != 1:
            ambiguous_groups += 1
            continue

        canonical_groups += 1
        original = originals[0]
        target_dir = data_dir / original.class_name

        for record in sorted(group, key=lambda item: (item.class_name, item.path.name.lower())):
            if record.class_name == original.class_name:
                continue

            target_path = target_dir / record.path.name
            action = "would_move"
            if target_path.exists():
                action = "skipped_destination_exists"
                destination_conflicts += 1
            elif args.apply:
                record.path.replace(target_path)
                action = "moved"
                moved_files += 1

            rows.append(
                {
                    "normalized_key": normalized_key,
                    "reason": "single_original_class_anchor",
                    "original_class": original.class_name,
                    "original_path": str(original.path),
                    "source_class": record.class_name,
                    "source_path": str(record.path),
                    "target_class": original.class_name,
                    "target_path": str(target_path),
                    "action": action,
                }
            )

    write_report(rows, args.report_csv)

    action_label = "Moved" if args.apply else "Would move"
    action_count = sum(1 for row in rows if row["action"] in {"would_move", "moved"})
    print(f"Scanned files: {len(records)}")
    print(f"Cross-label normalized groups: {cross_label_groups}")
    print(f"Canonical single-original groups: {canonical_groups}")
    print(f"Ambiguous groups skipped: {ambiguous_groups}")
    print(f"{action_label} files: {action_count}")
    print(f"Destination conflicts skipped: {destination_conflicts}")
    print(f"Report written to: {args.report_csv.resolve()}")
    if not args.apply:
        print("Dry run only. Re-run with --apply to move files.")
    else:
        class_counts = Counter(path.parent.name for path in iter_dataset_files(data_dir))
        print(f"Class counts after relabel: {dict(class_counts)}")


if __name__ == "__main__":
    main()
