"""
Find and optionally remove exact duplicate dataset images.

Default behavior is a dry run:
- groups files by SHA-1 content hash
- keeps the oldest modified file in each duplicate group
- reports which newer files would be deleted

Safety defaults:
- only scans Mild / Moderate / Severe folders
- skips cross-label duplicate groups unless --include-cross-label is passed
- does not delete anything unless --apply is provided
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


CLASS_NAMES = ["Mild", "Moderate", "Severe"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class FileRecord:
    path: Path
    class_name: str
    sha1: str
    mtime: float
    size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove exact duplicate dataset files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing Mild/Moderate/Severe folders.",
    )
    parser.add_argument(
        "--include-cross-label",
        action="store_true",
        help="Also remove duplicates when the same image exists in different class folders.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag the script only prints a dry run.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("duplicate_report.csv"),
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


def sha1_of_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_records(data_dir: Path) -> List[FileRecord]:
    records: List[FileRecord] = []
    for path in iter_dataset_files(data_dir):
        stat = path.stat()
        records.append(
            FileRecord(
                path=path,
                class_name=path.parent.name,
                sha1=sha1_of_file(path),
                mtime=stat.st_mtime,
                size=stat.st_size,
            )
        )
    return records


def choose_keeper(records: List[FileRecord]) -> FileRecord:
    return sorted(
        records,
        key=lambda item: (
            item.mtime,
            len(str(item.path)),
            str(item.path).lower(),
        ),
    )[0]


def write_report(rows: List[Dict[str, str]], report_path: Path) -> None:
    if not rows:
        rows = [
            {
                "sha1": "",
                "class_names": "",
                "keep_path": "",
                "delete_path": "",
                "keep_mtime": "",
                "delete_mtime": "",
                "reason": "",
                "action": "",
            }
        ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()

    records = collect_records(data_dir)
    grouped: Dict[str, List[FileRecord]] = defaultdict(list)
    for record in records:
        grouped[record.sha1].append(record)

    total_groups = 0
    duplicate_groups = 0
    same_label_groups = 0
    cross_label_groups = 0
    deleted_count = 0
    skipped_cross_label_count = 0
    rows: List[Dict[str, str]] = []

    for sha1, group in sorted(grouped.items()):
        total_groups += 1
        if len(group) < 2:
            continue

        duplicate_groups += 1
        class_names = sorted({item.class_name for item in group})
        is_cross_label = len(class_names) > 1
        if is_cross_label:
            cross_label_groups += 1
        else:
            same_label_groups += 1

        keeper = choose_keeper(group)
        to_delete = sorted(
            [item for item in group if item.path != keeper.path],
            key=lambda item: (item.mtime, str(item.path).lower()),
        )

        if is_cross_label and not args.include_cross_label:
            skipped_cross_label_count += len(to_delete)
            for record in to_delete:
                rows.append(
                    {
                        "sha1": sha1,
                        "class_names": "|".join(class_names),
                        "keep_path": str(keeper.path),
                        "delete_path": str(record.path),
                        "keep_mtime": str(keeper.mtime),
                        "delete_mtime": str(record.mtime),
                        "reason": "cross_label_duplicate_skipped",
                        "action": "skipped",
                    }
                )
            continue

        for record in to_delete:
            action = "would_delete"
            if args.apply:
                record.path.unlink()
                action = "deleted"
                deleted_count += 1

            rows.append(
                {
                    "sha1": sha1,
                    "class_names": "|".join(class_names),
                    "keep_path": str(keeper.path),
                    "delete_path": str(record.path),
                    "keep_mtime": str(keeper.mtime),
                    "delete_mtime": str(record.mtime),
                    "reason": "cross_label_duplicate" if is_cross_label else "same_label_duplicate",
                    "action": action,
                }
            )

    write_report(rows, args.report_csv)

    action_label = "Deleted" if args.apply else "Would delete"
    print(f"Scanned files: {len(records)}")
    print(f"Hash groups: {total_groups}")
    print(f"Duplicate groups: {duplicate_groups}")
    print(f"Same-label duplicate groups: {same_label_groups}")
    print(f"Cross-label duplicate groups: {cross_label_groups}")
    print(f"{action_label} files: {sum(1 for row in rows if row['action'] in {'would_delete', 'deleted'})}")
    print(f"Skipped cross-label duplicates: {skipped_cross_label_count}")
    print(f"Report written to: {args.report_csv.resolve()}")
    if not args.apply:
        print("Dry run only. Re-run with --apply to delete files.")
    if cross_label_groups and not args.include_cross_label:
        print("Cross-label duplicates were skipped. Use --include-cross-label to remove them.")


if __name__ == "__main__":
    main()
