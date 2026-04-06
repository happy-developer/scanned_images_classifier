from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr_image_text.data import load_images_from_subdirs, resolve_default_data_root


CSV_COLUMNS = ["File Name", "OCRed Text", "split", "source_path", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a manual annotation CSV from batch_2 and tests_AB image subsets."
    )
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--output-csv", type=str, default="notebooks/artifacts/real_val_manual_annotation.csv")
    parser.add_argument(
        "--batch2-subdirs",
        type=str,
        default="batch_2/batch_2/batch2_1,batch_2/batch_2/batch2_2,batch_2/batch_2/batch2_3",
        help="Comma-separated batch_2 image subdirectories to include.",
    )
    parser.add_argument(
        "--tests-ab-subdirs",
        type=str,
        default="tests_AB",
        help="Comma-separated tests_AB image subdirectories to include.",
    )
    parser.add_argument("--max-batch2-samples", type=int, default=0, help="Limit batch_2 rows; 0 means no limit.")
    parser.add_argument("--max-tests-ab-samples", type=int, default=0, help="Limit tests_AB rows; 0 means no limit.")
    return parser.parse_args()


def _split_csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _rel_source_path(data_root: Path, image_path: Path) -> str:
    try:
        return image_path.resolve().relative_to(data_root.resolve()).as_posix()
    except Exception:
        return image_path.resolve().as_posix()


def _collect_rows(data_root: Path, split_name: str, subdirs: Iterable[str], limit: int) -> list[dict[str, str]]:
    image_paths = load_images_from_subdirs(data_root, tuple(subdirs))
    if limit > 0:
        image_paths = image_paths[:limit]

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for image_path in image_paths:
        source_path = _rel_source_path(data_root, image_path)
        key = source_path.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "File Name": image_path.name,
                "OCRed Text": "",
                "split": split_name,
                "source_path": source_path,
                "notes": "",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    data_root = resolve_default_data_root(args.data_root)
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    batch2_rows = _collect_rows(
        data_root=data_root,
        split_name="batch_2",
        subdirs=_split_csv_list(args.batch2_subdirs),
        limit=args.max_batch2_samples,
    )
    tests_ab_rows = _collect_rows(
        data_root=data_root,
        split_name="tests_AB",
        subdirs=_split_csv_list(args.tests_ab_subdirs),
        limit=args.max_tests_ab_samples,
    )

    rows = batch2_rows + tests_ab_rows
    rows.sort(key=lambda row: (row["split"], row["source_path"].lower()))

    if not rows:
        raise FileNotFoundError(
            "No images were found in the requested batch_2/tests_AB subsets. Check the subdir arguments and data root."
        )

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "output_csv": str(output_csv),
        "data_root": str(data_root),
        "num_rows": len(rows),
        "num_batch2_rows": len(batch2_rows),
        "num_tests_ab_rows": len(tests_ab_rows),
        "batch2_subdirs": list(_split_csv_list(args.batch2_subdirs)),
        "tests_ab_subdirs": list(_split_csv_list(args.tests_ab_subdirs)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
