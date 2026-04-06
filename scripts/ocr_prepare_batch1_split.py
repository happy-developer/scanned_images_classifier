from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr_image_text.data import load_multi_ocr_sources, resolve_default_data_root


CSV_COLUMNS = ["File Name", "OCRed Text", "source_path", "split"]
DEFAULT_TRAIN_CSVS = "batch_1/batch_1/batch1_1.csv,batch_1/batch_1/batch1_2.csv,batch_1/batch_1/batch1_3.csv"
DEFAULT_IMAGE_SUBDIRS = "batch_1/batch_1/batch1_1,batch_1/batch_1/batch1_2,batch_1/batch_1/batch1_3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic 70/30 train/test split from batch_1 OCR labels."
    )
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of records assigned to train split (default: 0.7 for 70/30).",
    )
    parser.add_argument(
        "--train-csvs",
        type=str,
        default=DEFAULT_TRAIN_CSVS,
        help="Comma-separated batch_1 CSV files.",
    )
    parser.add_argument(
        "--image-subdirs",
        type=str,
        default=DEFAULT_IMAGE_SUBDIRS,
        help="Comma-separated batch_1 image subdirectories aligned with --train-csvs.",
    )
    return parser.parse_args()


def _split_arg(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _to_source_path(data_root: Path, image_path: Path) -> str:
    try:
        return image_path.resolve().relative_to(data_root.resolve()).as_posix()
    except Exception:
        return image_path.resolve().as_posix()


def _rows_from_records(records, data_root: Path, split_name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rec in records:
        rows.append(
            {
                "File Name": rec.img_name,
                "OCRed Text": rec.ocr_text,
                "source_path": _to_source_path(data_root, rec.image_path),
                "split": split_name,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.train_ratio <= 0.0 or args.train_ratio >= 1.0:
        raise ValueError("--train-ratio must be strictly between 0 and 1.")

    data_root = resolve_default_data_root(args.data_root)
    train_csvs = _split_arg(args.train_csvs)
    image_subdirs = _split_arg(args.image_subdirs)

    records = load_multi_ocr_sources(
        data_root=data_root,
        csv_paths=train_csvs,
        image_subdirs=image_subdirs,
    )

    records = sorted(records, key=lambda rec: _to_source_path(data_root, rec.image_path).lower())
    rng = random.Random(int(args.seed))
    rng.shuffle(records)

    total = len(records)
    train_count = int(total * float(args.train_ratio))
    train_records = records[:train_count]
    test_records = records[train_count:]

    output_dir = Path(args.output_dir).resolve()
    train_csv_path = output_dir / "batch1_train_split.csv"
    test_csv_path = output_dir / "batch1_test_split.csv"
    stats_path = output_dir / "batch1_split_stats.json"

    train_rows = _rows_from_records(train_records, data_root=data_root, split_name="train")
    test_rows = _rows_from_records(test_records, data_root=data_root, split_name="test")

    _write_csv(train_csv_path, train_rows)
    _write_csv(test_csv_path, test_rows)

    payload = {
        "data_root": str(data_root),
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "csv_sources": list(train_csvs),
        "image_subdirs": list(image_subdirs),
        "total_records": total,
        "train_records": len(train_rows),
        "test_records": len(test_rows),
        "train_csv_path": str(train_csv_path),
        "test_csv_path": str(test_csv_path),
    }
    stats_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stats_path": str(stats_path), "stats": payload}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
