from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a train-ready CSV from manually annotated real invoices/tickets."
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--input-csv", type=str, default="outputs/real_val_to_annotate.csv")
    parser.add_argument("--output-csv", type=str, default="outputs/real_train_annotated.csv")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=20,
        help="Drop too-short labels to reduce noisy pseudo-annotations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    input_csv = (data_root / args.input_csv).resolve()
    output_csv = (data_root / args.output_csv).resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    kept_rows: list[dict[str, str]] = []
    skipped_empty = 0
    skipped_missing_image = 0
    skipped_short = 0

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"File Name", "OCRed Text"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing required columns {sorted(missing)} in {input_csv}")

        for row in reader:
            file_name = str(row.get("File Name", "") or "").strip()
            text = str(row.get("OCRed Text", "") or "").strip()
            source_path = str(row.get("source_path", "") or "").strip()
            if not file_name or not text:
                skipped_empty += 1
                continue
            if len(text) < max(1, int(args.min_text_chars)):
                skipped_short += 1
                continue

            # Validate image reference when possible.
            image_exists = False
            if source_path:
                sp = Path(source_path)
                if sp.is_absolute():
                    image_exists = sp.exists()
                else:
                    image_exists = (data_root / sp).exists() or (input_csv.parent / sp).exists()
            if not image_exists:
                image_exists = (data_root / file_name).exists() or (input_csv.parent / file_name).exists()

            if not image_exists:
                skipped_missing_image += 1
                continue

            kept_rows.append(
                {
                    "File Name": file_name,
                    "OCRed Text": text,
                    "source_path": source_path,
                }
            )

    if args.max_rows and args.max_rows > 0:
        kept_rows = kept_rows[: args.max_rows]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["File Name", "OCRed Text", "source_path"])
        writer.writeheader()
        writer.writerows(kept_rows)

    print(
        json.dumps(
            {
                "input_csv": str(input_csv),
                "output_csv": str(output_csv),
                "num_rows": len(kept_rows),
                "skipped_empty": skipped_empty,
                "skipped_short": skipped_short,
                "skipped_missing_image": skipped_missing_image,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
