from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

CSV_COLUMNS = ["File Name", "OCRed Text", "source_path", "split"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic locked out-of-template split from annotated real OCR rows."
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--input-csv", type=str, default="outputs/real_train_annotated.csv")
    parser.add_argument("--train-output-csv", type=str, default="outputs/real_train_locked_split.csv")
    parser.add_argument("--holdout-output-csv", type=str, default="outputs/real_hors_template_locked_split.csv")
    parser.add_argument("--lock-json", type=str, default="outputs/real_hors_template_split_lock.json")
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--min-text-chars", type=int, default=20)
    parser.add_argument(
        "--group-key",
        type=str,
        default="auto",
        choices=("auto", "source_dir", "file_prefix"),
        help="Grouping strategy used to avoid leakage between train and holdout.",
    )
    return parser.parse_args()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _file_prefix(file_name: str) -> str:
    stem = Path(file_name).stem
    parts = re.split(r"[-_\s]+", stem)
    return (parts[0] if parts and parts[0] else stem).lower()


def _source_group(source_path: str, file_name: str) -> str:
    source = str(source_path or "").strip()
    if source:
        sp = Path(source)
        # Keep first meaningful folder token as grouping key.
        parts = [part for part in sp.parts if part and part not in (".", "..")]
        if parts:
            return parts[0].lower()
    parent_name = Path(file_name).parent.name
    if parent_name:
        return parent_name.lower()
    return _file_prefix(file_name)


def _group_key(row: dict[str, str], mode: str) -> str:
    file_name = _normalize_text(row.get("File Name", ""))
    source_path = _normalize_text(row.get("source_path", ""))
    if mode == "file_prefix":
        return _file_prefix(file_name)
    if mode == "source_dir":
        return _source_group(source_path, file_name)
    # auto
    return _source_group(source_path, file_name)


def _write_rows(path: Path, rows: list[dict[str, str]], split_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "File Name": _normalize_text(row.get("File Name", "")),
                    "OCRed Text": _normalize_text(row.get("OCRed Text", "")),
                    "source_path": _normalize_text(row.get("source_path", "")),
                    "split": split_name,
                }
            )


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    input_csv = (data_root / args.input_csv).resolve()
    train_csv = (data_root / args.train_output_csv).resolve()
    holdout_csv = (data_root / args.holdout_output_csv).resolve()
    lock_json = (data_root / args.lock_json).resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not (0.0 < float(args.holdout_ratio) < 1.0):
        raise ValueError("--holdout-ratio must be in (0,1)")

    rows: list[dict[str, str]] = []
    skipped_missing = 0
    skipped_short = 0

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"File Name", "OCRed Text"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing required columns {sorted(missing)} in {input_csv}")

        for row in reader:
            file_name = _normalize_text(row.get("File Name", ""))
            text = _normalize_text(row.get("OCRed Text", ""))
            if not file_name or not text:
                skipped_missing += 1
                continue
            if len(text) < max(1, int(args.min_text_chars)):
                skipped_short += 1
                continue
            rows.append(
                {
                    "File Name": file_name,
                    "OCRed Text": text,
                    "source_path": _normalize_text(row.get("source_path", "")),
                }
            )

    if len(rows) < 2:
        raise ValueError("Need at least 2 annotated rows to build a train/holdout split.")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row, args.group_key)].append(row)

    rng = random.Random(int(args.seed))
    group_keys = sorted(grouped.keys())
    rng.shuffle(group_keys)

    holdout_groups: set[str] = set()
    if len(group_keys) > 1:
        target_holdout = max(1, int(round(len(group_keys) * float(args.holdout_ratio))))
        target_holdout = min(target_holdout, len(group_keys) - 1)
        holdout_groups = set(group_keys[:target_holdout])

    train_rows: list[dict[str, str]] = []
    holdout_rows: list[dict[str, str]] = []

    if holdout_groups:
        for group_key_name, group_rows in grouped.items():
            if group_key_name in holdout_groups:
                holdout_rows.extend(group_rows)
            else:
                train_rows.extend(group_rows)
    else:
        # Fallback: if only one group exists, split rows deterministically.
        rows_sorted = sorted(rows, key=lambda item: item["File Name"].lower())
        rng.shuffle(rows_sorted)
        holdout_count = max(1, int(round(len(rows_sorted) * float(args.holdout_ratio))))
        holdout_count = min(holdout_count, len(rows_sorted) - 1)
        holdout_rows = rows_sorted[:holdout_count]
        train_rows = rows_sorted[holdout_count:]

    if not train_rows or not holdout_rows:
        raise RuntimeError("Failed to build non-empty train and holdout splits.")

    train_rows = sorted(train_rows, key=lambda item: item["File Name"].lower())
    holdout_rows = sorted(holdout_rows, key=lambda item: item["File Name"].lower())

    _write_rows(train_csv, train_rows, "train")
    _write_rows(holdout_csv, holdout_rows, "hors_template_holdout")

    lock_payload = {
        "schema_version": "real_hors_template_split.v1",
        "input_csv": str(input_csv),
        "train_output_csv": str(train_csv),
        "holdout_output_csv": str(holdout_csv),
        "num_rows_total": len(rows),
        "num_train_rows": len(train_rows),
        "num_holdout_rows": len(holdout_rows),
        "num_groups": len(grouped),
        "holdout_groups": sorted(holdout_groups),
        "seed": int(args.seed),
        "holdout_ratio": float(args.holdout_ratio),
        "group_key": args.group_key,
        "skipped_missing": skipped_missing,
        "skipped_short": skipped_short,
        "holdout_files": [row["File Name"] for row in holdout_rows],
    }
    lock_json.parent.mkdir(parents=True, exist_ok=True)
    lock_json.write_text(json.dumps(lock_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(lock_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
