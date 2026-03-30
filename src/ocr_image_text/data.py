from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class OCRRecord:
    img_name: str
    image_path: Path
    ocr_text: str


def load_ocr_csv(csv_path: Path, image_dir: Path) -> List[OCRRecord]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    records: List[OCRRecord] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = str(row.get("File Name", "")).strip()
            text = str(row.get("OCRed Text", "")).strip()
            if not img_name or not text:
                continue
            image_path = (image_dir / img_name).resolve()
            if not image_path.exists():
                continue
            records.append(OCRRecord(img_name=img_name, image_path=image_path, ocr_text=text))

    if not records:
        raise ValueError(f"No valid OCR records found in {csv_path}")
    return records


def load_default_train_eval(
    data_root: Path,
    train_csv: str,
    eval_csv: str,
    image_subdir_train: str,
    image_subdir_eval: str,
) -> Tuple[List[OCRRecord], List[OCRRecord]]:
    train_records = load_ocr_csv(data_root / train_csv, data_root / image_subdir_train)
    eval_records = load_ocr_csv(data_root / eval_csv, data_root / image_subdir_eval)
    return train_records, eval_records


def resolve_default_data_root(cli_data_root: str | None = None) -> Path:
    candidates = []
    if cli_data_root:
        candidates.append(Path(cli_data_root))
    candidates.extend([
        Path("data/kaggle_invoice_images"),
        Path("notebooks/data/kaggle_invoice_images"),
    ])
    for candidate in candidates:
        p = candidate.resolve()
        if p.exists():
            return p
    raise FileNotFoundError("No dataset root found. Tried data/kaggle_invoice_images and notebooks/data/kaggle_invoice_images.")
