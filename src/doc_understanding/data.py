from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class InvoiceRecord:
    img_name: str
    image_path: Path
    invoice_data: dict
    ocr_text: str


def _parse_json_data(raw: str) -> dict:
    payload = json.loads(raw)
    return dict(payload.get("invoice", {}))


def load_invoice_csv(csv_path: Path, image_dir: Path) -> List[InvoiceRecord]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    records: List[InvoiceRecord] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = str(row.get("File Name", "")).strip()
            if not img_name:
                continue
            invoice = _parse_json_data(str(row.get("Json Data", "{}")))
            image_path = (image_dir / img_name).resolve()
            if not image_path.exists():
                continue
            records.append(
                InvoiceRecord(
                    img_name=img_name,
                    image_path=image_path,
                    invoice_data=invoice,
                    ocr_text=str(row.get("OCRed Text", "")),
                )
            )
    if not records:
        raise ValueError(f"No valid records found in {csv_path}")
    return records


def load_default_train_eval(data_root: Path, train_csv: str, eval_csv: str, image_subdir_train: str, image_subdir_eval: str):
    train_records = load_invoice_csv(data_root / train_csv, data_root / image_subdir_train)
    eval_records = load_invoice_csv(data_root / eval_csv, data_root / image_subdir_eval)
    return train_records, eval_records
