from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .config import TARGET_FIELDS

PROMPT_TEMPLATE = (
    "Extract invoice fields as JSON with keys "
    "client_name, client_address, seller_name, seller_address, invoice_number, invoice_date.\n"
    "Invoice OCR text:\n{ocr_text}"
)


@dataclass(frozen=True)
class CPURecord:
    img_name: str
    ocr_text: str
    target_json: str
    target_dict: dict


def _normalize_invoice(invoice: dict) -> dict:
    out = {}
    for field in TARGET_FIELDS:
        val = str(invoice.get(field, "")).replace("\n", ", ").strip()
        out[field] = val
    return out


def _parse_row(row: dict) -> CPURecord:
    invoice = json.loads(str(row.get("Json Data", "{}"))).get("invoice", {})
    target = _normalize_invoice(invoice)
    return CPURecord(
        img_name=str(row.get("File Name", "")).strip(),
        ocr_text=str(row.get("OCRed Text", "")).strip(),
        target_json=json.dumps(target, ensure_ascii=False),
        target_dict=target,
    )


def load_cpu_records(csv_path: Path) -> List[CPURecord]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    records: List[CPURecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = _parse_row(row)
            if rec.ocr_text:
                records.append(rec)
    if not records:
        raise ValueError(f"No valid OCR records found in {csv_path}")
    return records


def records_to_text2text(records: Iterable[CPURecord]) -> list[dict]:
    dataset = []
    for rec in records:
        dataset.append(
            {
                "input_text": PROMPT_TEMPLATE.format(ocr_text=rec.ocr_text),
                "target_text": rec.target_json,
                "img_name": rec.img_name,
            }
        )
    return dataset
