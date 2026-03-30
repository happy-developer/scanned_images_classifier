from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List

TARGET_FIELDS = (
    "client_name",
    "client_address",
    "seller_name",
    "seller_address",
    "invoice_number",
    "invoice_date",
)

INSTRUCTION = (
    "Extract all information from this invoice image and return it in JSON format "
    "with the following fields: client_name, client_address, seller_name, "
    "seller_address, invoice_number, and invoice_date."
)


@dataclass(frozen=True)
class TargetInvoice:
    client_name: str
    client_address: str
    seller_name: str
    seller_address: str
    invoice_number: str
    invoice_date: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def normalize_address(value: str) -> str:
    return (value or "").replace("\n", ", ").strip()


def build_target_invoice(raw_invoice: dict) -> TargetInvoice:
    cleaned = dict(raw_invoice)
    cleaned["client_address"] = normalize_address(cleaned.get("client_address", ""))
    cleaned["seller_address"] = normalize_address(cleaned.get("seller_address", ""))
    return TargetInvoice(
        client_name=str(cleaned.get("client_name", "")).strip(),
        client_address=str(cleaned.get("client_address", "")).strip(),
        seller_name=str(cleaned.get("seller_name", "")).strip(),
        seller_address=str(cleaned.get("seller_address", "")).strip(),
        invoice_number=str(cleaned.get("invoice_number", "")).strip(),
        invoice_date=str(cleaned.get("invoice_date", "")).strip(),
    )


def record_to_messages(record: "InvoiceRecord", image_obj) -> List[dict]:
    response_text = build_target_invoice(record.invoice_data).to_json()
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": INSTRUCTION},
                {"type": "image", "image": image_obj},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}],
        },
    ]


def safe_extract_json(text: str) -> dict | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


def field_exact_match(pred: dict, target: dict, fields: Iterable[str] = TARGET_FIELDS) -> dict:
    result = {}
    for field in fields:
        pv = str(pred.get(field, "")).strip()
        tv = str(target.get(field, "")).strip()
        result[field] = int(pv == tv)
    return result
