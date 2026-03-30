from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from .data import InvoiceRecord
from .formatting import build_target_invoice, field_exact_match


def evaluate_records(records: Iterable[InvoiceRecord], predictions: Dict[str, dict]) -> dict:
    rows: List[dict] = []
    for rec in records:
        target = build_target_invoice(rec.invoice_data).__dict__
        pred = predictions.get(rec.img_name, {})
        match = field_exact_match(pred, target)
        rows.append({"img_name": rec.img_name, **match})

    if not rows:
        return {"num_samples": 0, "field_accuracy": {}, "overall_exact_match": 0.0}

    fields = [k for k in rows[0].keys() if k != "img_name"]
    field_accuracy = {f: sum(r[f] for r in rows) / len(rows) for f in fields}
    overall = sum(1 for r in rows if all(r[f] == 1 for f in fields)) / len(rows)

    return {
        "num_samples": len(rows),
        "field_accuracy": field_accuracy,
        "overall_exact_match": overall,
    }
