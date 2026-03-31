from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

from .data import OCRRecord
from .formatting import normalize_text


def _levenshtein(a: Sequence[str] | str, b: Sequence[str] | str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            subst = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, subst))
        prev = cur
    return prev[-1]


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _extract_key_fields(text: str) -> Dict[str, str]:
    t = normalize_text(text).lower()
    one_line = re.sub(r"\s+", " ", t)

    invoice = ""
    m_invoice = re.search(r"invoice\s*(?:no|number|#)\s*[:\-]?\s*([a-z0-9\-\/]+)", one_line)
    if m_invoice:
        invoice = m_invoice.group(1).strip()

    date = ""
    m_date = re.search(r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b", one_line)
    if m_date:
        date = m_date.group(1).strip()

    vat = ""
    m_vat = re.search(r"vat[^%\n]{0,30}(\d{1,2}(?:[.,]\d+)?\s*%)", one_line)
    if not m_vat:
        m_vat = re.search(r"\b(\d{1,2}(?:[.,]\d+)?\s*%)\b", one_line)
    if m_vat:
        vat = re.sub(r"\s+", "", m_vat.group(1))

    total = ""
    m_total = re.search(
        r"(?:gross\s*worth|total|amount\s*due|net\s*worth)[^0-9]{0,20}([0-9]+(?:[.,][0-9]{2})?)",
        one_line,
    )
    if m_total:
        total = m_total.group(1).strip()

    return {
        "invoice_no": invoice,
        "date": date,
        "total": total,
        "vat": vat,
    }


def evaluate_records(records: Iterable[OCRRecord], predictions: Dict[str, str]) -> dict:
    rows: List[dict] = []
    field_names = ["invoice_no", "date", "total", "vat"]
    field_correct = {name: 0 for name in field_names}
    field_covered = {name: 0 for name in field_names}

    for rec in records:
        target = normalize_text(rec.ocr_text)
        pred = normalize_text(predictions.get(rec.img_name, ""))
        char_edits = _levenshtein(pred, target)
        target_chars = max(len(target), 1)
        pred_words = pred.split()
        target_words = target.split()
        word_edits = _levenshtein(pred_words, target_words)
        target_word_count = max(len(target_words), 1)

        target_fields = _extract_key_fields(target)
        pred_fields = _extract_key_fields(pred)
        for field in field_names:
            target_value = target_fields.get(field, "")
            pred_value = pred_fields.get(field, "")
            if target_value:
                field_covered[field] += 1
                field_correct[field] += int(pred_value == target_value)

        rows.append(
            {
                "img_name": rec.img_name,
                "exact_match": int(pred == target),
                "cer": _safe_div(char_edits, target_chars),
                "char_accuracy": max(0.0, 1.0 - _safe_div(char_edits, target_chars)),
                "word_accuracy": max(0.0, 1.0 - _safe_div(word_edits, target_word_count)),
            }
        )

    if not rows:
        return {"num_samples": 0, "exact_match": 0.0, "avg_char_accuracy": 0.0, "avg_word_accuracy": 0.0}

    return {
        "num_samples": len(rows),
        "exact_match": sum(r["exact_match"] for r in rows) / len(rows),
        "avg_cer": sum(r["cer"] for r in rows) / len(rows),
        "avg_char_accuracy": sum(r["char_accuracy"] for r in rows) / len(rows),
        "avg_word_accuracy": sum(r["word_accuracy"] for r in rows) / len(rows),
        "field_exact_match": {
            field: _safe_div(field_correct[field], field_covered[field]) for field in field_names
        },
        "field_coverage": field_covered,
    }
