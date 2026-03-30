from __future__ import annotations

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


def evaluate_records(records: Iterable[OCRRecord], predictions: Dict[str, str]) -> dict:
    rows: List[dict] = []
    for rec in records:
        target = normalize_text(rec.ocr_text)
        pred = normalize_text(predictions.get(rec.img_name, ""))
        char_edits = _levenshtein(pred, target)
        target_chars = max(len(target), 1)
        pred_words = pred.split()
        target_words = target.split()
        word_edits = _levenshtein(pred_words, target_words)
        target_word_count = max(len(target_words), 1)

        rows.append(
            {
                "img_name": rec.img_name,
                "exact_match": int(pred == target),
                "char_accuracy": max(0.0, 1.0 - _safe_div(char_edits, target_chars)),
                "word_accuracy": max(0.0, 1.0 - _safe_div(word_edits, target_word_count)),
            }
        )

    if not rows:
        return {"num_samples": 0, "exact_match": 0.0, "avg_char_accuracy": 0.0, "avg_word_accuracy": 0.0}

    return {
        "num_samples": len(rows),
        "exact_match": sum(r["exact_match"] for r in rows) / len(rows),
        "avg_char_accuracy": sum(r["char_accuracy"] for r in rows) / len(rows),
        "avg_word_accuracy": sum(r["word_accuracy"] for r in rows) / len(rows),
    }
