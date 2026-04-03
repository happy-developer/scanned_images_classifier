from __future__ import annotations

import re
from typing import Dict, Iterable, Sequence

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


def _tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"\s+", normalize_text(text).strip().lower()) if token]


def repetition_rate(text: str) -> float:
    normalized = normalize_text(text)
    chars = [ch for ch in normalized if not ch.isspace()]
    char_rate = 0.0
    if chars:
        char_rate = _safe_div(len(chars) - len(set(chars)), len(chars))

    tokens = _tokenize(text)
    token_rate = 0.0
    if tokens:
        token_rate = _safe_div(len(tokens) - len(set(tokens)), len(tokens))

    # Heuristic that captures both repeated characters and repeated tokens.
    return max(char_rate, token_rate)


def evaluate_records(
    records: Iterable[OCRRecord],
    predictions: Dict[str, str],
    *,
    compute_wer: bool = True,
) -> dict:
    records = list(records)
    if not records:
        payload: dict[str, object] = {
            "mode": "supervised",
            "num_samples": 0,
            "cer": 0.0,
            "non_empty_rate": 0.0,
            "avg_prediction_chars": 0.0,
            "avg_reference_chars": 0.0,
            "prediction_reference_length_ratio": 0.0,
            "repetition_rate": 0.0,
            "total_prediction_chars": 0,
            "total_reference_chars": 0,
        }
        if compute_wer:
            payload["wer"] = 0.0
        return payload

    total_char_edits = 0
    total_ref_chars = 0
    total_pred_chars = 0
    total_word_edits = 0
    total_ref_words = 0
    non_empty = 0
    repetition_sum = 0.0

    for rec in records:
        target = normalize_text(rec.ocr_text)
        pred = normalize_text(predictions.get(rec.img_name, ""))

        total_char_edits += _levenshtein(pred, target)
        total_pred_chars += len(pred)
        total_ref_chars += len(target)
        non_empty += int(bool(pred.strip()))
        repetition_sum += repetition_rate(pred)

        if compute_wer:
            target_words = _tokenize(target)
            pred_words = _tokenize(pred)
            total_word_edits += _levenshtein(pred_words, target_words)
            total_ref_words += len(target_words)

    num_samples = len(records)
    payload = {
        "mode": "supervised",
        "num_samples": num_samples,
        "cer": _safe_div(total_char_edits, total_ref_chars),
        "non_empty_rate": _safe_div(non_empty, num_samples),
        "avg_prediction_chars": _safe_div(total_pred_chars, num_samples),
        "avg_reference_chars": _safe_div(total_ref_chars, num_samples),
        "prediction_reference_length_ratio": _safe_div(total_pred_chars, total_ref_chars),
        "repetition_rate": _safe_div(repetition_sum, num_samples),
        "total_prediction_chars": total_pred_chars,
        "total_reference_chars": total_ref_chars,
    }
    if compute_wer:
        payload["wer"] = _safe_div(total_word_edits, total_ref_words)
    return payload


def summarize_predictions(predictions: Iterable[str]) -> dict:
    normalized_predictions = [normalize_text(pred) for pred in predictions]
    num_samples = len(normalized_predictions)
    total_pred_chars = sum(len(pred) for pred in normalized_predictions)
    non_empty = sum(int(bool(pred.strip())) for pred in normalized_predictions)
    repetition = _safe_div(sum(repetition_rate(pred) for pred in normalized_predictions), num_samples)

    return {
        "mode": "unlabeled",
        "num_samples": num_samples,
        "non_empty_rate": _safe_div(non_empty, num_samples),
        "avg_prediction_chars": _safe_div(total_pred_chars, num_samples),
        "repetition_rate": repetition,
        "total_prediction_chars": total_pred_chars,
    }
