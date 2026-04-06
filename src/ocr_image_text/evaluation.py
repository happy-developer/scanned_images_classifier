from __future__ import annotations

from datetime import datetime
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


_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "invoice_no": (
        r"invoice\s*(?:no\.?|number|n[o#])",
        r"inv(?:oice)?\s*(?:no\.?|number)",
    ),
    "date_of_issue": (
        r"date\s*(?:of\s*issue|issue)",
        r"issue\s*date",
        r"invoice\s*date",
        r"date\s*of\s*invoice",
        r"date",
    ),
    "tax_id": (
        r"tax\s*id",
        r"tax\s*identification\s*number",
        r"vat(?:\s*(?:id|no\.?|number))?",
        r"tva",
        r"tin",
        r"nif",
        r"siret",
        r"siren",
    ),
    "iban": (r"iban",),
}

_MONTHS = {
    "jan": 1,
    "january": 1,
    "janvier": 1,
    "feb": 2,
    "february": 2,
    "fev": 2,
    "fevrier": 2,
    "mar": 3,
    "march": 3,
    "mars": 3,
    "apr": 4,
    "april": 4,
    "avr": 4,
    "avril": 4,
    "may": 5,
    "mai": 5,
    "jun": 6,
    "june": 6,
    "juin": 6,
    "jul": 7,
    "july": 7,
    "juil": 7,
    "juillet": 7,
    "aug": 8,
    "august": 8,
    "aou": 8,
    "aout": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "septembre": 9,
    "oct": 10,
    "october": 10,
    "octobre": 10,
    "nov": 11,
    "november": 11,
    "novembre": 11,
    "dec": 12,
    "december": 12,
    "decembre": 12,
}

_DATE_PATTERNS = (
    re.compile(r"\b(?P<year>\d{4})[-/.](?P<month>\d{1,2})[-/.](?P<day>\d{1,2})\b"),
    re.compile(r"\b(?P<day>\d{1,2})[-/.](?P<month>\d{1,2})[-/.](?P<year>\d{4})\b"),
    re.compile(r"\b(?P<day>\d{1,2})\s+(?P<month>[a-zA-Z]+)\s+(?P<year>\d{4})\b"),
)

_IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s*[A-Z0-9]{4}){3,7}\b", re.IGNORECASE)


def _compact_field_value(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def _normalize_date_candidate(value: str) -> str:
    text = normalize_text(value).lower()
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groupdict()
        day = int(groups["day"])
        year = int(groups["year"])
        month_raw = groups["month"]
        if month_raw.isdigit():
            month = int(month_raw)
        else:
            month = _MONTHS.get(month_raw, 0)
            if not month:
                month = _MONTHS.get(month_raw[:3], 0)
        if not month:
            continue
        try:
            return datetime(year, month, day).date().isoformat()
        except ValueError:
            continue
    return ""


def _normalize_field_value(field_name: str, value: str) -> str:
    normalized_value = normalize_text(value)
    if not normalized_value:
        return ""
    if field_name == "date_of_issue":
        return _normalize_date_candidate(normalized_value)
    if field_name == "iban":
        return _compact_field_value(normalized_value)
    return _compact_field_value(normalized_value)


def _extract_field_value_from_text(field_name: str, text: str) -> str:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return ""

    aliases = _FIELD_ALIASES.get(field_name, ())
    for alias in aliases:
        pattern = re.compile(
            rf"(?is)(?<!\w)(?:{alias})(?!\w)(?:\s*[:=#-]\s*|\s+)?(?P<value>[^\n|;]+)"
        )
        match = pattern.search(normalized_text)
        if match:
            normalized_value = _normalize_field_value(field_name, match.group("value"))
            if normalized_value:
                return normalized_value

    if field_name == "iban":
        match = _IBAN_PATTERN.search(normalized_text)
        if match:
            return _compact_field_value(match.group(0))

    if field_name == "date_of_issue":
        # Keep date extraction permissive across the full text because date labels vary a lot.
        candidate = _normalize_date_candidate(normalized_text)
        return candidate if candidate else ""

    return ""


def _evaluate_field_metrics(
    records: Sequence[OCRRecord],
    predictions: Dict[str, str],
) -> tuple[float, dict[str, float], dict[str, float]]:
    fields = tuple(_FIELD_ALIASES.keys())
    covered_counts = {field: 0 for field in fields}
    correct_counts = {field: 0 for field in fields}

    for rec in records:
        target = normalize_text(rec.ocr_text)
        pred = normalize_text(predictions.get(rec.img_name, ""))

        for field in fields:
            target_value = _extract_field_value_from_text(field, target)
            pred_value = _extract_field_value_from_text(field, pred)
            if not target_value:
                continue
            covered_counts[field] += 1
            if target_value == pred_value:
                correct_counts[field] += 1

    overall_covered = sum(covered_counts.values())
    overall_correct = sum(correct_counts.values())
    exact_match_by_name = {
        field: _safe_div(correct_counts[field], covered_counts[field]) for field in fields
    }
    coverage_by_name = {
        field: _safe_div(covered_counts[field], len(records)) for field in fields
    }
    return _safe_div(overall_correct, overall_covered), exact_match_by_name, coverage_by_name


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
            "field_exact_match_overall": 0.0,
            "field_exact_match_by_name": {},
            "field_coverage_by_name": {},
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
    field_exact_match_overall, field_exact_match_by_name, field_coverage_by_name = _evaluate_field_metrics(
        records, predictions
    )
    payload = {
        "mode": "supervised",
        "num_samples": num_samples,
        "cer": _safe_div(total_char_edits, total_ref_chars),
        "field_exact_match_overall": field_exact_match_overall,
        "field_exact_match_by_name": field_exact_match_by_name,
        "field_coverage_by_name": field_coverage_by_name,
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
