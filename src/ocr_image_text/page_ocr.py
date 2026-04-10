from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

from .formatting import normalize_text


@dataclass(frozen=True)
class CropRegion:
    box: Tuple[int, int, int, int]
    label: str


@dataclass(frozen=True)
class SegmentationPlan:
    crop_regions: List[CropRegion]
    used_full_page_fallback: bool
    strategy: str
    fallback_reason: str | None = None
    original_crop_count: int = 0
    deduplicated_crop_count: int = 0
    duplicate_crop_count: int = 0


def _ensure_rgb(image: Image.Image, use_grayscale: bool) -> Image.Image:
    converted = image.convert("RGB")
    if use_grayscale:
        converted = converted.convert("L").convert("RGB")
    return converted


def _otsu_threshold(gray: np.ndarray) -> int:
    values = np.asarray(gray, dtype=np.uint8).ravel()
    if values.size == 0:
        return 0

    hist = np.bincount(values, minlength=256).astype(np.float64)
    total = float(values.size)
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = float(mu[-1])

    denominator = omega * (1.0 - omega)
    denominator[denominator == 0.0] = np.nan
    sigma_b_sq = (mu_t * omega - mu) ** 2 / denominator
    if np.all(np.isnan(sigma_b_sq)):
        return int(np.median(values))

    return int(np.nanargmax(sigma_b_sq))


def _find_spans(values: Sequence[int], minimum: int) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start: int | None = None
    for index, score in enumerate(values):
        if score >= minimum:
            if start is None:
                start = index
            continue
        if start is not None:
            spans.append((start, index - 1))
            start = None
    if start is not None:
        spans.append((start, len(values) - 1))
    return spans


def _merge_spans(spans: Sequence[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    if not spans:
        return []

    merged: List[Tuple[int, int]] = [tuple(spans[0])]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 <= max_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _expand_box(
    left: int,
    top: int,
    right: int,
    bottom: int,
    width: int,
    height: int,
    pad_x: int,
    pad_y: int,
) -> Tuple[int, int, int, int]:
    expanded_left = max(0, left - pad_x)
    expanded_top = max(0, top - pad_y)
    expanded_right = min(width, right + pad_x)
    expanded_bottom = min(height, bottom + pad_y)
    return expanded_left, expanded_top, expanded_right, expanded_bottom


def _box_area(box: Tuple[int, int, int, int]) -> int:
    left, top, right, bottom = box
    return max(0, right - left) * max(0, bottom - top)



def _box_intersection_area(
    left_box: Tuple[int, int, int, int],
    right_box: Tuple[int, int, int, int],
) -> int:
    left = max(left_box[0], right_box[0])
    top = max(left_box[1], right_box[1])
    right = min(left_box[2], right_box[2])
    bottom = min(left_box[3], right_box[3])
    return _box_area((left, top, right, bottom))


def _boxes_too_similar(
    left_box: Tuple[int, int, int, int],
    right_box: Tuple[int, int, int, int],
    *,
    overlap_threshold: float = 0.95,
    iou_threshold: float = 0.85,
) -> bool:
    left_area = _box_area(left_box)
    right_area = _box_area(right_box)
    if left_area <= 0 or right_area <= 0:
        return False

    intersection = _box_intersection_area(left_box, right_box)
    if intersection <= 0:
        return False

    min_area = float(min(left_area, right_area))
    union_area = float(left_area + right_area - intersection)
    overlap_ratio = intersection / max(1.0, min_area)
    iou = intersection / max(1.0, union_area)
    return overlap_ratio >= overlap_threshold or iou >= iou_threshold


def _dedupe_crop_regions(regions: Sequence[CropRegion]) -> Tuple[List[CropRegion], int]:
    deduped: List[CropRegion] = []
    skipped = 0
    for region in regions:
        if deduped and _boxes_too_similar(deduped[-1].box, region.box):
            skipped += 1
            continue
        deduped.append(region)
    return deduped, skipped


def _text_similarity_key(text: str) -> str:
    normalized = normalize_text(text).casefold()
    return re.sub(r'[^0-9a-z]+', ' ', normalized).strip()


def _text_structure_key(text: str) -> str:
    key = _text_similarity_key(text)
    # Ignore numeric variation when comparing repeated invoice-like hallucinations.
    key = re.sub(r"\d", "0", key)
    return re.sub(r"\s+", " ", key).strip()


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[0-9a-z]+", _text_similarity_key(left)))
    right_tokens = set(re.findall(r"[0-9a-z]+", _text_similarity_key(right)))
    if not left_tokens or not right_tokens:
        return 0.0
    inter = len(left_tokens.intersection(right_tokens))
    union = len(left_tokens.union(right_tokens))
    return float(inter) / float(max(1, union))


def _is_near_duplicate_text(previous: str, current: str) -> bool:
    previous_key = _text_similarity_key(previous)
    current_key = _text_similarity_key(current)
    if not previous_key or not current_key:
        return False
    if previous_key == current_key:
        return True
    if _text_structure_key(previous_key) == _text_structure_key(current_key):
        return True

    previous_len = len(previous_key)
    current_len = len(current_key)
    if min(previous_len, current_len) < 12:
        return False

    ratio = SequenceMatcher(None, previous_key, current_key).ratio()
    jaccard = _token_jaccard(previous_key, current_key)
    return ratio >= 0.92 or jaccard >= 0.9


def _field_label_counts(text: str) -> Dict[str, int]:
    normalized = normalize_text(text).casefold()
    if not normalized:
        return {"invoice": 0, "date": 0, "tax_id": 0, "iban": 0, "items": 0}
    return {
        "invoice": len(re.findall(r"\binvoice(?:\s*(?:no\.?|number|n[o#]))?\b", normalized)),
        "date": len(re.findall(r"\bdate(?:\s*of\s*issue)?\b", normalized)),
        "tax_id": len(re.findall(r"\btax\s*id\b", normalized)),
        "iban": len(re.findall(r"\biban\b", normalized)),
        "items": len(re.findall(r"\bitems?\b", normalized)),
    }


def _field_hit_count(text: str) -> int:
    counts = _field_label_counts(text)
    return sum(1 for value in counts.values() if value > 0)


def _has_repeated_key_fields(text: str) -> bool:
    counts = _field_label_counts(text)
    return bool(
        counts["tax_id"] > 1
        or counts["iban"] > 1
        or counts["invoice"] > 1
    )


def _dedupe_neighboring_text_segments(segment_texts: Sequence[str]) -> Tuple[List[str], int]:
    deduped: List[str] = []
    skipped = 0
    for text in segment_texts:
        normalized = normalize_text(text)
        if not normalized:
            continue
        if any(_is_near_duplicate_text(existing, normalized) for existing in deduped):
            skipped += 1
            continue
        deduped.append(normalized)
    return deduped, skipped


def _filter_noisy_segments(segment_texts: Sequence[str]) -> Tuple[List[str], int]:
    filtered: List[str] = []
    skipped = 0
    field_heavy_seen = False
    for text in segment_texts:
        normalized = normalize_text(text)
        if not normalized:
            continue
        if _has_repeated_key_fields(normalized):
            skipped += 1
            continue
        field_hits = _field_hit_count(normalized)
        if field_hits >= 3:
            if field_heavy_seen:
                skipped += 1
                continue
            field_heavy_seen = True
        filtered.append(normalized)
    return filtered, skipped


def _full_page_plan(page_size: Tuple[int, int], reason: str) -> SegmentationPlan:
    width, height = page_size
    return SegmentationPlan(
        crop_regions=[CropRegion(box=(0, 0, width, height), label="full_page")],
        used_full_page_fallback=True,
        strategy="full_page_fallback",
        fallback_reason=reason,
        original_crop_count=1,
        deduplicated_crop_count=1,
        duplicate_crop_count=0,
    )


def segment_page(
    image: Image.Image,
    *,
    segmentation_mode: str = "line_only",
    max_regions: int = 32,
) -> SegmentationPlan:
    rgb = image.convert("RGB")
    width, height = rgb.size
    mode = str(segmentation_mode or "line_only").strip().lower()
    if mode not in {"line_only", "line_block", "full_page"}:
        mode = "line_only"

    if mode == "full_page":
        return SegmentationPlan(
            crop_regions=[CropRegion(box=(0, 0, width, height), label="full_page")],
            used_full_page_fallback=False,
            strategy="full_page",
            fallback_reason=None,
            original_crop_count=1,
            deduplicated_crop_count=1,
            duplicate_crop_count=0,
        )

    if width <= 1 or height <= 1:
        return _full_page_plan((width, height), "image_too_small")

    gray = ImageOps.autocontrast(rgb.convert("L"))
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray_array = np.asarray(gray, dtype=np.uint8)
    if gray_array.size == 0:
        return _full_page_plan((width, height), "empty_image_array")

    threshold = _otsu_threshold(gray_array)
    ink_mask = gray_array <= threshold
    if not np.any(ink_mask):
        return _full_page_plan((width, height), "no_ink_detected")

    mask_image = Image.fromarray((ink_mask.astype(np.uint8) * 255), mode="L")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(3))
    mask = np.asarray(mask_image) > 0
    if not np.any(mask):
        return _full_page_plan((width, height), "mask_cleared_by_dilation")

    row_scores = mask.sum(axis=1)
    row_threshold = max(2, int(width * 0.005))
    row_spans = _merge_spans(_find_spans(row_scores, row_threshold), max_gap=2)
    if not row_spans:
        return _full_page_plan((width, height), "no_text_rows_detected")

    crop_regions: List[CropRegion] = []
    scored_regions: List[Tuple[float, int, CropRegion]] = []
    area_floor = max(256, int(width * height * 0.0005))
    pad_x = max(6, width // 80)
    pad_y = max(4, height // 100)
    min_band_height = max(8, height // 320)
    adaptive_max_regions = max(8, min(int(max_regions), max(8, height // 70)))

    for y0, y1 in row_spans:
        band = mask[y0 : y1 + 1, :]
        band_height = max(1, y1 - y0 + 1)
        if band_height < min_band_height:
            continue
        band_density = float(band.sum()) / float(max(1, band_height * width))
        if band_density < 0.004:
            continue

        if mode == "line_only":
            box = _expand_box(0, y0, width, y1 + 1, width, height, pad_x=pad_x, pad_y=pad_y)
            if _box_area(box) >= area_floor:
                region = CropRegion(box=box, label="line")
                score = band_density * max(1.0, float(band_height))
                scored_regions.append((score, y0, region))
        else:
            col_scores = band.sum(axis=0)
            col_threshold = max(2, int(band_height * 0.01))
            col_spans = _merge_spans(_find_spans(col_scores, col_threshold), max_gap=max(3, width // 100))
            if not col_spans:
                col_spans = [(0, width - 1)]

            for x0, x1 in col_spans:
                box = _expand_box(x0, y0, x1 + 1, y1 + 1, width, height, pad_x=pad_x, pad_y=pad_y)
                if _box_area(box) < area_floor:
                    continue
                region = CropRegion(box=box, label="block" if len(col_spans) > 1 else "line")
                box_width = max(1, x1 - x0 + 1)
                score = band_density * max(1.0, float(band_height)) * (float(box_width) / float(max(1, width)))
                scored_regions.append((score, y0, region))

    if scored_regions:
        # Keep the most content-dense regions first, then restore reading order.
        top_scored = sorted(scored_regions, key=lambda item: item[0], reverse=True)[:adaptive_max_regions]
        crop_regions = [region for _, _, region in sorted(top_scored, key=lambda item: item[1])]

    if not crop_regions:
        return _full_page_plan((width, height), "no_usable_crops")

    original_crop_count = len(crop_regions)
    deduplicated_crop_regions, duplicate_crop_count = _dedupe_crop_regions(crop_regions)
    if not deduplicated_crop_regions:
        return _full_page_plan((width, height), "no_usable_crops")

    total_crop_area = sum(_box_area(region.box) for region in deduplicated_crop_regions)
    page_area = width * height
    if total_crop_area <= 0 or total_crop_area / max(1, page_area) < 0.01:
        return _full_page_plan((width, height), "crop_coverage_too_small")

    return SegmentationPlan(
        crop_regions=deduplicated_crop_regions,
        used_full_page_fallback=False,
        strategy="line_only_crops" if mode == "line_only" else "line_block_crops",
        fallback_reason=None,
        original_crop_count=original_crop_count,
        deduplicated_crop_count=len(deduplicated_crop_regions),
        duplicate_crop_count=duplicate_crop_count,
    )


def _truncate_text(text: str, max_chars: int) -> str:
    max_chars = max(0, int(max_chars))
    if max_chars <= 0:
        return ""
    normalized = normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip()


def _count_invoice_markers(text: str) -> int:
    normalized = normalize_text(text).casefold()
    if not normalized:
        return 0

    patterns = (
        r"\binvoice(?:\s*(?:no\.?|number|n[o#]|date|id))?\b",
        r"\bfacture(?:\s*(?:no\.?|number|n[o#]|date|id))?\b",
    )
    return sum(len(list(re.finditer(pattern, normalized))) for pattern in patterns)


def _collapse_repeated_tokens(text: str, *, max_consecutive: int = 2) -> str:
    lines = [line.strip() for line in normalize_text(text).split("\n") if line.strip()]
    if not lines:
        return ""

    compact_lines: List[str] = []
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        out: List[str] = []
        previous = ""
        streak = 0
        for token in tokens:
            key = token.casefold()
            if key == previous:
                streak += 1
            else:
                previous = key
                streak = 1
            if streak <= max(1, int(max_consecutive)):
                out.append(token)
        if out:
            compact_lines.append(" ".join(out).strip())

    return "\n".join(compact_lines).strip()


def _collapse_repeated_ngrams(text: str) -> str:
    # Collapse repeated short phrase loops that often appear in OCR hallucinations.
    normalized = normalize_text(text)
    if not normalized:
        return ""

    cleaned = normalized
    for ngram_len in (6, 5, 4, 3, 2):
        # Example matched sequence: "a b c a b c a b c" -> "a b c"
        pattern = re.compile(
            rf"(?i)\b((?:\w+\s+){{{ngram_len - 1}}}\w+)(?:\s+\1)+\b"
        )
        cleaned = pattern.sub(r"\1", cleaned)
    return cleaned.strip()


def _dedupe_lines_global(text: str) -> str:
    lines = [line.strip() for line in normalize_text(text).split("\n") if line.strip()]
    if not lines:
        return ""

    deduped: List[str] = []
    seen_keys: set[str] = set()
    for line in lines:
        key = _text_similarity_key(line)
        if not key:
            continue
        if key in seen_keys:
            continue
        if deduped and _is_near_duplicate_text(deduped[-1], line):
            continue
        deduped.append(line)
        seen_keys.add(key)
    return "\n".join(deduped).strip()


def _remove_noisy_field_lines(text: str) -> str:
    lines = [line.strip() for line in normalize_text(text).split("\n") if line.strip()]
    if not lines:
        return ""

    kept: List[str] = []
    heavy_line_seen = False
    for line in lines:
        if _has_repeated_key_fields(line):
            continue
        field_hits = _field_hit_count(line)
        if field_hits >= 3:
            if heavy_line_seen:
                continue
            heavy_line_seen = True
        kept.append(line)
    return "\n".join(kept).strip()


def _limit_invoice_sections(text: str, max_invoice_markers: int) -> str:
    marker_limit = max(1, int(max_invoice_markers))
    normalized = normalize_text(text)
    if not normalized:
        return ""

    marker_pattern = re.compile(
        r"(?i)\b(?:invoice(?:\s*(?:no\.?|number|n[o#]|date|id))?|facture(?:\s*(?:no\.?|number|n[o#]|date|id))?)\b"
    )
    matches = list(marker_pattern.finditer(normalized))
    if len(matches) <= marker_limit:
        return normalized
    cutoff = matches[marker_limit].start()
    return normalized[:cutoff].rstrip()


def _postprocess_prediction_text(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return ""
    cleaned = _remove_noisy_field_lines(cleaned)
    cleaned = _dedupe_lines_global(cleaned)
    cleaned = _collapse_repeated_ngrams(cleaned)
    cleaned = _collapse_repeated_tokens(cleaned, max_consecutive=2)
    return normalize_text(cleaned)


def _build_generation_kwargs(
    *,
    max_new_tokens: int,
    num_beams: int,
    temperature: float,
    length_penalty: float,
    no_repeat_ngram_size: int,
    repetition_penalty: float,
    tokenizer: Any | None,
) -> Dict[str, Any]:
    suppress_tokens: List[int] = []
    if tokenizer is not None:
        if getattr(tokenizer, "bos_token_id", None) is not None:
            suppress_tokens.append(int(tokenizer.bos_token_id))
        if getattr(tokenizer, "pad_token_id", None) is not None:
            suppress_tokens.append(int(tokenizer.pad_token_id))

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "num_beams": max(1, int(num_beams)),
        "length_penalty": float(length_penalty),
        "no_repeat_ngram_size": max(0, int(no_repeat_ngram_size)),
        "repetition_penalty": float(repetition_penalty),
        "do_sample": float(temperature) > 0.0,
    }
    if suppress_tokens:
        gen_kwargs["suppress_tokens"] = suppress_tokens
    if float(temperature) > 0.0:
        gen_kwargs["temperature"] = float(temperature)
    return gen_kwargs


def _decode_outputs(processor: Any, output_ids: Any) -> List[str]:
    try:
        decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
    except Exception as exc:  # pragma: no cover - surfaced to callers
        raise RuntimeError(f"Failed to decode OCR outputs: {exc}") from exc
    return [str(item).strip() for item in decoded]


def _run_model_on_images(
    model: Any,
    processor: Any,
    images: Sequence[Image.Image],
    *,
    generation_kwargs: Dict[str, Any],
) -> List[str]:
    if not images:
        return []

    try:
        inputs = processor(images=list(images), return_tensors="pt")
    except Exception as exc:  # pragma: no cover - surfaced to callers
        raise RuntimeError(f"Failed to preprocess OCR crops: {exc}") from exc

    with torch.no_grad():
        output_ids = model.generate(pixel_values=inputs.pixel_values, **generation_kwargs)
    return _decode_outputs(processor, output_ids)


def _run_crop_first_ocr(
    model: Any,
    processor: Any,
    image: Image.Image,
    *,
    segmentation_mode: str = "line_only",
    max_new_tokens: int,
    num_beams: int,
    temperature: float,
    length_penalty: float,
    no_repeat_ngram_size: int,
    repetition_penalty: float,
    max_chars_per_segment: int = 256,
    max_total_chars: int = 1200,
    max_invoice_markers_per_page: int = 1,
    hard_truncate_segment_text: bool = True,
    batch_size: int = 6,
    max_crops: int = 28,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    page_image = image.convert("RGB")
    requested_mode = str(segmentation_mode or "line_only").strip().lower()
    # TrOCR behaves more reliably on line-level OCR than on larger block/page generation.
    effective_mode = "line_only" if requested_mode in {"line_only", "line_block"} else requested_mode
    plan = segment_page(page_image, segmentation_mode=effective_mode, max_regions=max(4, int(max_crops)))

    def _ocr_segments(regions: Sequence[CropRegion]) -> List[str]:
        generation_kwargs = _build_generation_kwargs(
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            tokenizer=getattr(processor, "tokenizer", None),
        )
        decoded_segments: List[str] = []
        for start in range(0, len(regions), max(1, int(batch_size))):
            batch = regions[start : start + max(1, int(batch_size))]
            batch_images = [page_image.crop(region.box) for region in batch]
            decoded_segments.extend(
                _run_model_on_images(
                    model,
                    processor,
                    batch_images,
                    generation_kwargs=generation_kwargs,
                )
            )
        return decoded_segments

    def _extract_output(
        active_plan: SegmentationPlan,
    ) -> tuple[List[str], List[str], int, int, str, str, int, int, int, bool]:
        segment_texts = _ocr_segments(active_plan.crop_regions)
        deduped_segment_texts, duplicate_segment_count = _dedupe_neighboring_text_segments(segment_texts)
        filtered_segment_texts, noisy_segment_count = _filter_noisy_segments(deduped_segment_texts)

        cleaned_segment_texts: List[str] = []
        segment_truncation_blocked = False
        segment_truncation_count = 0
        for text in filtered_segment_texts:
            normalized_text = normalize_text(text)
            if not normalized_text:
                continue
            if len(normalized_text) > max_chars_per_segment:
                if hard_truncate_segment_text:
                    normalized_text = _truncate_text(normalized_text, max_chars_per_segment)
                    segment_truncation_count += 1
                else:
                    segment_truncation_blocked = True
            cleaned_segment_texts.append(normalized_text)

        raw_output = "\n".join(cleaned_segment_texts).strip()
        normalized_output = _postprocess_prediction_text(raw_output)
        total_chars = len(normalized_output)
        marker_count = _count_invoice_markers(normalized_output)
        return (
            segment_texts,
            filtered_segment_texts,
            duplicate_segment_count,
            noisy_segment_count,
            raw_output,
            normalized_output,
            segment_truncation_count,
            total_chars,
            marker_count,
            segment_truncation_blocked,
        )

    fallback_reason = None
    guardrail_marker_cap_applied = False
    guardrail_char_cap_applied = False
    try:
        for _ in range(2):
            (
                segment_texts,
                deduped_segment_texts,
                duplicate_segment_count,
                noisy_segment_count,
                raw_output,
                normalized_output,
                segment_truncation_count,
                total_chars,
                marker_count,
                segment_truncation_blocked,
            ) = _extract_output(plan)
            if (
                segment_truncation_blocked
                and not plan.used_full_page_fallback
                and plan.strategy != "full_page"
            ):
                fallback_reason = "segment_too_long"
                plan = _full_page_plan(page_image.size, fallback_reason)
                continue

            if marker_count > max_invoice_markers_per_page:
                normalized_output = _limit_invoice_sections(normalized_output, max_invoice_markers_per_page)
                raw_output = normalized_output
                marker_count = _count_invoice_markers(normalized_output)
                total_chars = len(normalized_output)
                guardrail_marker_cap_applied = True

            if total_chars > max_total_chars:
                normalized_output = _truncate_text(normalized_output, max_total_chars)
                raw_output = normalized_output
                total_chars = len(normalized_output)
                guardrail_char_cap_applied = True
            break
        else:  # pragma: no cover - defensive safeguard
            raise RuntimeError("OCR guardrails failed to converge")
    except Exception as exc:
        if plan.used_full_page_fallback:
            raise
        fallback_reason = f"crop_ocr_failed: {exc}"
        plan = _full_page_plan(page_image.size, fallback_reason)
        (
            segment_texts,
            deduped_segment_texts,
            duplicate_segment_count,
            noisy_segment_count,
            raw_output,
            normalized_output,
            segment_truncation_count,
            total_chars,
            marker_count,
            _segment_truncation_blocked,
        ) = _extract_output(plan)
        if total_chars > max_total_chars:
            normalized_output = _truncate_text(normalized_output, max_total_chars)
            raw_output = normalized_output
            guardrail_char_cap_applied = True

    if not normalize_text(raw_output) and not plan.used_full_page_fallback:
        fallback_reason = "empty_crop_output"
        plan = _full_page_plan(page_image.size, fallback_reason)
        (
            segment_texts,
            deduped_segment_texts,
            duplicate_segment_count,
            noisy_segment_count,
            raw_output,
            normalized_output,
            segment_truncation_count,
            total_chars,
            marker_count,
            _segment_truncation_blocked,
        ) = _extract_output(plan)
        if total_chars > max_total_chars:
            normalized_output = _truncate_text(normalized_output, max_total_chars)
            raw_output = normalized_output
            guardrail_char_cap_applied = True

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "prediction": normalized_output,
        "extracted_text": normalized_output,
        "normalized_output": normalized_output,
        "raw_output": raw_output,
        "latency_ms": round(latency_ms, 3),
        "requested_segmentation_mode": requested_mode,
        "segmentation_strategy": plan.strategy,
        "used_full_page_fallback": bool(plan.used_full_page_fallback),
        "fallback_reason": plan.fallback_reason,
        "crop_count": len(plan.crop_regions),
        "original_crop_count": int(getattr(plan, "original_crop_count", len(plan.crop_regions))),
        "deduplicated_crop_count": int(getattr(plan, "deduplicated_crop_count", len(plan.crop_regions))),
        "duplicate_crop_count": int(getattr(plan, "duplicate_crop_count", 0)),
        "segment_count": len(segment_texts),
        "deduplicated_segment_count": len(deduped_segment_texts),
        "duplicate_segment_count": int(duplicate_segment_count),
        "noisy_segment_count": int(noisy_segment_count),
        "segment_truncation_count": int(segment_truncation_count),
        "max_chars_per_segment": int(max_chars_per_segment),
        "max_total_chars": int(max_total_chars),
        "max_invoice_markers_per_page": int(max_invoice_markers_per_page),
        "hard_truncate_segment_text": bool(hard_truncate_segment_text),
        "guardrail_marker_cap_applied": bool(guardrail_marker_cap_applied),
        "guardrail_char_cap_applied": bool(guardrail_char_cap_applied),
        "invoice_marker_count": int(marker_count),
        "max_crops": int(max_crops),
        "crop_batch_size": int(batch_size),
    }






