from __future__ import annotations

import time
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


def _full_page_plan(page_size: Tuple[int, int], reason: str) -> SegmentationPlan:
    width, height = page_size
    return SegmentationPlan(
        crop_regions=[CropRegion(box=(0, 0, width, height), label="full_page")],
        used_full_page_fallback=True,
        strategy="full_page_fallback",
        fallback_reason=reason,
    )


def segment_page(
    image: Image.Image,
    *,
    max_regions: int = 64,
) -> SegmentationPlan:
    rgb = image.convert("RGB")
    width, height = rgb.size
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
    row_threshold = max(2, int(width * 0.004))
    row_spans = _merge_spans(_find_spans(row_scores, row_threshold), max_gap=1)
    if not row_spans:
        return _full_page_plan((width, height), "no_text_rows_detected")

    crop_regions: List[CropRegion] = []
    area_floor = max(256, int(width * height * 0.0005))
    pad_x = max(6, width // 80)
    pad_y = max(4, height // 100)

    for y0, y1 in row_spans:
        band = mask[y0 : y1 + 1, :]
        band_height = max(1, y1 - y0 + 1)
        col_scores = band.sum(axis=0)
        col_threshold = max(2, int(band_height * 0.01))
        col_spans = _merge_spans(_find_spans(col_scores, col_threshold), max_gap=max(3, width // 100))
        if not col_spans:
            col_spans = [(0, width - 1)]

        for x0, x1 in col_spans:
            box = _expand_box(x0, y0, x1 + 1, y1 + 1, width, height, pad_x=pad_x, pad_y=pad_y)
            if _box_area(box) < area_floor:
                continue
            crop_regions.append(CropRegion(box=box, label="block" if len(col_spans) > 1 else "line"))
            if len(crop_regions) >= max_regions:
                break
        if len(crop_regions) >= max_regions:
            break

    if not crop_regions:
        return _full_page_plan((width, height), "no_usable_crops")

    total_crop_area = sum(_box_area(region.box) for region in crop_regions)
    page_area = width * height
    if total_crop_area <= 0 or total_crop_area / max(1, page_area) < 0.01:
        return _full_page_plan((width, height), "crop_coverage_too_small")

    return SegmentationPlan(
        crop_regions=crop_regions,
        used_full_page_fallback=False,
        strategy="line_block_crops",
        fallback_reason=None,
    )


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
    max_new_tokens: int,
    num_beams: int,
    temperature: float,
    length_penalty: float,
    no_repeat_ngram_size: int,
    repetition_penalty: float,
    batch_size: int = 4,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    page_image = image.convert("RGB")
    plan = segment_page(page_image)

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

    try:
        segment_texts = _ocr_segments(plan.crop_regions)
        raw_output = "\n".join(text for text in segment_texts if text).strip()
    except Exception as exc:
        if plan.used_full_page_fallback:
            raise
        fallback_plan = _full_page_plan(page_image.size, f"crop_ocr_failed: {exc}")
        segment_texts = _ocr_segments(fallback_plan.crop_regions)
        raw_output = "\n".join(text for text in segment_texts if text).strip()
        plan = fallback_plan
    else:
        if not normalize_text(raw_output) and not plan.used_full_page_fallback:
            fallback_plan = _full_page_plan(page_image.size, "empty_crop_output")
            segment_texts = _ocr_segments(fallback_plan.crop_regions)
            raw_output = "\n".join(text for text in segment_texts if text).strip()
            plan = fallback_plan

    normalized_output = normalize_text(raw_output)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "prediction": normalized_output,
        "extracted_text": normalized_output,
        "normalized_output": normalized_output,
        "raw_output": raw_output,
        "latency_ms": round(latency_ms, 3),
        "segmentation_strategy": plan.strategy,
        "used_full_page_fallback": bool(plan.used_full_page_fallback),
        "fallback_reason": plan.fallback_reason,
        "crop_count": len(plan.crop_regions),
    }






