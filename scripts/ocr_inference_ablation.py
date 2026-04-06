from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _collect_image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    images: list[Path] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            images.append(path.resolve())
    return images


def _resolve_tests_ab_dir(data_root: Path | None, tests_dir: str) -> Path | None:
    candidates: list[Path] = []
    raw = Path(tests_dir)
    if raw.is_absolute():
        candidates.append(raw)
    else:
        if data_root is not None:
            candidates.append(data_root / raw)
        candidates.append(PROJECT_ROOT / raw)
    if data_root is not None:
        candidates.append(data_root / "tests_AB")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    search_roots = [p for p in [data_root, PROJECT_ROOT] if p is not None and p.exists()]
    for root in search_roots:
        for found in root.rglob("tests_AB"):
            if found.is_dir():
                return found.resolve()
    return None


def _preprocess_image(image_path: Path, use_grayscale: bool):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    if use_grayscale:
        image = image.convert("L").convert("RGB")
    return image


def _clean_repeated_lines(text: str) -> str:
    lines: list[str] = []
    for line in _normalize_text(text).split("\n"):
        if not line:
            continue
        if lines and lines[-1].casefold() == line.casefold():
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _apply_guard_rails(text: str, guard_rail: str) -> str:
    normalized = _clean_repeated_lines(text)
    if guard_rail == "off":
        return normalized
    if guard_rail == "balanced":
        return normalized
    if guard_rail == "strict":
        normalized = re.sub(r"(\b\w+\b(?:\s+\b\w+\b){2,})(?:\s+\1)+", r"\1", normalized, flags=re.I)
        return normalized[:4096].strip()
    return normalized


def _has_repetition_proxy(text: str) -> bool:
    tokens = re.findall(r"\w+", text.casefold())
    if len(tokens) < 8:
        return False
    trigrams = [tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    return len(set(trigrams)) < len(trigrams)


def _count_invoice_markers(text: str) -> int:
    return len(re.findall(r"\binvoice\b", text, flags=re.I))


def _generation_params(guard_rail: str) -> tuple[int, float, bool]:
    if guard_rail == "off":
        return 0, 1.0, False
    if guard_rail == "strict":
        return 6, 1.25, True
    return 4, 1.15, True


@dataclass(frozen=True)
class AblationConfig:
    max_new_tokens: int
    num_beams: int
    segmentation_mode: str
    guard_rail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "segmentation_mode": self.segmentation_mode,
            "guard_rail": self.guard_rail,
        }


def _run_model_text(
    predictor: Any,
    image_path: Path,
    *,
    segmentation_mode: str,
    max_new_tokens: int,
    num_beams: int,
    guard_rail: str,
) -> tuple[str, dict[str, Any]]:
    from src.ocr_image_text.page_ocr import (
        _build_generation_kwargs,
        _dedupe_neighboring_text_segments,
        _full_page_plan,
        _run_model_on_images,
        segment_page,
    )

    image = _preprocess_image(image_path, use_grayscale=guard_rail != "off")
    model = predictor.model
    processor = predictor.processor
    no_repeat_ngram_size, repetition_penalty, dedupe_segments = _generation_params(guard_rail)

    generation_kwargs = _build_generation_kwargs(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        temperature=0.0,
        length_penalty=1.0,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        tokenizer=getattr(processor, "tokenizer", None),
    )

    metrics: dict[str, Any] = {
        "segmentation_mode": segmentation_mode,
        "guard_rail": guard_rail,
        "dedupe_segments": dedupe_segments,
    }

    if segmentation_mode == "full_page":
        output_texts = _run_model_on_images(model, processor, [image], generation_kwargs=generation_kwargs)
        raw_output = "\n".join(output_texts).strip()
        metrics["crop_count"] = 1
        metrics["used_full_page_fallback"] = True
        metrics["segmentation_strategy"] = "full_page"
    else:
        plan = segment_page(image, segmentation_mode=segmentation_mode)
        regions = plan.crop_regions
        output_texts = []
        if regions:
            batch_size = 4
            for start in range(0, len(regions), batch_size):
                batch = regions[start : start + batch_size]
                batch_images = [image.crop(region.box) for region in batch]
                output_texts.extend(
                    _run_model_on_images(model, processor, batch_images, generation_kwargs=generation_kwargs)
                )
        if dedupe_segments:
            output_texts, duplicate_count = _dedupe_neighboring_text_segments(output_texts)
        else:
            duplicate_count = 0
        raw_output = "\n".join(output_texts).strip()
        metrics["crop_count"] = len(regions)
        metrics["duplicate_segment_count"] = duplicate_count
        metrics["used_full_page_fallback"] = bool(plan.used_full_page_fallback)
        metrics["segmentation_strategy"] = plan.strategy
        metrics["fallback_reason"] = plan.fallback_reason
        if segmentation_mode == "auto" and not raw_output and not plan.used_full_page_fallback:
            fallback_plan = _full_page_plan(image.size, "empty_crop_output")
            output_texts = _run_model_on_images(
                model,
                processor,
                [image],
                generation_kwargs=generation_kwargs,
            )
            raw_output = "\n".join(output_texts).strip()
            metrics["used_full_page_fallback"] = True
            metrics["segmentation_strategy"] = fallback_plan.strategy
            metrics["fallback_reason"] = fallback_plan.fallback_reason

    normalized = _apply_guard_rails(raw_output, guard_rail)
    return normalized, metrics


def _parse_config_list(value: str, cast: Any) -> list[Any]:
    return [cast(item) for item in _split_csv(value)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short OCR inference ablation grid on tests_AB with current model artifacts and safe fallbacks."
    )
    parser.add_argument("--data-root", type=str, default="", help="Dataset root containing tests_AB. Falls back to Kaggle cache resolution.")
    parser.add_argument("--tests-dir", type=str, default="tests_AB", help="Relative or absolute path to the tests_AB folder.")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/doc_understanding_ocr_cpu",
        help="OCR model artifacts directory used by the current model.",
    )
    parser.add_argument(
        "--max-new-tokens-values",
        type=str,
        default="64,96",
        help="Comma-separated values for max_new_tokens.",
    )
    parser.add_argument(
        "--num-beams-values",
        type=str,
        default="1,4",
        help="Comma-separated values for num_beams.",
    )
    parser.add_argument(
        "--segmentation-modes",
        type=str,
        default="line_only,full_page",
        help="Comma-separated segmentation modes: auto, line_only, line_block, full_page.",
    )
    parser.add_argument(
        "--guard-rails",
        type=str,
        default="balanced,strict",
        help="Comma-separated guard rail presets: off, balanced, strict.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Optional maximum number of images to evaluate. Use 0 for all images.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/ocr_inference_ablation.json",
        help="Where to write the ablation JSON report.",
    )
    parser.add_argument("--image-size", type=int, default=768, help="Resize target used when loading the current OCR model.")
    return parser.parse_args()


def _evaluate_config(
    predictor: Any,
    image_paths: list[Path],
    config: AblationConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    chars: list[int] = []
    latencies: list[float] = []
    repetition_flags: list[float] = []
    invoice_marker_total = 0
    failures = 0

    for image_path in image_paths:
        started = time.perf_counter()
        try:
            prediction, metrics = _run_model_text(
                predictor,
                image_path,
                segmentation_mode=config.segmentation_mode,
                max_new_tokens=config.max_new_tokens,
                num_beams=config.num_beams,
                guard_rail=config.guard_rail,
            )
            error = None
            available = True
        except Exception as exc:  # pragma: no cover - runtime safety
            prediction = ""
            metrics = {
                "segmentation_mode": config.segmentation_mode,
                "guard_rail": config.guard_rail,
                "error": f"{type(exc).__name__}: {exc}",
            }
            error = metrics["error"]
            available = False
            failures += 1

        latency_ms = (time.perf_counter() - started) * 1000.0
        normalized = _normalize_text(prediction)
        if available:
            chars.append(len(normalized))
            latencies.append(latency_ms)
            repetition_flags.append(1.0 if _has_repetition_proxy(normalized) else 0.0)
            invoice_marker_total += _count_invoice_markers(normalized)

        rows.append(
            {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "prediction": normalized,
                "latency_ms": round(latency_ms, 3) if available else None,
                "availability": available,
                "error": error,
                "metrics": metrics,
            }
        )

    avg_chars = round(statistics.mean(chars), 2) if chars else None
    avg_latency_ms = round(statistics.mean(latencies), 3) if latencies else None
    repetition_rate = round(statistics.mean(repetition_flags), 4) if repetition_flags else None
    return {
        "config": config.as_dict(),
        "num_images": len(image_paths),
        "num_failures": failures,
        "avg_chars": avg_chars,
        "avg_latency_ms": avg_latency_ms,
        "repetition_rate": repetition_rate,
        "invoice_marker_count": invoice_marker_total,
        "rows": rows,
    }


def _rank_configs(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _sort_key(item: dict[str, Any]) -> tuple[float, float, float, float, int]:
        avg_chars_raw = item.get("avg_chars")
        repetition_raw = item.get("repetition_rate")
        latency_raw = item.get("avg_latency_ms")

        avg_chars = float(avg_chars_raw) if avg_chars_raw is not None else 1e18
        repetition_rate = float(repetition_raw) if repetition_raw is not None else 1e18
        avg_latency = float(latency_raw) if latency_raw is not None else 1e18
        invoice_markers = int(item.get("invoice_marker_count") or 0)
        marker_excess = max(0, invoice_markers - 1)
        empty_penalty = 1.0 if avg_chars < 50 else 0.0
        return (empty_penalty, avg_chars, repetition_rate, avg_latency, marker_excess)

    ranked = sorted(results, key=_sort_key)
    for position, item in enumerate(ranked, start=1):
        item["rank"] = position
        item["score_tuple"] = [
            item.get("avg_chars"),
            item.get("repetition_rate"),
            item.get("avg_latency_ms"),
            item.get("invoice_marker_count"),
        ]
    return ranked


def main() -> None:
    args = parse_args()

    try:
        from src.data_access.kagglehub_resolver import resolve_kaggle_dataset_root
    except Exception as exc:  # pragma: no cover - repository import failure
        payload = {
            "schema_version": "ocr_inference_ablation.v1",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    data_root: Path | None = None
    try:
        data_root = resolve_kaggle_dataset_root(args.data_root or None)
    except Exception:
        if args.data_root:
            candidate = Path(args.data_root).expanduser().resolve()
            if candidate.exists():
                data_root = candidate

    tests_ab_dir = _resolve_tests_ab_dir(data_root, args.tests_dir)
    image_paths = _collect_image_files(tests_ab_dir) if tests_ab_dir is not None else []
    if args.max_samples and args.max_samples > 0:
        image_paths = image_paths[: args.max_samples]

    if not image_paths:
        payload = {
            "schema_version": "ocr_inference_ablation.v1",
            "status": "error",
            "dataset_root": str(data_root) if data_root is not None else None,
            "tests_ab_dir": str(tests_ab_dir) if tests_ab_dir is not None else None,
            "error": "No images found in tests_AB.",
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    try:
        from src.ocr_image_text.config import InferConfig
        from src.ocr_image_text.inference import load_predictor
    except Exception as exc:  # pragma: no cover - repository import failure
        payload = {
            "schema_version": "ocr_inference_ablation.v1",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    try:
        predictor = load_predictor(
            InferConfig(
                artifacts_dir=artifacts_dir,
                image_size=args.image_size,
                use_grayscale=True,
                max_new_tokens=96,
                num_beams=4,
                temperature=0.0,
                length_penalty=1.0,
                no_repeat_ngram_size=6,
                repetition_penalty=1.2,
            )
        )
    except Exception as exc:
        payload = {
            "schema_version": "ocr_inference_ablation.v1",
            "status": "error",
            "dataset_root": str(data_root) if data_root is not None else None,
            "tests_ab_dir": str(tests_ab_dir) if tests_ab_dir is not None else None,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    max_new_tokens_values = _parse_config_list(args.max_new_tokens_values, int)
    num_beams_values = _parse_config_list(args.num_beams_values, int)
    segmentation_modes = _split_csv(args.segmentation_modes)
    guard_rails = _split_csv(args.guard_rails)

    configs = [
        AblationConfig(max_new_tokens=mnt, num_beams=nb, segmentation_mode=seg, guard_rail=guard)
        for mnt, nb, seg, guard in product(max_new_tokens_values, num_beams_values, segmentation_modes, guard_rails)
    ]

    results = [_evaluate_config(predictor, image_paths, config) for config in configs]
    ranked = _rank_configs(results)

    payload = {
        "schema_version": "ocr_inference_ablation.v1",
        "dataset_root": str(data_root) if data_root is not None else None,
        "tests_ab_dir": str(tests_ab_dir) if tests_ab_dir is not None else None,
        "artifacts_dir": str(artifacts_dir),
        "num_images": len(image_paths),
        "grid": {
            "max_new_tokens_values": max_new_tokens_values,
            "num_beams_values": num_beams_values,
            "segmentation_modes": segmentation_modes,
            "guard_rails": guard_rails,
        },
        "results": ranked,
        "top_configs": ranked[: min(5, len(ranked))],
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "num_images": len(image_paths),
                "top_configs": [
                    {
                        "rank": item["rank"],
                        "config": item["config"],
                        "avg_chars": item["avg_chars"],
                        "repetition_rate": item["repetition_rate"],
                        "avg_latency_ms": item["avg_latency_ms"],
                        "invoice_marker_count": item["invoice_marker_count"],
                    }
                    for item in payload["top_configs"]
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()



