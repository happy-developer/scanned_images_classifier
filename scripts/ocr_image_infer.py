from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR image->text inference on one image.")
    parser.add_argument("--artifacts-dir", type=str, default="notebooks/artifacts/doc_understanding_ocr_cpu")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--no-grayscale", action="store_true", help="Disable grayscale preprocessing")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=5)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--segmentation-mode", type=str, default="line_only", choices=("line_only", "line_block", "full_page"))
    parser.add_argument("--max-chars-per-segment", type=int, default=256)
    parser.add_argument("--max-total-chars", type=int, default=1200)
    parser.add_argument("--max-invoice-markers-per-page", type=int, default=2)
    parser.add_argument("--max-crops", type=int, default=16)
    parser.add_argument("--crop-batch-size", type=int, default=6)
    parser.add_argument(
        "--hard-truncate-segment-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Truncate overlong segment text before joining (use --no-hard-truncate-segment-text to disable)",
    )
    return parser.parse_args()


def _resolve_image(image_arg: str, data_root: Path) -> Path:
    candidate = Path(image_arg)
    if candidate.exists():
        return candidate.resolve()
    candidate2 = data_root / image_arg
    if candidate2.exists():
        return candidate2.resolve()
    raise FileNotFoundError(f"Image not found: {image_arg}")


def main() -> None:
    args = parse_args()
    from src.ocr_image_text.config import InferConfig
    from src.ocr_image_text.data import resolve_default_data_root
    from src.ocr_image_text.inference import load_predictor

    data_root = resolve_default_data_root(args.data_root)
    cfg = InferConfig(
        artifacts_dir=Path(args.artifacts_dir).resolve(),
        image_size=args.image_size,
        use_grayscale=not args.no_grayscale,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        temperature=args.temperature,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        segmentation_mode=args.segmentation_mode,
        max_chars_per_segment=args.max_chars_per_segment,
        max_total_chars=args.max_total_chars,
        max_invoice_markers_per_page=args.max_invoice_markers_per_page,
        hard_truncate_segment_text=args.hard_truncate_segment_text,
        max_crops=args.max_crops,
        crop_batch_size=args.crop_batch_size,
    )
    predictor = load_predictor(cfg)
    image_path = _resolve_image(args.image, data_root)
    result = predictor.predict(image_path)
    result["data_root"] = str(data_root)
    result["image_path"] = str(image_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

