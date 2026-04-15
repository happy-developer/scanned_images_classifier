from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefill OCRed Text in annotation CSV with current OCR model.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--annotations-csv", type=str, default="outputs/real_val_to_annotate.csv")
    parser.add_argument("--output-csv", type=str, default="outputs/real_val_prefilled_model.csv")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--artifacts-dir", type=str, default="notebooks/artifacts/doc_understanding_ocr_cpu")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--no-grayscale", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=5)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--segmentation-mode", type=str, default="line_only", choices=("line_only", "line_block", "full_page"))
    parser.add_argument("--max-chars-per-segment", type=int, default=256)
    parser.add_argument("--max-total-chars", type=int, default=1200)
    parser.add_argument("--max-invoice-markers-per-page", type=int, default=2)
    parser.add_argument("--max-crops", type=int, default=16)
    parser.add_argument("--crop-batch-size", type=int, default=6)
    parser.add_argument("--hard-truncate-segment-text", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _resolve_image_path(data_root: Path, row: dict[str, str], csv_path: Path) -> Path | None:
    source_path = str(row.get("source_path", "") or "").strip()
    file_name = str(row.get("File Name", "") or "").strip()

    candidates: list[Path] = []
    if source_path:
        src = Path(source_path)
        if src.is_absolute():
            candidates.append(src)
        else:
            candidates.append((data_root / src).resolve())
            candidates.append((csv_path.parent / src).resolve())
    if file_name:
        candidates.append((data_root / file_name).resolve())
        candidates.append((csv_path.parent / file_name).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()

    from src.ocr_image_text.config import InferConfig
    from src.ocr_image_text.inference import load_predictor

    data_root = Path(args.data_root).resolve()
    annotations_csv = Path(args.annotations_csv).resolve()
    output_csv = Path(args.output_csv).resolve()

    with annotations_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"Empty annotations CSV: {annotations_csv}")
        fieldnames = list(rows[0].keys())

    predictor = load_predictor(
        InferConfig(
            artifacts_dir=Path(args.artifacts_dir).resolve(),
            image_size=args.image_size,
            use_grayscale=not args.no_grayscale,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
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
    )

    filled = 0
    skipped_existing = 0
    missing_image = 0
    processed = 0

    for row in rows:
        if args.max_rows and processed >= args.max_rows:
            break
        processed += 1

        text = str(row.get("OCRed Text", "") or "").strip()
        if text:
            skipped_existing += 1
            continue

        image_path = _resolve_image_path(data_root, row, annotations_csv)
        if image_path is None:
            missing_image += 1
            continue

        out = predictor.predict(
            image_path,
            segmentation_mode=args.segmentation_mode,
            max_chars_per_segment=args.max_chars_per_segment,
            max_total_chars=args.max_total_chars,
            max_invoice_markers_per_page=args.max_invoice_markers_per_page,
            hard_truncate_segment_text=args.hard_truncate_segment_text,
            max_crops=args.max_crops,
            crop_batch_size=args.crop_batch_size,
        )
        row["OCRed Text"] = str(out.get("prediction", ""))
        filled += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "output_csv": str(output_csv),
                "num_rows": len(rows),
                "processed_rows": processed,
                "filled_rows": filled,
                "skipped_existing": skipped_existing,
                "missing_image": missing_image,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

