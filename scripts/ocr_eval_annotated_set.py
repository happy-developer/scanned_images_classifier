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
    parser = argparse.ArgumentParser(
        description="Evaluate OCR CER/WER on a manually annotated CSV built from batch_2/tests_AB."
    )
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--annotations-csv", type=str, default="outputs/real_val_to_annotate.csv")
    parser.add_argument("--artifacts-dir", type=str, default="notebooks/artifacts/doc_understanding_ocr_cpu")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--no-grayscale", action="store_true", help="Disable grayscale preprocessing")
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
    parser.add_argument(
        "--hard-truncate-segment-text",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-wer", action="store_true", help="Skip WER computation")
    return parser.parse_args()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _empty_diagnostics() -> dict[str, float]:
    return {
        "num_predictions": 0.0,
        "segment_count_sum": 0.0,
        "segments_after_dedup_sum": 0.0,
        "segments_kept_sum": 0.0,
        "duplicate_segment_count_sum": 0.0,
        "noisy_segment_count_sum": 0.0,
        "marker_cap_rejected_markers_sum": 0.0,
        "char_cap_trimmed_chars_sum": 0.0,
        "crop_count_sum": 0.0,
        "segmentation_latency_ms_sum": 0.0,
        "ocr_latency_ms_sum": 0.0,
        "postprocess_latency_ms_sum": 0.0,
    }


def _accumulate_diagnostics(diag: dict[str, float], output: dict[str, object]) -> None:
    diag["num_predictions"] += 1.0
    diag["segment_count_sum"] += _safe_int(output.get("segment_count"), 0)
    diag["segments_after_dedup_sum"] += _safe_int(output.get("deduplicated_segment_count"), 0)
    diag["segments_kept_sum"] += _safe_int(output.get("segments_kept_count"), 0)
    diag["duplicate_segment_count_sum"] += _safe_int(output.get("duplicate_segment_count"), 0)
    diag["noisy_segment_count_sum"] += _safe_int(output.get("noisy_segment_count"), 0)
    diag["marker_cap_rejected_markers_sum"] += _safe_int(output.get("marker_cap_rejected_markers_count"), 0)
    diag["char_cap_trimmed_chars_sum"] += _safe_int(output.get("char_cap_trimmed_chars"), 0)
    diag["crop_count_sum"] += _safe_int(output.get("crop_count"), 0)
    diag["segmentation_latency_ms_sum"] += _safe_float(output.get("segmentation_latency_ms"), 0.0)
    diag["ocr_latency_ms_sum"] += _safe_float(output.get("ocr_latency_ms"), 0.0)
    diag["postprocess_latency_ms_sum"] += _safe_float(output.get("postprocess_latency_ms"), 0.0)


def _finalize_diagnostics(diag: dict[str, float]) -> dict[str, float]:
    n = max(1.0, diag["num_predictions"])
    segment_total = max(1.0, diag["segment_count_sum"])
    return {
        "avg_crop_count": diag["crop_count_sum"] / n,
        "avg_segment_count": diag["segment_count_sum"] / n,
        "avg_segments_after_dedup_count": diag["segments_after_dedup_sum"] / n,
        "avg_segments_kept_count": diag["segments_kept_sum"] / n,
        "segments_kept_ratio": diag["segments_kept_sum"] / segment_total,
        "duplicate_segment_ratio": diag["duplicate_segment_count_sum"] / segment_total,
        "noisy_segment_ratio": diag["noisy_segment_count_sum"] / segment_total,
        "avg_marker_cap_rejected_markers_count": diag["marker_cap_rejected_markers_sum"] / n,
        "avg_char_cap_trimmed_chars": diag["char_cap_trimmed_chars_sum"] / n,
        "avg_segmentation_latency_ms": diag["segmentation_latency_ms_sum"] / n,
        "avg_ocr_latency_ms": diag["ocr_latency_ms_sum"] / n,
        "avg_postprocess_latency_ms": diag["postprocess_latency_ms_sum"] / n,
    }


def _resolve_image_path(data_root: Path, row: dict[str, str], csv_path: Path) -> Path | None:
    source_path = str(row.get("source_path", "") or "").strip()
    file_name = str(row.get("File Name", "") or "").strip()

    candidates: list[Path] = []
    if source_path:
        src = Path(source_path)
        if src.is_absolute():
            candidates.append(src)
        else:
            candidates.append(data_root / src)
            candidates.append((csv_path.parent / src).resolve())

    if file_name:
        candidates.append((data_root / file_name).resolve())
        candidates.append((csv_path.parent / file_name).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def main() -> None:
    args = parse_args()

    from src.ocr_image_text.config import InferConfig
    from src.ocr_image_text.data import OCRRecord, resolve_default_data_root
    from src.ocr_image_text.evaluation import evaluate_records
    from src.ocr_image_text.inference import load_predictor

    data_root = resolve_default_data_root(args.data_root)
    annotations_csv = Path(args.annotations_csv).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not annotations_csv.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {annotations_csv}")

    records: list[OCRRecord] = []
    skipped_missing_label = 0
    skipped_missing_image = 0

    with annotations_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"File Name", "OCRed Text"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"CSV missing required columns {sorted(missing)}: {annotations_csv}")

        for row in reader:
            file_name = str(row.get("File Name", "") or "").strip()
            text = str(row.get("OCRed Text", "") or "").strip()
            if not file_name:
                continue
            if not text:
                skipped_missing_label += 1
                continue
            image_path = _resolve_image_path(data_root, row, annotations_csv)
            if image_path is None:
                skipped_missing_image += 1
                continue
            records.append(OCRRecord(img_name=file_name, image_path=image_path, ocr_text=text))

    if args.max_samples and args.max_samples > 0:
        records = records[: args.max_samples]

    if not records:
        raise ValueError(
            "No labeled records available for CER/WER. Fill column 'OCRed Text' in annotations CSV first."
        )

    predictor = load_predictor(
        InferConfig(
            artifacts_dir=artifacts_dir,
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

    predictions: dict[str, str] = {}
    sample_rows: list[dict[str, str]] = []
    total_latency_ms = 0.0
    diag = _empty_diagnostics()
    for rec in records:
        out = predictor.predict(
            rec.image_path,
            segmentation_mode=args.segmentation_mode,
            max_chars_per_segment=args.max_chars_per_segment,
            max_total_chars=args.max_total_chars,
            max_invoice_markers_per_page=args.max_invoice_markers_per_page,
            hard_truncate_segment_text=args.hard_truncate_segment_text,
            max_crops=args.max_crops,
            crop_batch_size=args.crop_batch_size,
        )
        _accumulate_diagnostics(diag, out)
        pred = str(out.get("prediction", ""))
        predictions[rec.img_name] = pred
        total_latency_ms += float(out.get("latency_ms", 0.0))
        if len(sample_rows) < 5:
            sample_rows.append(
                {
                    "image": rec.img_name,
                    "reference": rec.ocr_text,
                    "prediction": pred,
                }
            )

    metrics = evaluate_records(records, predictions, compute_wer=not args.no_wer)
    metrics.update(_finalize_diagnostics(diag))

    payload = {
        "schema_version": "ocr_eval_annotated_set.v2",
        "mode": "supervised_validation",
        "quality_validation": True,
        "annotations_csv": str(annotations_csv),
        "data_root": str(data_root),
        "num_samples": len(records),
        "skipped_missing_label": skipped_missing_label,
        "skipped_missing_image": skipped_missing_image,
        "decode": {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "length_penalty": args.length_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "repetition_penalty": args.repetition_penalty,
            "segmentation_mode": args.segmentation_mode,
            "max_chars_per_segment": args.max_chars_per_segment,
            "max_total_chars": args.max_total_chars,
            "max_invoice_markers_per_page": args.max_invoice_markers_per_page,
            "hard_truncate_segment_text": args.hard_truncate_segment_text,
            "max_crops": args.max_crops,
            "crop_batch_size": args.crop_batch_size,
        },
        "metrics": metrics,
        "avg_latency_ms": total_latency_ms / len(records) if records else 0.0,
        "sample_predictions": sample_rows,
    }

    output_path = artifacts_dir / "eval_metrics_real_val.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"metrics_path": str(output_path), "num_samples": len(records), "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
