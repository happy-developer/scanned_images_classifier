from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BATCH1_TEST_SPLIT_REL = "outputs/batch1_test_split.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate OCR model with supervised CER/WER by default on batch1 test split. "
            "Unlabeled mode is explicit and does not validate quality."
        )
    )
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument(
        "--use-batch1-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use outputs/batch1_test_split.csv with image subdir '.' when available (default: true).",
    )
    parser.add_argument("--artifacts-dir", type=str, default="notebooks/artifacts/doc_understanding_ocr_cpu")
    parser.add_argument("--eval-csv", type=str, default="")
    parser.add_argument("--image-subdir-eval", type=str, default="batch_2/batch_2/batch2_1")
    parser.add_argument(
        "--eval-image-subdirs",
        type=str,
        default="batch_2/batch_2/batch2_1,batch_2/batch_2/batch2_2,batch_2/batch_2/batch2_3",
        help="Comma-separated unlabeled validation image folders",
    )
    parser.add_argument(
        "--require-supervised-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a labeled eval CSV; use --allow-unlabeled-eval to bypass this guard.",
    )
    parser.add_argument(
        "--allow-unlabeled-eval",
        action="store_true",
        help="Allow unlabeled inference-only evaluation when no labeled CSV is available.",
    )
    parser.add_argument("--max-samples", type=int, default=0)
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
        help="Truncate overlong segment text before joining.",
    )
    parser.add_argument("--latency-target-ms", type=float, default=20000.0)
    parser.add_argument("--cer-target", type=float, default=0.70)
    parser.add_argument("--field-exact-match-min", type=float, default=0.0)
    parser.add_argument("--max-noisy-segment-ratio", type=float, default=0.60)
    parser.add_argument("--no-wer", action="store_true", help="Skip WER computation in supervised evaluation")
    return parser.parse_args()


def _build_infer_config(args: argparse.Namespace, artifacts_dir: Path):
    from src.ocr_image_text.config import InferConfig

    return InferConfig(
        artifacts_dir=artifacts_dir,
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
        "noisy_rejected_repeated_fields_sum": 0.0,
        "noisy_rejected_field_heavy_cap_sum": 0.0,
        "marker_cap_rejected_markers_sum": 0.0,
        "char_cap_trimmed_chars_sum": 0.0,
        "crop_count_sum": 0.0,
        "guardrail_marker_cap_applied_count": 0.0,
        "guardrail_char_cap_applied_count": 0.0,
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
    diag["noisy_rejected_repeated_fields_sum"] += _safe_int(output.get("noisy_rejected_repeated_fields_count"), 0)
    diag["noisy_rejected_field_heavy_cap_sum"] += _safe_int(output.get("noisy_rejected_field_heavy_cap_count"), 0)
    diag["marker_cap_rejected_markers_sum"] += _safe_int(output.get("marker_cap_rejected_markers_count"), 0)
    diag["char_cap_trimmed_chars_sum"] += _safe_int(output.get("char_cap_trimmed_chars"), 0)
    diag["crop_count_sum"] += _safe_int(output.get("crop_count"), 0)
    diag["guardrail_marker_cap_applied_count"] += 1.0 if bool(output.get("guardrail_marker_cap_applied", False)) else 0.0
    diag["guardrail_char_cap_applied_count"] += 1.0 if bool(output.get("guardrail_char_cap_applied", False)) else 0.0
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
        "avg_duplicate_segment_count": diag["duplicate_segment_count_sum"] / n,
        "avg_noisy_segment_count": diag["noisy_segment_count_sum"] / n,
        "avg_noisy_rejected_repeated_fields_count": diag["noisy_rejected_repeated_fields_sum"] / n,
        "avg_noisy_rejected_field_heavy_cap_count": diag["noisy_rejected_field_heavy_cap_sum"] / n,
        "avg_marker_cap_rejected_markers_count": diag["marker_cap_rejected_markers_sum"] / n,
        "avg_char_cap_trimmed_chars": diag["char_cap_trimmed_chars_sum"] / n,
        "guardrail_marker_cap_applied_rate": diag["guardrail_marker_cap_applied_count"] / n,
        "guardrail_char_cap_applied_rate": diag["guardrail_char_cap_applied_count"] / n,
        "avg_segmentation_latency_ms": diag["segmentation_latency_ms_sum"] / n,
        "avg_ocr_latency_ms": diag["ocr_latency_ms_sum"] / n,
        "avg_postprocess_latency_ms": diag["postprocess_latency_ms_sum"] / n,
    }


def main() -> None:
    args = parse_args()

    from src.ocr_image_text.data import load_images_from_subdirs, load_ocr_csv, resolve_default_data_root
    from src.ocr_image_text.evaluation import evaluate_records, summarize_predictions
    from src.ocr_image_text.inference import load_predictor

    data_root = resolve_default_data_root(args.data_root)
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    eval_csv_arg = args.eval_csv
    image_subdir_eval_arg = args.image_subdir_eval

    if args.use_batch1_split:
        split_test_path = (data_root / BATCH1_TEST_SPLIT_REL).resolve()
        if not split_test_path.exists():
            raise FileNotFoundError(
                "Batch1 split evaluation requested but split CSV file is missing: "
                + str(split_test_path)
                + ". Expected file: outputs/batch1_test_split.csv under --data-root."
            )
        eval_csv_arg = BATCH1_TEST_SPLIT_REL
        image_subdir_eval_arg = "."

    payload: dict = {
        "schema_version": "ocr_image_eval.v5",
        "decode": {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "temperature": args.temperature,
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
        "image_size": args.image_size,
        "use_grayscale": not args.no_grayscale,
        "latency_target_ms": float(args.latency_target_ms),
        "go_no_go_targets": {
            "cer_target": float(args.cer_target),
            "field_exact_match_min": float(args.field_exact_match_min),
            "max_noisy_segment_ratio": float(args.max_noisy_segment_ratio),
            "max_latency_ms": float(args.latency_target_ms),
        },
        "quality_validation": False,
    }
    effective_size: int | None = None

    can_fallback_to_unlabeled = bool(args.allow_unlabeled_eval) or not bool(args.require_supervised_eval)
    eval_csv_path = (data_root / eval_csv_arg) if eval_csv_arg else None
    eval_img_dir = data_root / image_subdir_eval_arg

    predictor = load_predictor(_build_infer_config(args, artifacts_dir))

    if eval_csv_arg and eval_csv_path and eval_csv_path.exists() and eval_img_dir.exists():
        records = load_ocr_csv(eval_csv_path, eval_img_dir)
        if args.max_samples > 0:
            records = records[: args.max_samples]

        preds: dict[str, str] = {}
        rows: list[dict[str, object]] = []
        total_latency = 0.0
        fallback_count = 0
        diag = _empty_diagnostics()

        for rec in records:
            output = predictor.predict(rec.image_path)
            _accumulate_diagnostics(diag, output)
            if effective_size is None:
                effective_size = int(output.get("effective_image_size", args.image_size))
            pred = str(output.get("prediction", ""))
            lat = float(output.get("latency_ms", 0.0))
            used_fallback = bool(output.get("used_full_page_fallback", False))
            preds[rec.img_name] = pred
            total_latency += lat
            fallback_count += int(used_fallback)
            rows.append(
                {
                    "img_name": rec.img_name,
                    "image_path": str(rec.image_path),
                    "prediction": pred,
                    "latency_ms": lat,
                    "used_full_page_fallback": used_fallback,
                    "fallback_reason": output.get("fallback_reason"),
                    "segment_count": _safe_int(output.get("segment_count"), 0),
                    "segments_kept_count": _safe_int(output.get("segments_kept_count"), 0),
                    "noisy_segment_count": _safe_int(output.get("noisy_segment_count"), 0),
                    "duplicate_segment_count": _safe_int(output.get("duplicate_segment_count"), 0),
                    "marker_cap_rejected_markers_count": _safe_int(output.get("marker_cap_rejected_markers_count"), 0),
                    "char_cap_trimmed_chars": _safe_int(output.get("char_cap_trimmed_chars"), 0),
                    "segmentation_latency_ms": _safe_float(output.get("segmentation_latency_ms"), 0.0),
                    "ocr_latency_ms": _safe_float(output.get("ocr_latency_ms"), 0.0),
                    "postprocess_latency_ms": _safe_float(output.get("postprocess_latency_ms"), 0.0),
                }
            )

        payload["mode"] = "supervised_batch1_test_split_validation" if args.use_batch1_split else "supervised_validation"
        payload["quality_validation"] = True
        metrics = evaluate_records(records, preds, compute_wer=not args.no_wer)
        n = max(len(records), 1)
        avg_latency_ms = total_latency / n
        metrics["avg_latency_ms"] = avg_latency_ms
        metrics["full_page_fallback_rate"] = fallback_count / n
        metrics["latency_target_ms"] = float(args.latency_target_ms)
        metrics["latency_target_met"] = bool(avg_latency_ms <= float(args.latency_target_ms))
        metrics.update(_finalize_diagnostics(diag))
        metrics["cer_target"] = float(args.cer_target)
        metrics["cer_target_met"] = bool(_safe_float(metrics.get("cer"), 1.0) < float(args.cer_target))
        metrics["field_exact_match_min"] = float(args.field_exact_match_min)
        metrics["field_exact_match_target_met"] = bool(
            _safe_float(metrics.get("field_exact_match_overall"), 0.0) > float(args.field_exact_match_min)
        )
        metrics["max_noisy_segment_ratio"] = float(args.max_noisy_segment_ratio)
        metrics["noisy_segment_ratio_target_met"] = bool(
            _safe_float(metrics.get("noisy_segment_ratio"), 1.0) <= float(args.max_noisy_segment_ratio)
        )
        metrics["go_no_go_ready_for_epoch8"] = bool(
            metrics.get("cer_target_met")
            and metrics.get("field_exact_match_target_met")
            and metrics.get("latency_target_met")
            and metrics.get("noisy_segment_ratio_target_met")
        )
        payload["metrics"] = metrics
        payload["metrics_kind"] = "supervised_quality"
        payload["num_samples"] = len(records)
        payload["sample_predictions"] = rows[:5]
    else:
        if not can_fallback_to_unlabeled:
            missing = []
            if not eval_csv_arg:
                missing.append("--eval-csv not provided")
            elif not eval_csv_path or not eval_csv_path.exists():
                missing.append(f"eval CSV not found: {eval_csv_path}")
            if eval_csv_arg and not eval_img_dir.exists():
                missing.append(f"image directory not found: {eval_img_dir}")
            raise FileNotFoundError(
                "Supervised evaluation is required but unavailable. "
                + "; ".join(missing)
                + ". Pass --allow-unlabeled-eval only if you intentionally want prediction-only output."
            )

        subdirs = tuple(s.strip() for s in args.eval_image_subdirs.split(",") if s.strip())
        image_paths = load_images_from_subdirs(data_root, subdirs)
        if args.max_samples > 0:
            image_paths = image_paths[: args.max_samples]

        rows = []
        lat_sum = 0.0
        fallback_count = 0
        diag = _empty_diagnostics()
        for p in image_paths:
            out = predictor.predict(p)
            _accumulate_diagnostics(diag, out)
            if effective_size is None:
                effective_size = int(out.get("effective_image_size", args.image_size))
            pred = str(out.get("prediction", ""))
            lat = float(out.get("latency_ms", 0.0))
            used_fallback = bool(out.get("used_full_page_fallback", False))
            rows.append(
                {
                    "image_path": str(p),
                    "prediction": pred,
                    "latency_ms": lat,
                    "used_full_page_fallback": used_fallback,
                    "fallback_reason": out.get("fallback_reason"),
                    "segment_count": _safe_int(out.get("segment_count"), 0),
                    "segments_kept_count": _safe_int(out.get("segments_kept_count"), 0),
                    "noisy_segment_count": _safe_int(out.get("noisy_segment_count"), 0),
                    "duplicate_segment_count": _safe_int(out.get("duplicate_segment_count"), 0),
                    "marker_cap_rejected_markers_count": _safe_int(out.get("marker_cap_rejected_markers_count"), 0),
                    "char_cap_trimmed_chars": _safe_int(out.get("char_cap_trimmed_chars"), 0),
                    "segmentation_latency_ms": _safe_float(out.get("segmentation_latency_ms"), 0.0),
                    "ocr_latency_ms": _safe_float(out.get("ocr_latency_ms"), 0.0),
                    "postprocess_latency_ms": _safe_float(out.get("postprocess_latency_ms"), 0.0),
                }
            )
            lat_sum += lat
            fallback_count += int(used_fallback)

        pred_path = artifacts_dir / "batch2_validation_predictions.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        n = len(rows)
        summary = summarize_predictions(r["prediction"] for r in rows)
        avg_latency_ms = (lat_sum / n) if n else 0.0
        fallback_rate = (fallback_count / n) if n else 0.0
        summary["avg_latency_ms"] = avg_latency_ms
        summary["full_page_fallback_rate"] = fallback_rate
        summary["latency_target_ms"] = float(args.latency_target_ms)
        summary["latency_target_met"] = bool(avg_latency_ms <= float(args.latency_target_ms))
        summary.update(_finalize_diagnostics(diag))
        summary["cer_target"] = float(args.cer_target)
        summary["field_exact_match_min"] = float(args.field_exact_match_min)
        summary["max_noisy_segment_ratio"] = float(args.max_noisy_segment_ratio)
        summary["go_no_go_ready_for_epoch8"] = False

        print(
            "WARNING: unlabeled evaluation mode is inference-only; it does not measure CER/WER quality.",
            file=sys.stderr,
        )
        payload["mode"] = "unlabeled_inference_only"
        payload["quality_validation"] = False
        payload["warning"] = {
            "severity": "high",
            "code": "UNLABELED_EVAL_MODE",
            "message": (
                "This run has no supervised labels. The output contains prediction summaries and latency only, "
                "so it must not be interpreted as quality validation."
            ),
        }
        payload["metrics_kind"] = "prediction_summary"
        payload["prediction_summary"] = summary
        payload["metrics"] = summary
        payload["num_samples"] = n
        payload["predictions_path"] = str(pred_path)
        payload["sample_predictions"] = rows[:5]

    payload["effective_image_size"] = int(effective_size or args.image_size)

    out_path = artifacts_dir / "eval_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"metrics_path": str(out_path), "metrics": payload}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()




