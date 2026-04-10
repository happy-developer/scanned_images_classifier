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
    parser.add_argument("--max-invoice-markers-per-page", type=int, default=1)
    parser.add_argument("--max-crops", type=int, default=28)
    parser.add_argument("--crop-batch-size", type=int, default=6)
    parser.add_argument(
        "--hard-truncate-segment-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Truncate overlong segment text before joining.",
    )
    parser.add_argument("--latency-target-ms", type=float, default=20000.0)
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
        "schema_version": "ocr_image_eval.v4",
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

        for rec in records:
            output = predictor.predict(rec.image_path)
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
        for p in image_paths:
            out = predictor.predict(p)
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
