from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR model (supervised if CSV exists, else unlabeled batch_2 validation).")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--artifacts-dir", type=str, default="notebooks/artifacts/doc_understanding_ocr_cpu")

    parser.add_argument("--eval-csv", type=str, default="")
    parser.add_argument("--image-subdir-eval", type=str, default="batch_2/batch_2/batch2_1")
    parser.add_argument(
        "--eval-image-subdirs",
        type=str,
        default="batch_2/batch_2/batch2_1,batch_2/batch_2/batch2_2,batch_2/batch_2/batch2_3",
        help="Comma-separated unlabeled validation image folders",
    )

    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--no-grayscale", action="store_true", help="Disable grayscale preprocessing")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-wer", action="store_true", help="Skip WER computation in supervised evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.ocr_image_text.config import InferConfig
    from src.ocr_image_text.data import load_images_from_subdirs, load_ocr_csv, resolve_default_data_root
    from src.ocr_image_text.evaluation import evaluate_records, summarize_predictions
    from src.ocr_image_text.inference import load_predictor

    data_root = resolve_default_data_root(args.data_root)
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    predictor = load_predictor(
        InferConfig(
            artifacts_dir=artifacts_dir,
            image_size=args.image_size,
            use_grayscale=not args.no_grayscale,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
        )
    )

    payload: dict = {
        "schema_version": "ocr_image_eval.v2",
        "decode": {
            "num_beams": args.num_beams,
            "length_penalty": args.length_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "repetition_penalty": args.repetition_penalty,
        },
        "image_size": args.image_size,
        "use_grayscale": not args.no_grayscale,
        "compute_wer": not args.no_wer,
    }
    effective_size: int | None = None

    eval_csv_path = (data_root / args.eval_csv) if args.eval_csv else None
    eval_img_dir = data_root / args.image_subdir_eval

    if eval_csv_path and eval_csv_path.exists() and eval_img_dir.exists():
        records = load_ocr_csv(eval_csv_path, eval_img_dir)
        if args.max_samples > 0:
            records = records[: args.max_samples]

        preds = {}
        for rec in records:
            output = predictor.predict(rec.image_path)
            if effective_size is None:
                effective_size = int(output.get("effective_image_size", args.image_size))
            preds[rec.img_name] = str(output.get("prediction", ""))

        payload["mode"] = "supervised"
        payload["metrics"] = evaluate_records(records, preds, compute_wer=not args.no_wer)
        payload["num_samples"] = len(records)
    else:
        subdirs = tuple(s.strip() for s in args.eval_image_subdirs.split(",") if s.strip())
        image_paths = load_images_from_subdirs(data_root, subdirs)
        if args.max_samples > 0:
            image_paths = image_paths[: args.max_samples]

        rows = []
        lat_sum = 0.0
        for p in image_paths:
            out = predictor.predict(p)
            if effective_size is None:
                effective_size = int(out.get("effective_image_size", args.image_size))
            pred = str(out.get("prediction", ""))
            lat = float(out.get("latency_ms", 0.0))
            rows.append({
                "image_path": str(p),
                "prediction": pred,
                "latency_ms": lat,
            })
            lat_sum += lat

        pred_path = artifacts_dir / "batch2_validation_predictions.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        n = len(rows)
        payload["mode"] = "unlabeled_batch2_validation"
        payload["num_samples"] = n
        payload["metrics"] = summarize_predictions(r["prediction"] for r in rows)
        payload["avg_latency_ms"] = (lat_sum / n) if n else 0.0
        payload["predictions_path"] = str(pred_path)
        payload["sample_predictions"] = rows[:5]

    payload["effective_image_size"] = int(effective_size or args.image_size)

    out_path = artifacts_dir / "eval_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"metrics_path": str(out_path), "metrics": payload}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
