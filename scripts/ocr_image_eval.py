from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR image->text model on batch_1 eval split.")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--artifacts-dir", type=str, default="notebooks/artifacts/doc_understanding_ocr_cpu")
    parser.add_argument("--eval-csv", type=str, default="batch_1/batch_1/batch1_2.csv")
    parser.add_argument("--image-subdir-eval", type=str, default="batch_1/batch_1/batch1_2")
    parser.add_argument(
        "--train-csvs",
        type=str,
        default="batch_1/batch_1/batch1_1.csv,batch_1/batch_1/batch1_3.csv",
        help="Comma-separated training CSV list (for dataset loading parity)",
    )
    parser.add_argument(
        "--image-subdirs-train",
        type=str,
        default="batch_1/batch_1/batch1_1,batch_1/batch_1/batch1_3",
        help="Comma-separated image subdir list matching --train-csvs order",
    )
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)

    parser.add_argument(
        "--batch2-annotations-csv",
        type=str,
        default="",
        help="Optional manual annotation CSV for batch_2 (columns: File Name, OCRed Text)",
    )
    parser.add_argument(
        "--image-subdir-batch2",
        type=str,
        default="batch_2/batch_2/batch2_1",
        help="Image folder for batch_2 manual annotation CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.ocr_image_text.config import InferConfig
    from src.ocr_image_text.data import load_default_train_eval, load_ocr_csv, resolve_default_data_root
    from src.ocr_image_text.evaluation import evaluate_records
    from src.ocr_image_text.inference import load_predictor

    data_root = resolve_default_data_root(args.data_root)
    train_csvs = tuple(s.strip() for s in args.train_csvs.split(",") if s.strip())
    image_subdirs_train = tuple(s.strip() for s in args.image_subdirs_train.split(",") if s.strip())

    _, eval_records = load_default_train_eval(
        data_root=data_root,
        train_csvs=train_csvs,
        eval_csv=args.eval_csv,
        image_subdirs_train=image_subdirs_train,
        image_subdir_eval=args.image_subdir_eval,
    )
    if args.max_samples > 0:
        eval_records = eval_records[: args.max_samples]

    predictor = load_predictor(
        InferConfig(
            artifacts_dir=Path(args.artifacts_dir).resolve(),
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
        )
    )
    preds = {}
    for rec in eval_records:
        output = predictor.predict(rec.image_path)
        preds[rec.img_name] = str(output.get("prediction", ""))

    eval_metrics = evaluate_records(eval_records, preds)

    payload = {
        "batch1_eval": eval_metrics,
        "num_samples_requested": args.max_samples,
        "num_samples_effective": len(eval_records),
    }

    if args.batch2_annotations_csv:
        batch2_records = load_ocr_csv(
            data_root / args.batch2_annotations_csv,
            data_root / args.image_subdir_batch2,
        )
        batch2_preds = {}
        for rec in batch2_records:
            output = predictor.predict(rec.image_path)
            batch2_preds[rec.img_name] = str(output.get("prediction", ""))
        payload["batch2_manual_eval"] = evaluate_records(batch2_records, batch2_preds)
        payload["batch2_manual_num_samples"] = len(batch2_records)

    out_path = Path(args.artifacts_dir).resolve() / "eval_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"metrics_path": str(out_path), "metrics": payload}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
