from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OCR image->text model on batch_1 CSV labels.")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="artifacts/doc_understanding_ocr_cpu")
    parser.add_argument("--model-name", type=str, default="microsoft/trocr-small-printed")
    parser.add_argument("--train-csv", type=str, default="batch_1/batch_1/batch1_1.csv")
    parser.add_argument("--eval-csv", type=str, default="batch_1/batch_1/batch1_2.csv")
    parser.add_argument("--image-subdir-train", type=str, default="batch_1/batch_1/batch1_1")
    parser.add_argument("--image-subdir-eval", type=str, default="batch_1/batch_1/batch1_2")
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.ocr_image_text.config import TrainConfig
    from src.ocr_image_text.data import resolve_default_data_root
    from src.ocr_image_text.train import run_training

    data_root = resolve_default_data_root(args.data_root)
    cfg = TrainConfig(
        data_root=data_root,
        output_dir=Path(args.output_dir).resolve(),
        model_name=args.model_name,
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        image_subdir_train=args.image_subdir_train,
        image_subdir_eval=args.image_subdir_eval,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
    )
    summary = run_training(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
