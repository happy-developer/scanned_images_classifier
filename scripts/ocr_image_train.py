from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BATCH1_TRAIN_SPLIT_REL = "outputs/batch1_train_split.csv"
BATCH1_TEST_SPLIT_REL = "outputs/batch1_test_split.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OCR image->text model on batch_1 CSV labels.")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument(
        "--use-batch1-split",
        action="store_true",
        help=(
            "Use batch_1 split CSVs from outputs/batch1_train_split.csv and outputs/batch1_test_split.csv "
            "with image subdir set to '.'."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="notebooks/artifacts/doc_understanding_ocr_cpu")
    parser.add_argument("--model-name", type=str, default="microsoft/trocr-small-printed")
    parser.add_argument(
        "--train-csvs",
        type=str,
        default="batch_1/batch_1/batch1_1.csv,batch_1/batch_1/batch1_2.csv,batch_1/batch_1/batch1_3.csv",
        help="Comma-separated training CSV list",
    )
    parser.add_argument("--eval-csv", type=str, default="")
    parser.add_argument(
        "--image-subdirs-train",
        type=str,
        default="batch_1/batch_1/batch1_1,batch_1/batch_1/batch1_2,batch_1/batch_1/batch1_3",
        help="Comma-separated image subdir list matching --train-csvs order",
    )
    parser.add_argument("--image-subdir-eval", type=str, default="batch_2/batch_2/batch2_1")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--require-supervised-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a labeled eval CSV for training; use --allow-unlabeled-eval to bypass this guard.",
    )
    parser.add_argument(
        "--allow-unlabeled-eval",
        action="store_true",
        help="Allow training to proceed without a supervised eval CSV.",
    )
    parser.add_argument(
        "--allow-long-training",
        action="store_true",
        help="Allow epochs greater than 3.",
    )
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--no-grayscale", action="store_true", help="Disable grayscale preprocessing")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.ocr_image_text.config import TrainConfig
    from src.ocr_image_text.data import resolve_default_data_root
    from src.ocr_image_text.train import run_training

    data_root = resolve_default_data_root(args.data_root)

    if args.use_batch1_split:
        split_train_path = (data_root / BATCH1_TRAIN_SPLIT_REL).resolve()
        split_test_path = (data_root / BATCH1_TEST_SPLIT_REL).resolve()
        missing = [str(path) for path in (split_train_path, split_test_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Batch1 split mode enabled but split CSV files are missing: "
                + "; ".join(missing)
                + ". Expected files: outputs/batch1_train_split.csv and outputs/batch1_test_split.csv under --data-root."
            )

        train_csvs = (BATCH1_TRAIN_SPLIT_REL,)
        eval_csv = BATCH1_TEST_SPLIT_REL
        image_subdirs_train = (".",)
        image_subdir_eval = "."
    else:
        train_csvs = tuple(s.strip() for s in args.train_csvs.split(",") if s.strip())
        eval_csv = args.eval_csv
        image_subdirs_train = tuple(s.strip() for s in args.image_subdirs_train.split(",") if s.strip())
        image_subdir_eval = args.image_subdir_eval

    cfg = TrainConfig(
        data_root=data_root,
        output_dir=Path(args.output_dir).resolve(),
        model_name=args.model_name,
        train_csvs=train_csvs,
        eval_csv=eval_csv,
        image_subdirs_train=image_subdirs_train,
        image_subdir_eval=image_subdir_eval,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        train_epochs=args.epochs,
        allow_long_training=args.allow_long_training,
        require_supervised_eval=args.require_supervised_eval,
        allow_unlabeled_eval=args.allow_unlabeled_eval,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        use_grayscale=not args.no_grayscale,
        generation_num_beams=args.num_beams,
        generation_length_penalty=args.length_penalty,
        generation_no_repeat_ngram_size=args.no_repeat_ngram_size,
        generation_repetition_penalty=args.repetition_penalty,
    )
    summary = run_training(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
