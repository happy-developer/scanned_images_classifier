from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.doc_understanding.config import TrainConfig
from src.doc_understanding.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train doc-understanding model from Kaggle invoice dataset.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="artifacts/doc_understanding")
    parser.add_argument("--train-csv", type=str, default="batch_1/batch_1/batch1_1.csv")
    parser.add_argument("--eval-csv", type=str, default="batch_1/batch_1/batch1_2.csv")
    parser.add_argument("--image-subdir-train", type=str, default="batch_1/batch_1/batch1_1")
    parser.add_argument("--image-subdir-eval", type=str, default="batch_1/batch_1/batch1_2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="Run smoke mode without full fine-tuning.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        data_root=Path(args.data_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        image_subdir_train=args.image_subdir_train,
        image_subdir_eval=args.image_subdir_eval,
        train_epochs=args.epochs,
        smoke_mode=args.smoke,
    )
    summary = run_training(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
