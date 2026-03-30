from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU fine-tuning (FLAN-T5-small) for invoice doc understanding.")
    parser.add_argument("--data-root", type=str, default="data/kaggle_invoice_images")
    parser.add_argument("--output-dir", type=str, default="artifacts/doc_understanding_cpu")
    parser.add_argument("--model-name", type=str, default="google/flan-t5-small")
    parser.add_argument("--max-train-samples", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.doc_understanding_cpu.config import CPUTrainConfig
    from src.doc_understanding_cpu.train import run_cpu_training

    cfg = CPUTrainConfig(
        data_root=Path(args.data_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        model_name=args.model_name,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
    )
    summary = run_cpu_training(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

