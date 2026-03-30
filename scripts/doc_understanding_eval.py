from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.doc_understanding.config import InferConfig, TrainConfig
from src.doc_understanding.data import load_default_train_eval
from src.doc_understanding.evaluation import evaluate_records
from src.doc_understanding.inference import load_predictor
from src.doc_understanding.formatting import INSTRUCTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate doc-understanding model on eval split.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/doc_understanding")
    parser.add_argument("--eval-csv", type=str, default="batch_1/batch_1/batch1_2.csv")
    parser.add_argument("--image-subdir-eval", type=str, default="batch_1/batch_1/batch1_2")
    parser.add_argument("--train-csv", type=str, default="batch_1/batch_1/batch1_1.csv")
    parser.add_argument("--image-subdir-train", type=str, default="batch_1/batch_1/batch1_1")
    parser.add_argument("--max-samples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = TrainConfig(
        data_root=Path(args.data_root).resolve(),
        output_dir=Path(args.artifacts_dir).resolve(),
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        image_subdir_train=args.image_subdir_train,
        image_subdir_eval=args.image_subdir_eval,
    )
    _, eval_records = load_default_train_eval(
        data_root=train_cfg.data_root,
        train_csv=train_cfg.train_csv,
        eval_csv=train_cfg.eval_csv,
        image_subdir_train=train_cfg.image_subdir_train,
        image_subdir_eval=train_cfg.image_subdir_eval,
    )
    eval_records = eval_records[: args.max_samples]

    predictor = load_predictor(InferConfig(artifacts_dir=Path(args.artifacts_dir).resolve()))
    preds = {}
    for rec in eval_records:
        output = predictor.predict(rec.image_path, instruction=INSTRUCTION)
        pred = output.get("prediction")
        preds[rec.img_name] = pred if isinstance(pred, dict) else {}

    metrics = evaluate_records(eval_records, preds)
    out_path = Path(args.artifacts_dir).resolve() / "eval_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "metrics_path": str(out_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
