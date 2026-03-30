from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CPU doc-understanding model.")
    parser.add_argument("--data-root", type=str, default="data/kaggle_invoice_images")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/doc_understanding_cpu")
    parser.add_argument("--eval-csv", type=str, default="batch_1/batch_1/batch1_2.csv")
    parser.add_argument("--max-samples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.doc_understanding_cpu.config import CPUInferConfig
    from src.doc_understanding_cpu.inference import load_cpu_predictor
    from src.doc_understanding_cpu.data import load_cpu_records
    from src.doc_understanding_cpu.eval import evaluate_cpu_predictions, write_eval_metrics

    data_root = Path(args.data_root).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()

    predictor = load_cpu_predictor(CPUInferConfig(artifacts_dir=artifacts_dir))

    records = load_cpu_records(data_root / args.eval_csv)[: args.max_samples]
    predictions = {}
    for rec in records:
        out = predictor.predict(rec.ocr_text)
        pred = out.get("prediction")
        predictions[rec.img_name] = pred if isinstance(pred, dict) else {}

    metrics = evaluate_cpu_predictions(records, predictions)
    out = {
        "metrics": metrics,
        "metrics_path": str((artifacts_dir / "eval_metrics_cpu.json").resolve()),
    }
    write_eval_metrics(artifacts_dir / "eval_metrics_cpu.json", out)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
