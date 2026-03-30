from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU inference (OCR text -> JSON) with fine-tuned FLAN-T5 model.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/doc_understanding_cpu")
    parser.add_argument("--ocr-text", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.doc_understanding_cpu.config import CPUInferConfig
    from src.doc_understanding_cpu.inference import load_cpu_predictor

    cfg = CPUInferConfig(artifacts_dir=Path(args.artifacts_dir).resolve(), max_new_tokens=args.max_new_tokens)
    predictor = load_cpu_predictor(cfg)
    out = predictor.predict(args.ocr_text, max_new_tokens=args.max_new_tokens)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
