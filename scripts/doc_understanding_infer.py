from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.doc_understanding.config import InferConfig
from src.doc_understanding.inference import load_predictor
from src.doc_understanding.formatting import INSTRUCTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned doc-understanding model.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/doc_understanding")
    parser.add_argument("--data-root", type=str, default="data/kaggle_invoice_images")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--instruction", type=str, default=None)
    return parser.parse_args()


def _resolve_image(image_arg: str, data_root: Path) -> Path:
    candidate = Path(image_arg)
    if candidate.exists():
        return candidate.resolve()
    candidate2 = (data_root / image_arg)
    if candidate2.exists():
        return candidate2.resolve()
    raise FileNotFoundError(f"Image not found: {image_arg}")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    cfg = InferConfig(artifacts_dir=Path(args.artifacts_dir).resolve())
    predictor = load_predictor(cfg)
    image_path = _resolve_image(args.image, data_root)
    result = predictor.predict(image_path, instruction=args.instruction or INSTRUCTION)
    result["data_root"] = str(data_root)
    result["image_path"] = str(image_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
