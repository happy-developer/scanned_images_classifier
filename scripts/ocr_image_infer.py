from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR image->text inference on one image.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/doc_understanding_ocr_cpu")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def _resolve_image(image_arg: str, data_root: Path) -> Path:
    candidate = Path(image_arg)
    if candidate.exists():
        return candidate.resolve()
    candidate2 = data_root / image_arg
    if candidate2.exists():
        return candidate2.resolve()
    raise FileNotFoundError(f"Image not found: {image_arg}")


def main() -> None:
    args = parse_args()
    from src.ocr_image_text.config import InferConfig
    from src.ocr_image_text.data import resolve_default_data_root
    from src.ocr_image_text.inference import load_predictor

    data_root = resolve_default_data_root(args.data_root)
    cfg = InferConfig(
        artifacts_dir=Path(args.artifacts_dir).resolve(),
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        temperature=args.temperature,
    )
    predictor = load_predictor(cfg)
    image_path = _resolve_image(args.image, data_root)
    result = predictor.predict(image_path)
    result["data_root"] = str(data_root)
    result["image_path"] = str(image_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
