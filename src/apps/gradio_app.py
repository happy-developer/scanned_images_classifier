from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_access.dataset_checks import validate_dataset_structure
from src.data_access.kagglehub_resolver import resolve_kaggle_dataset_root
from src.inference.config import default_config
from src.inference.errors import (
    DatasetUnavailableError,
    InferenceError,
    InferenceExecutionError,
)
from src.inference.model_loader import load_model
from src.inference.predictor import Predictor


def _build_error_payload(exc: Exception) -> Dict[str, Any]:
    if isinstance(exc, InferenceError):
        return exc.to_dict()
    return {
        "error": {
            "code": "INFERENCE_FAILED",
            "message": "Erreur inattendue inference",
            "details": {"reason": str(exc)},
        }
    }


def create_predict_fn(predictor: Predictor):
    def _predict(image, threshold: float):
        try:
            return predictor.predict(image=image, threshold=float(threshold))
        except Exception as exc:
            return _build_error_payload(exc)

    return _predict


def build_app(predictor: Predictor):
    import gradio as gr

    predict_fn = create_predict_fn(predictor)

    with gr.Blocks(title="Kaggle Inference Gradio") as app:
        gr.Markdown("# Inference Kaggle - Gradio")
        gr.Markdown("Upload image + threshold. Sortie structuree succes/erreur.")

        image_input = gr.Image(type="pil", label="Image test")
        threshold_input = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="threshold")
        submit_btn = gr.Button("Predire", variant="primary")

        json_output = gr.JSON(label="Prediction output")

        submit_btn.click(fn=predict_fn, inputs=[image_input, threshold_input], outputs=[json_output])

    return app


def parse_args() -> argparse.Namespace:
    cfg = default_config()
    parser = argparse.ArgumentParser(description="Lancer Gradio inference Kaggle")
    parser.add_argument("--model-path", type=str, default=str(cfg.model_path))
    parser.add_argument("--data-root", type=str, default=str(cfg.data_root))
    parser.add_argument("--logs-path", type=str, default=str(cfg.logs_path))
    parser.add_argument("--model-meta-path", type=str, default=str(cfg.model_meta_path))
    parser.add_argument("--host", type=str, default=cfg.host)
    parser.add_argument("--port", type=int, default=cfg.port)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        data_root = resolve_kaggle_dataset_root(args.data_root)
        dataset_context = validate_dataset_structure(data_root)
    except Exception as exc:
        raise DatasetUnavailableError(details={"reason": str(exc), "data_root": args.data_root}) from exc

    loaded_model = load_model(
        model_path=Path(args.model_path),
        model_meta_path=Path(args.model_meta_path),
        dataset_context=dataset_context,
    )

    predictor = Predictor(
        loaded_model=loaded_model,
        dataset_context=dataset_context,
        logs_path=Path(args.logs_path),
    )

    app = build_app(predictor)
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    try:
        main()
    except InferenceError:
        raise
    except Exception as exc:
        raise InferenceExecutionError(details={"reason": str(exc)}) from exc
