from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class AppError(RuntimeError):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "extracted_text": "",
            "normalized_text": "",
            "latency_ms": 0.0,
            "model_version": "unknown",
            "status": "error",
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }


@dataclass(frozen=True)
class AppConfig:
    model_dir: Path
    model_meta_path: Optional[Path]
    host: str
    port: int
    image_size: int
    use_grayscale: bool


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "notebooks" / "artifacts" / "doc_understanding_ocr_cpu" / "model"


def _default_model_meta_path() -> Path:
    return Path(__file__).resolve().parents[2] / "notebooks" / "artifacts" / "doc_understanding_ocr_cpu" / "model_meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lancer l'application Gradio OCR CPU (image -> texte).")
    parser.add_argument("--model-dir", type=str, default=str(_default_model_dir()))
    parser.add_argument("--model-meta-path", type=str, default=str(_default_model_meta_path()))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--no-grayscale", action="store_true", help="Disable grayscale preprocessing")
    return parser.parse_args()


def _load_model_version(model_dir: Path, model_meta_path: Optional[Path]) -> str:
    if model_meta_path and model_meta_path.exists():
        try:
            payload = json.loads(model_meta_path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, dict) and payload.get("version"):
                return str(payload["version"])
        except Exception:
            pass
    return f"{model_dir.name}-ocr-cpu"


def build_predict_fn(config: AppConfig):
    try:
        from src.ocr_image_text.config import InferConfig
        from src.ocr_image_text.inference import load_predictor
    except Exception as exc:
        raise AppError(
            code="DEPENDENCY_MISSING",
            message="Dependances OCR manquantes.",
            details={"reason": str(exc)},
        ) from exc

    if not config.model_dir.exists():
        raise AppError(
            code="MODEL_NOT_FOUND",
            message="Repertoire modele OCR introuvable.",
            details={"model_dir": str(config.model_dir)},
        )

    infer_cfg = InferConfig(
        artifacts_dir=config.model_dir.parent,
        image_size=config.image_size,
        use_grayscale=config.use_grayscale,
    )
    predictor = load_predictor(infer_cfg)
    model_version = _load_model_version(config.model_dir, config.model_meta_path)

    def _predict(
        image_path: str,
        max_new_tokens: int,
        num_beams: int,
        no_repeat_ngram_size: int,
        repetition_penalty: float,
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not image_path:
            err = AppError("INVALID_INPUT", "Aucune image fournie.").to_dict()
            return "", "", err

        try:
            result = predictor.predict(
                Path(image_path),
                max_new_tokens=int(max_new_tokens),
                num_beams=max(1, int(num_beams)),
                temperature=0.0,
                length_penalty=1.0,
                no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
                repetition_penalty=max(1.0, float(repetition_penalty)),
            )
            raw_text = str(result.get("raw_output", ""))
            normalized_text = str(result.get("prediction", ""))
            payload = {
                "extracted_text": raw_text,
                "normalized_text": normalized_text,
                "latency_ms": float(result.get("latency_ms", 0.0)),
                "model_version": model_version,
                "status": "ok",
                "error": None,
                "segmentation_strategy": result.get("segmentation_strategy", "unknown"),
                "used_full_page_fallback": bool(result.get("used_full_page_fallback", False)),
                "fallback_reason": result.get("fallback_reason"),
                "crop_count": int(result.get("crop_count", 0)),
                "effective_image_size": int(result.get("effective_image_size", config.image_size)),
                "use_grayscale": bool(config.use_grayscale),
            }
            return raw_text, normalized_text, payload
        except Exception as exc:
            err = AppError(
                code="INFERENCE_FAILED",
                message="Erreur lors de l'inference OCR CPU.",
                details={"reason": str(exc)},
            ).to_dict()
            return "", "", err

    return _predict


def build_app(config: AppConfig):
    try:
        import gradio as gr
    except Exception as exc:
        raise AppError(
            code="DEPENDENCY_MISSING",
            message="Gradio n'est pas installe.",
            details={"reason": str(exc)},
        ) from exc

    predict_fn = build_predict_fn(config)

    with gr.Blocks(title="OCR CPU - Gradio") as app:
        gr.Markdown("# OCR CPU - Image vers texte complet")
        gr.Markdown("Pipeline: segmentation lignes/blocs -> TrOCR par crop -> concaténation.")

        with gr.Row():
            image_input = gr.Image(type="filepath", label="Document image")
            with gr.Column():
                max_tokens_input = gr.Slider(minimum=16, maximum=1024, step=16, value=192, label="max_new_tokens")
                num_beams_input = gr.Slider(minimum=1, maximum=8, step=1, value=4, label="num_beams")
                no_repeat_input = gr.Slider(minimum=0, maximum=8, step=1, value=4, label="no_repeat_ngram_size")
                repetition_penalty_input = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    step=0.05,
                    value=1.15,
                    label="repetition_penalty",
                )
                run_button = gr.Button("Run OCR", variant="primary")

        raw_text_output = gr.Textbox(label="Texte OCR brut", lines=10)
        normalized_text_output = gr.Textbox(label="Texte OCR normalisé", lines=10)
        output_json = gr.JSON(label="Détails inférence")

        run_button.click(
            fn=predict_fn,
            inputs=[image_input, max_tokens_input, num_beams_input, no_repeat_input, repetition_penalty_input],
            outputs=[raw_text_output, normalized_text_output, output_json],
        )

    return app


def main() -> None:
    args = parse_args()
    config = AppConfig(
        model_dir=Path(args.model_dir),
        model_meta_path=Path(args.model_meta_path) if args.model_meta_path else None,
        host=args.host,
        port=args.port,
        image_size=int(args.image_size),
        use_grayscale=not bool(args.no_grayscale),
    )

    try:
        app = build_app(config)
    except AppError as exc:
        raise SystemExit(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))

    app.launch(server_name=config.host, server_port=config.port)


if __name__ == "__main__":
    main()
