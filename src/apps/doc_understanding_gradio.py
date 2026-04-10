from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

DEFAULT_INSTRUCTION = (
    "Extract all information from this invoice image and return a JSON object "
    "with keys: client_name, client_address, seller_name, seller_address, "
    "invoice_number, invoice_date."
)


class AppError(RuntimeError):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


@dataclass(frozen=True)
class AppConfig:
    model_dir: Path
    model_meta_path: Optional[Path]
    host: str
    port: int


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts" / "doc_understanding" / "model"


def _default_model_meta_path() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts" / "doc_understanding" / "model_meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lancer l'application Gradio doc understanding.")
    parser.add_argument("--model-dir", type=str, default=str(_default_model_dir()))
    parser.add_argument("--model-meta-path", type=str, default=str(_default_model_meta_path()))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    return parser.parse_args()


def _load_model_version(model_dir: Path, model_meta_path: Optional[Path]) -> str:
    if model_meta_path and model_meta_path.exists():
        try:
            data = json.loads(model_meta_path.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict) and data.get("version"):
                return str(data["version"])
        except Exception:
            pass
    return model_dir.name


def _extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _load_transformers_backend(model_dir: Path):
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq, AutoProcessor
    except Exception as exc:
        raise AppError(
            code="DEPENDENCY_MISSING",
            message="Dependances inference manquantes (transformers/torch).",
            details={"reason": str(exc)},
        ) from exc

    if not model_dir.exists():
        raise AppError(
            code="MODEL_NOT_FOUND",
            message="Repertoire modele introuvable.",
            details={"model_dir": str(model_dir)},
        )

    processor = AutoProcessor.from_pretrained(str(model_dir))

    model = None
    load_errors = []
    for cls in (AutoModelForImageTextToText, AutoModelForVision2Seq):
        try:
            model = cls.from_pretrained(str(model_dir))
            break
        except Exception as exc:
            load_errors.append(f"{cls.__name__}: {exc}")

    if model is None:
        raise AppError(
            code="MODEL_LOAD_FAILED",
            message="Impossible de charger le modele fine-tune depuis le dossier fourni.",
            details={"model_dir": str(model_dir), "errors": load_errors},
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, processor, device


def build_predict_fn(config: AppConfig):
    model, processor, device = _load_transformers_backend(config.model_dir)
    model_version = _load_model_version(config.model_dir, config.model_meta_path)

    def _predict(image: Image.Image, instruction: str, max_new_tokens: int, temperature: float) -> Dict[str, Any]:
        if image is None:
            return AppError("INVALID_INPUT", "Aucune image fournie.").to_dict()
        if not isinstance(image, Image.Image):
            return AppError("INVALID_INPUT", "Le fichier fourni n'est pas une image valide.").to_dict()

        prompt = (instruction or "").strip() or DEFAULT_INSTRUCTION
        if image.mode != "RGB":
            image = image.convert("RGB")

        try:
            import torch

            t0 = time.perf_counter()
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    do_sample=float(temperature) > 0.0,
                )
            decoded = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            parsed = _extract_json_blob(decoded)
            return {
                "prediction_json": parsed,
                "raw_text": decoded,
                "latency_ms": latency_ms,
                "model_version": model_version,
                "status": "ok",
            }
        except AppError as exc:
            return exc.to_dict()
        except Exception as exc:
            return AppError(
                code="INFERENCE_FAILED",
                message="Erreur lors de l'inference.",
                details={"reason": str(exc)},
            ).to_dict()

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

    with gr.Blocks(title="Doc Understanding - Gradio") as app:
        gr.Markdown("# Doc Understanding - Inference")
        gr.Markdown("Upload un document (image) et extrais les informations en JSON.")

        with gr.Row():
            image_input = gr.Image(type="pil", label="Document image")
            with gr.Column():
                instruction_input = gr.Textbox(label="Instruction", value=DEFAULT_INSTRUCTION, lines=4)
                max_tokens_input = gr.Slider(minimum=32, maximum=1024, step=16, value=256, label="max_new_tokens")
                temperature_input = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.0, label="temperature")
                run_button = gr.Button("Run inference", variant="primary")

        output_json = gr.JSON(label="Result")

        run_button.click(
            fn=predict_fn,
            inputs=[image_input, instruction_input, max_tokens_input, temperature_input],
            outputs=[output_json],
        )

    return app


def main() -> None:
    args = parse_args()
    config = AppConfig(
        model_dir=Path(args.model_dir),
        model_meta_path=Path(args.model_meta_path) if args.model_meta_path else None,
        host=args.host,
        port=args.port,
    )

    try:
        app = build_app(config)
    except AppError as exc:
        raise SystemExit(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))

    app.launch(server_name=config.host, server_port=config.port)


if __name__ == "__main__":
    main()

