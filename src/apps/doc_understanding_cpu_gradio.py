from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image


class AppError(RuntimeError):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "extracted_text": "",
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


def _load_ocr_backend(model_dir: Path):
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, VisionEncoderDecoderModel
    except Exception as exc:
        raise AppError(
            code="DEPENDENCY_MISSING",
            message="Dependances OCR manquantes (transformers/torch).",
            details={"reason": str(exc)},
        ) from exc

    if not model_dir.exists():
        raise AppError(
            code="MODEL_NOT_FOUND",
            message="Repertoire modele OCR introuvable.",
            details={"model_dir": str(model_dir)},
        )

    try:
        processor = AutoProcessor.from_pretrained(str(model_dir))
    except Exception as exc:
        raise AppError(
            code="PROCESSOR_LOAD_FAILED",
            message="Impossible de charger le processor OCR depuis le dossier fourni.",
            details={"model_dir": str(model_dir), "reason": str(exc)},
        ) from exc

    model = None
    load_errors = []
    load_candidates = [VisionEncoderDecoderModel, AutoModelForImageTextToText]
    try:
        import transformers
        vision2seq = getattr(transformers, "AutoModelForVision2Seq", None)
        if vision2seq is not None:
            load_candidates.append(vision2seq)
    except Exception:
        pass

    for cls in load_candidates:
        try:
            model = cls.from_pretrained(str(model_dir))
            break
        except Exception as exc:
            load_errors.append(f"{cls.__name__}: {exc}")

    if model is None:
        raise AppError(
            code="MODEL_LOAD_FAILED",
            message="Impossible de charger le modele OCR fine-tune depuis le dossier fourni.",
            details={"model_dir": str(model_dir), "errors": load_errors},
        )

    # Normalize generation tokens to avoid empty-decoding collapses when
    # decoder_start_token_id is incorrectly set to eos in saved configs.
    try:
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            start_id = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
            if start_id is None:
                start_id = 0
            model.config.decoder_start_token_id = int(start_id)
            model.config.eos_token_id = tok.eos_token_id
            model.config.pad_token_id = tok.pad_token_id
            if hasattr(model, "generation_config"):
                model.generation_config.decoder_start_token_id = int(start_id)
                model.generation_config.eos_token_id = tok.eos_token_id
                model.generation_config.pad_token_id = tok.pad_token_id
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, processor, device


def _open_image(image_path: str) -> Image.Image:
    p = Path(image_path)
    if not p.exists():
        raise AppError(
            code="INVALID_INPUT",
            message="Le fichier image est introuvable.",
            details={"image_path": image_path},
        )

    try:
        image = Image.open(p)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as exc:
        raise AppError(
            code="INVALID_IMAGE",
            message="Le fichier fourni n'est pas une image lisible.",
            details={"image_path": image_path, "reason": str(exc)},
        ) from exc


def _prepare_generation_inputs(processor: Any, image: Image.Image, device: str) -> Dict[str, Any]:
    try:
        inputs = processor(images=image, return_tensors="pt")
    except Exception as exc:
        raise AppError(
            code="PREPROCESS_FAILED",
            message="Echec du pretraitement OCR de l'image.",
            details={"reason": str(exc)},
        ) from exc

    if hasattr(inputs, "to"):
        return inputs.to(device)

    moved: Dict[str, Any] = {}
    for key, value in dict(inputs).items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _decode_output(processor: Any, output_ids: Any) -> str:
    try:
        decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
    except Exception as exc:
        raise AppError(
            code="DECODE_FAILED",
            message="Echec du decodage OCR.",
            details={"reason": str(exc)},
        ) from exc

    if not decoded:
        return ""
    return str(decoded[0]).strip()


def build_predict_fn(config: AppConfig):
    model, processor, device = _load_ocr_backend(config.model_dir)
    model_version = _load_model_version(config.model_dir, config.model_meta_path)

    def _predict(image_path: str, max_new_tokens: int, num_beams: int) -> Tuple[str, Dict[str, Any]]:
        if not image_path:
            err = AppError("INVALID_INPUT", "Aucune image fournie.").to_dict()
            return "", err

        try:
            image = _open_image(image_path)
        except AppError as exc:
            err = exc.to_dict()
            return "", err

        try:
            import torch

            t0 = time.perf_counter()
            inputs = _prepare_generation_inputs(processor, image, device)
            tok = getattr(processor, "tokenizer", None)
            suppress_tokens = []
            if tok is not None:
                if getattr(tok, "bos_token_id", None) is not None:
                    suppress_tokens.append(int(tok.bos_token_id))
                if getattr(tok, "pad_token_id", None) is not None:
                    suppress_tokens.append(int(tok.pad_token_id))

            gen_kwargs = {
                "max_new_tokens": int(max_new_tokens),
                "num_beams": max(1, int(num_beams)),
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 3,
                "early_stopping": True,
            }
            if suppress_tokens:
                gen_kwargs["suppress_tokens"] = suppress_tokens

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    **gen_kwargs,
                )
            extracted_text = _decode_output(processor, output_ids)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            result = {
                "extracted_text": extracted_text,
                "latency_ms": latency_ms,
                "model_version": model_version,
                "status": "ok",
                "error": None,
            }
            return extracted_text, result
        except AppError as exc:
            err = exc.to_dict()
            return "", err
        except Exception as exc:
            err = AppError(
                code="INFERENCE_FAILED",
                message="Erreur lors de l'inference OCR CPU.",
                details={"reason": str(exc)},
            ).to_dict()
            return "", err

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
        gr.Markdown("# OCR CPU - Image vers texte")
        gr.Markdown("Upload une image, lance l'inference OCR fine-tunee, puis recupere le texte extrait.")

        with gr.Row():
            image_input = gr.Image(type="filepath", label="Document image")
            with gr.Column():
                max_tokens_input = gr.Slider(minimum=16, maximum=1024, step=16, value=256, label="max_new_tokens")
                num_beams_input = gr.Slider(minimum=1, maximum=8, step=1, value=4, label="num_beams")
                run_button = gr.Button("Run OCR", variant="primary")

        extracted_text_output = gr.Textbox(label="Extracted text", lines=12)
        output_json = gr.JSON(label="Result")

        run_button.click(
            fn=predict_fn,
            inputs=[image_input, max_tokens_input, num_beams_input],
            outputs=[extracted_text_output, output_json],
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




