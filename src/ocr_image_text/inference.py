from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .config import InferConfig
from .formatting import normalize_text


@dataclass
class Predictor:
    model: Any
    processor: Any
    infer_config: InferConfig

    def predict(self, image_path: Path) -> Dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        start = time.perf_counter()
        tok = self.processor.tokenizer
        suppress_tokens = []
        if tok.bos_token_id is not None:
            suppress_tokens.append(int(tok.bos_token_id))
        if tok.pad_token_id is not None:
            suppress_tokens.append(int(tok.pad_token_id))
        gen_kwargs = {
            "pixel_values": inputs.pixel_values,
            "max_new_tokens": self.infer_config.max_new_tokens,
            "num_beams": max(1, int(self.infer_config.num_beams)),
            "length_penalty": float(self.infer_config.length_penalty),
            "no_repeat_ngram_size": max(0, int(self.infer_config.no_repeat_ngram_size)),
            "repetition_penalty": float(self.infer_config.repetition_penalty),
            "do_sample": float(self.infer_config.temperature) > 0.0,
        }
        if suppress_tokens:
            gen_kwargs["suppress_tokens"] = suppress_tokens
        if float(self.infer_config.temperature) > 0.0:
            gen_kwargs["temperature"] = float(self.infer_config.temperature)
        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0
        decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return {
            "prediction": normalize_text(decoded),
            "raw_output": decoded,
            "latency_ms": round(latency_ms, 3),
        }


def load_predictor(config: InferConfig) -> Predictor:
    model_dir = config.artifacts_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"No trained OCR model found in {model_dir}")

    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    tok = processor.tokenizer
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
        model.generation_config.num_beams = max(1, int(config.num_beams))
        model.generation_config.length_penalty = float(config.length_penalty)
        model.generation_config.no_repeat_ngram_size = max(0, int(config.no_repeat_ngram_size))
        model.generation_config.repetition_penalty = float(config.repetition_penalty)
    model.to("cpu")
    model.eval()
    return Predictor(model=model, processor=processor, infer_config=config)
