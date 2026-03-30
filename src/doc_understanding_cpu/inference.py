from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import CPUInferConfig
from .data import PROMPT_TEMPLATE


@dataclass
class CPUPredictor:
    model: Any
    tokenizer: Any

    def predict(self, ocr_text: str, max_new_tokens: int = 256) -> Dict[str, Any]:
        prompt = PROMPT_TEMPLATE.format(ocr_text=ocr_text)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        start = time.perf_counter()
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        pred = None
        try:
            pred = json.loads(text)
        except Exception:
            pred = text
        return {"prediction": pred, "raw_output": text, "latency_ms": round(latency_ms, 3)}


def load_cpu_predictor(config: CPUInferConfig) -> CPUPredictor:
    model_dir = config.artifacts_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return CPUPredictor(model=model, tokenizer=tokenizer)
