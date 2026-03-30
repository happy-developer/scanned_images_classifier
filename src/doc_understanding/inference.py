from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from .config import InferConfig
from .formatting import INSTRUCTION, safe_extract_json


@dataclass
class Predictor:
    mode: str
    model: Any = None
    processor: Any = None

    def predict(self, image_path: Path, instruction: str = INSTRUCTION) -> Dict[str, Any]:
        image = Image.open(image_path).convert("RGB")

        if self.mode == "smoke":
            return {
                "mode": "smoke",
                "instruction": instruction,
                "prediction": {
                    "client_name": "",
                    "client_address": "",
                    "seller_name": "",
                    "seller_address": "",
                    "invoice_number": "",
                    "invoice_date": "",
                },
            }

        messages = [{"role": "user", "content": [{"type": "text", "text": instruction}, {"type": "image", "image": image}]}]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
        eos_id = self.processor.tokenizer.eos_token_id
        end_turn_id = self.processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        result = self.model.generate(
            **inputs,
            max_new_tokens=256,
            eos_token_id=[eos_id, end_turn_id],
            pad_token_id=eos_id,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
        )
        decoded = self.processor.batch_decode(result, skip_special_tokens=True)[0]
        parsed = safe_extract_json(decoded)
        return {"mode": "full", "raw_output": decoded, "prediction": parsed if parsed is not None else decoded}


def load_predictor(config: InferConfig) -> Predictor:
    model_dir = config.artifacts_dir / "model"
    smoke_manifest = config.artifacts_dir / "smoke_manifest.json"

    if smoke_manifest.exists():
        return Predictor(mode="smoke")

    if not model_dir.exists():
        raise FileNotFoundError(f"No trained model found in {model_dir}")

    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_name=str(model_dir),
        dtype=None,
        max_seq_length=2048,
        load_in_4bit=True,
        full_finetuning=False,
    )
    FastVisionModel.for_inference(model)
    return Predictor(mode="full", model=model, processor=processor)
