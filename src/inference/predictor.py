from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from PIL import Image

from src.data_access.dataset_checks import DatasetContext

from .errors import InferenceExecutionError
from .model_loader import LoadedModel
from .preprocess import build_eval_transform, ensure_valid_image, infer_dataset_context_for_image
from .tracking import append_run, new_timing_ms, now_epoch_ms


@dataclass
class Predictor:
    loaded_model: LoadedModel
    dataset_context: DatasetContext
    logs_path: Path

    def predict(self, image: Image.Image, threshold: float = 0.5, image_path: str | None = None) -> Dict[str, object]:
        img = ensure_valid_image(image)
        tfm = build_eval_transform(self.loaded_model.img_size)

        started = time.perf_counter()
        try:
            with torch.no_grad():
                tensor = tfm(img).unsqueeze(0)
                logits = self.loaded_model.model(tensor)
                probs_tensor = torch.softmax(logits, dim=1)[0].cpu()
        except Exception as exc:
            raise InferenceExecutionError(details={"reason": str(exc)}) from exc

        probs = {
            class_name: float(probs_tensor[idx].item())
            for idx, class_name in enumerate(self.loaded_model.class_names)
        }
        top_label = max(probs, key=probs.get)
        confidence = float(probs[top_label])
        label = top_label if confidence >= float(threshold) else "unknown"

        payload: Dict[str, object] = {
            "run_id": str(uuid.uuid4()),
            "label": label,
            "confidence": confidence,
            "probs": probs,
            "model_version": self.loaded_model.model_version,
            "latency_ms": new_timing_ms(started),
            "dataset_context": {
                "root": self.dataset_context.root,
                "mode": self.dataset_context.mode,
                "num_classes": self.dataset_context.num_classes,
                "image_context": infer_dataset_context_for_image(image_path=image_path, dataset_root=Path(self.dataset_context.root)),
            },
        }

        append_run(
            self.logs_path,
            {
                "event": "inference_run",
                "timestamp_ms": now_epoch_ms(),
                "model_path": str(self.loaded_model.model_path),
                "result": payload,
            },
        )
        return payload
