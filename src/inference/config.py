from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceConfig:
    model_path: Path
    data_root: Path
    logs_path: Path
    model_meta_path: Path
    host: str = "127.0.0.1"
    port: int = 7860


def default_config() -> InferenceConfig:
    project_root = Path(__file__).resolve().parents[2]
    return InferenceConfig(
        model_path=project_root / "artifacts" / "scanned_images_resnet18.pt",
        data_root=project_root / "data" / "kaggle_invoice_images",
        logs_path=project_root / "artifacts" / "inference_runs.jsonl",
        model_meta_path=project_root / "artifacts" / "model_meta.json",
    )
