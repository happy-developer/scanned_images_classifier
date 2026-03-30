from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_NAME = "google/flan-t5-small"
TARGET_FIELDS = (
    "client_name",
    "client_address",
    "seller_name",
    "seller_address",
    "invoice_number",
    "invoice_date",
)


@dataclass(frozen=True)
class CPUTrainConfig:
    data_root: Path
    output_dir: Path
    model_name: str = DEFAULT_MODEL_NAME
    train_csv: str = "batch_1/batch_1/batch1_1.csv"
    eval_csv: str = "batch_1/batch_1/batch1_2.csv"
    max_train_samples: int = 128
    max_eval_samples: int = 64
    num_train_epochs: int = 50
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-5
    max_source_length: int = 512
    max_target_length: int = 256
    seed: int = 3407


@dataclass(frozen=True)
class CPUInferConfig:
    artifacts_dir: Path
    max_new_tokens: int = 256

