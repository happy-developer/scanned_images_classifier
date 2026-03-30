from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET_ID = "osamahosamabdellatif/high-quality-invoice-images-for-ocr"


@dataclass(frozen=True)
class TrainConfig:
    data_root: Path
    output_dir: Path
    train_csv: str = "batch_1/batch_1/batch1_1.csv"
    eval_csv: str = "batch_1/batch_1/batch1_2.csv"
    image_subdir_train: str = "batch_1/batch_1/batch1_1"
    image_subdir_eval: str = "batch_1/batch_1/batch1_2"
    max_seq_length: int = 2048
    learning_rate: float = 2e-4
    train_epochs: int = 2
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    random_seed: int = 3407
    load_in_4bit: bool = True
    smoke_mode: bool = False


@dataclass(frozen=True)
class InferConfig:
    artifacts_dir: Path
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
