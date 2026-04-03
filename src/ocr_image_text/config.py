from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_NAME = "microsoft/trocr-small-printed"


@dataclass(frozen=True)
class TrainConfig:
    data_root: Path
    output_dir: Path
    model_name: str = DEFAULT_MODEL_NAME
    # Multi-source train split support (same length as image_subdirs_train).
    train_csvs: tuple[str, ...] = (
        "batch_1/batch_1/batch1_1.csv",
        "batch_1/batch_1/batch1_2.csv",
        "batch_1/batch_1/batch1_3.csv",
    )
    # Optional supervised eval CSV. Leave empty when validating on unlabeled batch_2 images.
    eval_csv: str = ""
    image_subdirs_train: tuple[str, ...] = (
        "batch_1/batch_1/batch1_1",
        "batch_1/batch_1/batch1_2",
        "batch_1/batch_1/batch1_3",
    )
    image_subdir_eval: str = "batch_2/batch_2/batch2_1"
    # Use <=0 to train on the full train split.
    max_train_samples: int = 0
    # Use <=0 to evaluate on the full eval split.
    max_eval_samples: int = 0
    max_target_length: int = 256
    image_size: int = 768
    use_grayscale: bool = True
    learning_rate: float = 3e-5
    train_epochs: int = 10
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    random_seed: int = 3407
    lr_scheduler_type: str = "cosine"
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.0
    metric_for_best_model: str = "eval_cer"
    greater_is_better: bool = False
    generation_num_beams: int = 4
    generation_length_penalty: float = 1.0
    generation_no_repeat_ngram_size: int = 4
    generation_repetition_penalty: float = 1.15


@dataclass(frozen=True)
class InferConfig:
    artifacts_dir: Path
    image_size: int = 768
    use_grayscale: bool = True
    max_new_tokens: int = 192
    num_beams: int = 4
    temperature: float = 0.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 4
    repetition_penalty: float = 1.15
