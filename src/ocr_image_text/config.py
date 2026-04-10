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
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = False
    dataloader_persistent_workers: bool = False
    fp16: bool = False
    bf16: bool = False
    auto_hardware_profile: bool = True
    random_seed: int = 3407
    lr_scheduler_type: str = "cosine"
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.0
    metric_for_best_model: str = "eval_cer"
    greater_is_better: bool = False
    allow_long_training: bool = False
    require_supervised_eval: bool = True
    allow_unlabeled_eval: bool = False
    generation_num_beams: int = 1
    generation_length_penalty: float = 1.0
    generation_no_repeat_ngram_size: int = 5
    generation_repetition_penalty: float = 1.25


@dataclass(frozen=True)
class InferConfig:
    artifacts_dir: Path
    image_size: int = 768
    use_grayscale: bool = True
    max_new_tokens: int = 64
    num_beams: int = 1
    temperature: float = 0.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 5
    repetition_penalty: float = 1.3
    segmentation_mode: str = "line_only"
    max_chars_per_segment: int = 256
    max_total_chars: int = 1200
    max_invoice_markers_per_page: int = 1
    hard_truncate_segment_text: bool = True
    max_crops: int = 28
    crop_batch_size: int = 6
