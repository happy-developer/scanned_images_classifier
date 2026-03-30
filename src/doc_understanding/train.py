from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from .config import TrainConfig
from .data import load_default_train_eval
from .formatting import record_to_messages


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_conversation_dataset(records):
    dataset = []
    for rec in records:
        img = Image.open(rec.image_path).convert("RGB")
        dataset.append({"messages": record_to_messages(rec, img)})
    return dataset


def _run_unsloth_train(config: TrainConfig, train_records):
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer
    import torch

    model, processor = FastVisionModel.from_pretrained(
        model_name="unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
        dtype=None,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        full_finetuning=False,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=32,
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        random_state=config.random_seed,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
    )

    converted = _build_conversation_dataset(train_records)
    trainer = SFTTrainer(
        model=model,
        train_dataset=converted,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=SFTConfig(
            per_device_train_batch_size=config.per_device_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            num_train_epochs=config.train_epochs,
            learning_rate=config.learning_rate,
            logging_steps=10,
            save_strategy="steps",
            save_steps=50,
            optim="adamw_torch_fused",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=config.random_seed,
            output_dir=str(config.output_dir),
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=config.max_seq_length,
        ),
    )

    start = time.time()
    stats = trainer.train()
    runtime_sec = time.time() - start

    checkpoint_dir = config.output_dir / "model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))
    processor.save_pretrained(str(checkpoint_dir))

    return {
        "mode": "full_train",
        "train_runtime_sec": runtime_sec,
        "train_metrics": getattr(stats, "metrics", {}),
        "checkpoint_dir": str(checkpoint_dir),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def run_training(config: TrainConfig) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_records, eval_records = load_default_train_eval(
        data_root=config.data_root,
        train_csv=config.train_csv,
        eval_csv=config.eval_csv,
        image_subdir_train=config.image_subdir_train,
        image_subdir_eval=config.image_subdir_eval,
    )

    summary: Dict[str, Any] = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "num_train_records": len(train_records),
        "num_eval_records": len(eval_records),
    }

    if config.smoke_mode:
        smoke_payload = {
            **summary,
            "mode": "smoke",
            "message": "Smoke mode active: no fine-tuning executed.",
            "artifacts": {"smoke_manifest": str((config.output_dir / "smoke_manifest.json").resolve())},
        }
        _write_json(config.output_dir / "smoke_manifest.json", smoke_payload)
        _write_json(config.output_dir / "train_summary.json", smoke_payload)
        return smoke_payload

    try:
        train_payload = _run_unsloth_train(config, train_records)
        payload = {**summary, **train_payload}
    except Exception as exc:
        payload = {
            **summary,
            "mode": "fallback_smoke_after_error",
            "error": str(exc),
            "message": "Full fine-tuning failed in this environment; fallback artifact generated.",
        }
        _write_json(config.output_dir / "smoke_manifest.json", payload)

    _write_json(config.output_dir / "train_summary.json", payload)
    return payload
