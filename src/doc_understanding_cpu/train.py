from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .config import CPUTrainConfig
from .data import load_cpu_records, records_to_text2text


def _tokenize_batch(batch, tokenizer, max_source_length: int, max_target_length: int):
    model_inputs = tokenizer(batch["input_text"], truncation=True, padding="max_length", max_length=max_source_length)
    labels = tokenizer(text_target=batch["target_text"], truncation=True, padding="max_length", max_length=max_target_length)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class EpochMetricsPlotCallback(TrainerCallback):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.train_epochs = []
        self.train_losses = []
        self.eval_epochs = []
        self.eval_losses = []
        self.plot_path = self.output_dir / "training_metrics.png"

    def _render_plot(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        if not self.train_losses and not self.eval_losses:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if self.train_losses:
            ax.plot(self.train_epochs, self.train_losses, marker="o", label="train_loss")
        if self.eval_losses:
            ax.plot(self.eval_epochs, self.eval_losses, marker="o", label="eval_loss")
        ax.set_title("Training Metrics by Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)
        try:
            from IPython.display import Image, clear_output, display

            clear_output(wait=True)
            display(Image(filename=str(self.plot_path)))
        except Exception:
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs and "epoch" in logs:
            self.train_epochs.append(float(logs["epoch"]))
            self.train_losses.append(float(logs["loss"]))
            self._render_plot()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        if "eval_loss" in metrics and "epoch" in metrics:
            self.eval_epochs.append(float(metrics["epoch"]))
            self.eval_losses.append(float(metrics["eval_loss"]))
            self._render_plot()


def run_cpu_training(config: CPUTrainConfig) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_records = load_cpu_records(config.data_root / config.train_csv)[: config.max_train_samples]
    eval_records = load_cpu_records(config.data_root / config.eval_csv)[: config.max_eval_samples]

    train_rows = records_to_text2text(train_records)
    eval_rows = records_to_text2text(eval_records)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)

    train_tok = train_ds.map(
        lambda b: _tokenize_batch(b, tokenizer, config.max_source_length, config.max_target_length),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    eval_tok = eval_ds.map(
        lambda b: _tokenize_batch(b, tokenizer, config.max_source_length, config.max_target_length),
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    args = TrainingArguments(
        output_dir=str(config.output_dir / "checkpoints"),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=config.seed,
        remove_unused_columns=False,
    )

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    try:
        trainer_kwargs["processing_class"] = tokenizer
    except Exception:
        pass
    trainer = Trainer(**trainer_kwargs)
    metric_plot_callback = EpochMetricsPlotCallback(config.output_dir)
    trainer.add_callback(metric_plot_callback)

    stats = trainer.train()

    final_dir = config.output_dir / "model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    model_meta = {
        "version": "flan-t5-small-cpu-v1",
        "model_name": config.model_name,
        "labels": ["client_name", "client_address", "seller_name", "seller_address", "invoice_number", "invoice_date"],
        "trained_at": "local",
        "source_dataset": str(config.data_root),
    }
    _write_json(config.output_dir / "model_meta.json", model_meta)

    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "model_name": config.model_name,
        "num_train_records": len(train_records),
        "num_eval_records": len(eval_records),
        "train_metrics": getattr(stats, "metrics", {}),
        "model_dir": str(final_dir),
        "metrics_plot": str(metric_plot_callback.plot_path),
    }
    _write_json(config.output_dir / "train_summary.json", summary)
    return summary
