from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    TrainerCallback,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

from .config import TrainConfig
from .data import OCRRecord, load_ocr_csv, load_multi_ocr_sources
from .evaluation import _levenshtein, repetition_rate
from .formatting import normalize_text


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _set_processor_resize(processor: Any, image_size: int, model: Any | None = None) -> int:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return int(max(32, image_size))

    side = int(max(32, image_size))
    supported = None
    if model is not None:
        enc_cfg = getattr(getattr(model, "config", None), "encoder", None)
        if enc_cfg is not None and getattr(enc_cfg, "image_size", None) is not None:
            supported = int(enc_cfg.image_size)
        elif getattr(getattr(model, "encoder", None), "config", None) is not None:
            mcfg = model.encoder.config
            if getattr(mcfg, "image_size", None) is not None:
                supported = int(mcfg.image_size)
    if supported is not None:
        side = min(side, supported)

    image_processor.do_resize = True
    image_processor.size = {"height": side, "width": side}
    return side


def _preprocess_image(image_path: Path, use_grayscale: bool) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if use_grayscale:
        # Convert to grayscale then back to 3 channels for TrOCR compatibility.
        image = image.convert("L").convert("RGB")
    return image


def _resolve_eval_records(config: TrainConfig) -> tuple[list[OCRRecord], bool, str]:
    can_fallback_to_unlabeled = bool(config.allow_unlabeled_eval) or not bool(config.require_supervised_eval)
    if not config.eval_csv:
        if can_fallback_to_unlabeled:
            return [], False, "unlabeled"
        raise FileNotFoundError(
            "Supervised evaluation is required but --eval-csv was not provided. "
            "Provide a real labeled eval CSV or pass --allow-unlabeled-eval explicitly."
        )

    eval_csv_path = config.data_root / config.eval_csv
    eval_img_dir = config.data_root / config.image_subdir_eval
    if not eval_csv_path.exists() or not eval_img_dir.exists():
        if can_fallback_to_unlabeled:
            return [], False, "unlabeled"
        missing_parts = []
        if not eval_csv_path.exists():
            missing_parts.append(f"CSV missing: {eval_csv_path}")
        if not eval_img_dir.exists():
            missing_parts.append(f"image dir missing: {eval_img_dir}")
        raise FileNotFoundError(
            "Supervised evaluation is required but the validation set is incomplete. "
            + "; ".join(missing_parts)
            + ". Provide a real labeled eval CSV or pass --allow-unlabeled-eval explicitly."
        )

    eval_records = load_ocr_csv(eval_csv_path, eval_img_dir)
    return eval_records, True, "supervised"


class OCRTrainingPlotCallback(TrainerCallback):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.train_epochs: list[float] = []
        self.train_losses: list[float] = []
        self.eval_epochs: list[float] = []
        self.eval_losses: list[float] = []
        self.eval_cer: list[float] = []
        self.eval_steps: list[int] = []
        self.plot_path = self.output_dir / "training_metrics.png"
        self.history_path = self.output_dir / "metrics_history.json"

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
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        if self.train_losses:
            ax1.plot(self.train_epochs, self.train_losses, marker="o", label="train_loss", color="#1f77b4")
        if self.eval_losses:
            ax1.plot(self.eval_epochs, self.eval_losses, marker="o", label="eval_loss", color="#d62728")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(alpha=0.25)

        if self.eval_cer:
            ax2 = ax1.twinx()
            ax2.plot(self.eval_epochs, self.eval_cer, marker="x", linestyle="--", label="eval_cer", color="#2ca02c")
            ax2.set_ylabel("CER")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        else:
            ax1.legend(loc="upper right")

        ax1.set_title("OCR Training Metrics by Epoch")
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)

        try:
            from IPython.display import Image, clear_output, display

            clear_output(wait=True)
            display(Image(filename=str(self.plot_path)))
        except Exception:
            pass

    def _write_history(self) -> None:
        payload = {
            "train": [{"epoch": e, "loss": l} for e, l in zip(self.train_epochs, self.train_losses)],
            "eval": [
                {"epoch": e, "loss": l, "cer": c, "step": s}
                for e, l, c, s in zip(
                    self.eval_epochs,
                    self.eval_losses,
                    self.eval_cer + [0.0] * max(0, len(self.eval_losses) - len(self.eval_cer)),
                    self.eval_steps + [0] * max(0, len(self.eval_losses) - len(self.eval_steps)),
                )
            ],
        }
        _write_json(self.history_path, payload)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            epoch_value = logs.get("epoch", state.epoch)
            if epoch_value is None:
                epoch_value = float(len(self.train_epochs) + 1)
            self.train_epochs.append(float(epoch_value))
            self.train_losses.append(float(logs["loss"]))
            self._write_history()
            self._render_plot()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        if "eval_loss" in metrics:
            epoch_value = metrics.get("epoch", state.epoch)
            if epoch_value is None:
                epoch_value = float(len(self.eval_epochs) + 1)
            self.eval_epochs.append(float(epoch_value))
            self.eval_losses.append(float(metrics["eval_loss"]))
            self.eval_steps.append(int(getattr(state, "global_step", 0)))
            if "eval_cer" in metrics:
                self.eval_cer.append(float(metrics["eval_cer"]))
            self._write_history()
            self._render_plot()


def _select_checkpoint_with_sanity(callback: OCRTrainingPlotCallback, checkpoints_dir: Path) -> Dict[str, Any]:
    # Candidate set: epochs where eval_loss reaches a new running minimum.
    # Among those, pick minimal eval_cer (then lower loss, then earlier epoch).
    if not callback.eval_epochs or not callback.eval_losses or not callback.eval_cer:
        return {"strategy": "none", "reason": "missing_eval_metrics"}

    rows: list[dict[str, Any]] = []
    running_best_loss = float("inf")
    for idx in range(min(len(callback.eval_epochs), len(callback.eval_losses), len(callback.eval_cer))):
        epoch = float(callback.eval_epochs[idx])
        loss = float(callback.eval_losses[idx])
        cer = float(callback.eval_cer[idx])
        step = int(callback.eval_steps[idx]) if idx < len(callback.eval_steps) else 0
        is_running_min = loss <= running_best_loss + 1e-12
        if is_running_min:
            running_best_loss = loss
            rows.append({"epoch": epoch, "loss": loss, "cer": cer, "step": step})

    pool = rows if rows else [
        {
            "epoch": float(callback.eval_epochs[idx]),
            "loss": float(callback.eval_losses[idx]),
            "cer": float(callback.eval_cer[idx]),
            "step": int(callback.eval_steps[idx]) if idx < len(callback.eval_steps) else 0,
        }
        for idx in range(min(len(callback.eval_epochs), len(callback.eval_losses), len(callback.eval_cer)))
    ]

    chosen = sorted(pool, key=lambda r: (r["cer"], r["loss"], r["epoch"]))[0]
    ckpt = checkpoints_dir / f"checkpoint-{int(chosen['step'])}"
    if not ckpt.exists():
        existing = sorted(checkpoints_dir.glob("checkpoint-*"))
        if existing:
            ckpt = existing[-1]
        else:
            return {"strategy": "none", "reason": "checkpoint_not_found", "selected": chosen}

    return {
        "strategy": "running_min_eval_loss_then_min_eval_cer",
        "selected": chosen,
        "checkpoint_path": str(ckpt.resolve()),
    }


class OCRDataset(Dataset):
    def __init__(self, records: List[OCRRecord], processor: Any, max_target_length: int, use_grayscale: bool):
        self.records = records
        self.processor = processor
        self.max_target_length = max_target_length
        self.use_grayscale = use_grayscale

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        image = _preprocess_image(rec.image_path, use_grayscale=self.use_grayscale)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            normalize_text(rec.ocr_text),
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def _compute_cer_metrics(eval_preds, tokenizer) -> Dict[str, float]:
    pred_ids, label_ids = eval_preds
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    pred_ids = np.asarray(pred_ids)
    label_ids = np.asarray(label_ids)

    # Some backends may surface negative sentinels in predictions during eval;
    # sanitize them before decoding to avoid tokenizer overflow errors.
    pred_ids = pred_ids.astype(np.int64, copy=False)
    pred_ids[pred_ids < 0] = pad_id

    label_ids = label_ids.astype(np.int64, copy=False)
    label_ids[label_ids == -100] = pad_id
    label_ids[label_ids < 0] = pad_id

    decoded_preds = tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids.tolist(), skip_special_tokens=True)

    total_edits = 0
    total_chars = 0
    total_word_edits = 0
    total_words = 0
    exact = 0
    non_empty = 0
    pred_chars = 0
    ref_chars = 0
    rep_rate_sum = 0.0
    for pred, label in zip(decoded_preds, decoded_labels):
        p = normalize_text(pred)
        t = normalize_text(label)
        total_edits += _levenshtein(p, t)
        total_chars += max(len(t), 1)
        p_words = p.split()
        t_words = t.split()
        total_word_edits += _levenshtein(p_words, t_words)
        total_words += max(len(t_words), 1)
        exact += int(p == t)
        non_empty += int(bool(p.strip()))
        pred_chars += len(p)
        ref_chars += len(t)
        rep_rate_sum += repetition_rate(p)

    n = max(len(decoded_labels), 1)
    cer = float(total_edits) / float(max(total_chars, 1))
    wer = float(total_word_edits) / float(max(total_words, 1))
    return {
        "cer": cer,
        "wer": wer,
        "exact_match": float(exact) / float(n),
        "non_empty_rate": float(non_empty) / float(n),
        "prediction_reference_length_ratio": float(pred_chars) / float(max(ref_chars, 1)),
        "repetition_rate": float(rep_rate_sum) / float(n),
    }


def run_training(config: TrainConfig) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if int(config.train_epochs) > 10:
        raise ValueError(
            "Refusing to start training with epochs > 10. "
            "Keep runs <= 10 epochs and rely on early stopping on eval_cer."
        )

    eval_records, supervised_eval_enabled, eval_mode = _resolve_eval_records(config)
    train_records = load_multi_ocr_sources(
        data_root=config.data_root,
        csv_paths=config.train_csvs,
        image_subdirs=config.image_subdirs_train,
    )
    if config.max_train_samples > 0:
        train_records = train_records[: config.max_train_samples]
    if supervised_eval_enabled and config.max_eval_samples > 0:
        eval_records = eval_records[: config.max_eval_samples]

    processor = TrOCRProcessor.from_pretrained(config.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(config.model_name)
    effective_image_size = _set_processor_resize(processor, config.image_size, model=model)

    tok = processor.tokenizer
    start_id = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
    if start_id is None:
        start_id = 0
    setattr(model.config, "pad_token_id", tok.pad_token_id)
    setattr(model.config, "eos_token_id", tok.eos_token_id)
    setattr(model.config, "decoder_start_token_id", int(start_id))
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id
        model.generation_config.decoder_start_token_id = int(start_id)
        model.generation_config.num_beams = max(1, int(config.generation_num_beams))
        model.generation_config.length_penalty = float(config.generation_length_penalty)
        model.generation_config.no_repeat_ngram_size = max(0, int(config.generation_no_repeat_ngram_size))
        model.generation_config.repetition_penalty = float(config.generation_repetition_penalty)

    train_ds = OCRDataset(train_records, processor, config.max_target_length, use_grayscale=bool(config.use_grayscale))
    eval_ds = None
    if supervised_eval_enabled:
        eval_ds = OCRDataset(eval_records, processor, config.max_target_length, use_grayscale=bool(config.use_grayscale))

    args = Seq2SeqTrainingArguments(
        output_dir=str(config.output_dir / "checkpoints"),
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        num_train_epochs=config.train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=config.max_target_length,
        generation_num_beams=max(1, int(config.generation_num_beams)),
        logging_strategy="epoch",
        eval_strategy="epoch" if supervised_eval_enabled else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=bool(supervised_eval_enabled),
        metric_for_best_model=config.metric_for_best_model if supervised_eval_enabled else None,
        greater_is_better=config.greater_is_better if supervised_eval_enabled else None,
        report_to="none",
        seed=config.random_seed,
        data_seed=config.random_seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if supervised_eval_enabled else None,
        data_collator=default_data_collator,
        compute_metrics=(lambda p: _compute_cer_metrics(p, tok)) if supervised_eval_enabled else None,
    )
    metric_callback = OCRTrainingPlotCallback(config.output_dir)
    trainer.add_callback(metric_callback)
    if supervised_eval_enabled:
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=max(1, int(config.early_stopping_patience)),
                early_stopping_threshold=float(config.early_stopping_threshold),
            )
        )

    stats = trainer.train()

    selected_checkpoint_info: Dict[str, Any] | None = None
    if supervised_eval_enabled:
        selected_checkpoint_info = _select_checkpoint_with_sanity(metric_callback, config.output_dir / "checkpoints")
        selected_path = selected_checkpoint_info.get("checkpoint_path") if selected_checkpoint_info else None
        if selected_path and Path(selected_path).exists():
            model = VisionEncoderDecoderModel.from_pretrained(selected_path)
            setattr(model.config, "pad_token_id", tok.pad_token_id)
            setattr(model.config, "eos_token_id", tok.eos_token_id)
            setattr(model.config, "decoder_start_token_id", int(start_id))
            if hasattr(model, "generation_config"):
                model.generation_config.pad_token_id = tok.pad_token_id
                model.generation_config.eos_token_id = tok.eos_token_id
                model.generation_config.decoder_start_token_id = int(start_id)
                model.generation_config.num_beams = max(1, int(config.generation_num_beams))
                model.generation_config.length_penalty = float(config.generation_length_penalty)
                model.generation_config.no_repeat_ngram_size = max(0, int(config.generation_no_repeat_ngram_size))
                model.generation_config.repetition_penalty = float(config.generation_repetition_penalty)

    final_dir = config.output_dir / "model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    meta = {
        "version": f"{config.model_name.replace('/', '-')}-ocr-cpu-v2",
        "model_name": config.model_name,
        "task": "ocr_image_to_text",
        "source_dataset": str(config.data_root),
        "train_csvs": list(config.train_csvs),
        "eval_csv": config.eval_csv,
    }
    _write_json(config.output_dir / "model_meta.json", meta)

    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "effective_image_size": int(effective_image_size),
        "supervised_eval_enabled": supervised_eval_enabled,
        "eval_mode": eval_mode,
        "checkpoint_selection": selected_checkpoint_info,
        "num_train_records": len(train_records),
        "num_eval_records": len(eval_records),
        "train_metrics": getattr(stats, "metrics", {}),
        "model_dir": str(final_dir),
        "metrics_plot": str(metric_callback.plot_path),
        "metrics_history": str(metric_callback.history_path),
    }
    _write_json(config.output_dir / "train_summary.json", summary)
    return summary

