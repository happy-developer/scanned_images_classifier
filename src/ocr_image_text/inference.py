from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .config import InferConfig
from .page_ocr import _run_crop_first_ocr


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
        image = image.convert("L").convert("RGB")
    return image


@dataclass
class Predictor:
    model: Any
    processor: Any
    infer_config: InferConfig
    effective_image_size: int

    def predict(
        self,
        image_path: Path,
        *,
        segmentation_mode: str | None = None,
        max_new_tokens: int | None = None,
        num_beams: int | None = None,
        temperature: float | None = None,
        length_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        repetition_penalty: float | None = None,
        max_chars_per_segment: int | None = None,
        max_total_chars: int | None = None,
        max_invoice_markers_per_page: int | None = None,
        hard_truncate_segment_text: bool | None = None,
        max_crops: int | None = None,
        crop_batch_size: int | None = None,
    ) -> Dict[str, Any]:
        image = _preprocess_image(image_path, use_grayscale=bool(self.infer_config.use_grayscale))
        result = _run_crop_first_ocr(
            self.model,
            self.processor,
            image,
            segmentation_mode=self.infer_config.segmentation_mode if segmentation_mode is None else str(segmentation_mode),
            max_new_tokens=self.infer_config.max_new_tokens if max_new_tokens is None else int(max_new_tokens),
            num_beams=self.infer_config.num_beams if num_beams is None else int(num_beams),
            temperature=self.infer_config.temperature if temperature is None else float(temperature),
            length_penalty=self.infer_config.length_penalty if length_penalty is None else float(length_penalty),
            no_repeat_ngram_size=(
                self.infer_config.no_repeat_ngram_size
                if no_repeat_ngram_size is None
                else int(no_repeat_ngram_size)
            ),
            repetition_penalty=(
                self.infer_config.repetition_penalty
                if repetition_penalty is None
                else float(repetition_penalty)
            ),
            max_chars_per_segment=(
                self.infer_config.max_chars_per_segment
                if max_chars_per_segment is None
                else int(max_chars_per_segment)
            ),
            max_total_chars=(
                self.infer_config.max_total_chars if max_total_chars is None else int(max_total_chars)
            ),
            max_invoice_markers_per_page=(
                self.infer_config.max_invoice_markers_per_page
                if max_invoice_markers_per_page is None
                else int(max_invoice_markers_per_page)
            ),
            hard_truncate_segment_text=(
                self.infer_config.hard_truncate_segment_text
                if hard_truncate_segment_text is None
                else bool(hard_truncate_segment_text)
            ),
            max_crops=(
                self.infer_config.max_crops if max_crops is None else int(max_crops)
            ),
            batch_size=(
                self.infer_config.crop_batch_size if crop_batch_size is None else int(crop_batch_size)
            ),
        )
        result["effective_image_size"] = int(self.effective_image_size)
        return result


def load_predictor(config: InferConfig) -> Predictor:
    model_dir = config.artifacts_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"No trained OCR model found in {model_dir}")

    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    effective_image_size = _set_processor_resize(processor, config.image_size, model=model)
    tok = processor.tokenizer
    start_id = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
    if start_id is None:
        start_id = 0
    model.config.decoder_start_token_id = int(start_id)
    model.config.eos_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.decoder_start_token_id = int(start_id)
        model.generation_config.eos_token_id = tok.eos_token_id
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.num_beams = max(1, int(config.num_beams))
        model.generation_config.length_penalty = float(config.length_penalty)
        model.generation_config.no_repeat_ngram_size = max(0, int(config.no_repeat_ngram_size))
        model.generation_config.repetition_penalty = max(1.0, float(config.repetition_penalty))
    model.to("cpu")
    model.eval()
    return Predictor(
        model=model,
        processor=processor,
        infer_config=config,
        effective_image_size=int(effective_image_size),
    )
