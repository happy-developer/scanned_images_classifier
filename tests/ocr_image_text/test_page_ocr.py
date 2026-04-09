from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image, ImageDraw

from src.ocr_image_text.page_ocr import (
    CropRegion,
    SegmentationPlan,
    _dedupe_crop_regions,
    _dedupe_neighboring_text_segments,
    _run_crop_first_ocr,
    segment_page,
)


class _FakeProcessor:
    def __init__(self, decoded_batches: list[list[str]]) -> None:
        self.tokenizer = SimpleNamespace(bos_token_id=None, pad_token_id=None)
        self._decoded_batches = list(decoded_batches)

    def __call__(self, images, return_tensors):
        return SimpleNamespace(pixel_values=torch.zeros((len(images), 3, 4, 4)))

    def batch_decode(self, output_ids, skip_special_tokens=True):
        if not self._decoded_batches:
            raise AssertionError('batch_decode called more times than expected')
        return self._decoded_batches.pop(0)


class _FakeModel:
    def generate(self, pixel_values, **generation_kwargs):
        return object()


class OCRPageDeduplicationTest(unittest.TestCase):
    def test_line_only_mode_selects_full_width_line_crops(self) -> None:
        image = Image.new('RGB', (240, 120), 'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 16, 220, 30), fill='black')
        draw.rectangle((18, 72, 214, 88), fill='black')

        plan = segment_page(image, segmentation_mode='line_only')

        self.assertEqual(plan.strategy, 'line_only_crops')
        self.assertGreaterEqual(len(plan.crop_regions), 2)
        self.assertTrue(all(region.label == 'line' for region in plan.crop_regions))
        self.assertTrue(all(region.box[0] == 0 and region.box[2] == image.width for region in plan.crop_regions))

    def test_dedupes_neighboring_text_segments_without_reordering(self) -> None:
        segments = [
            'Section 1 - Intro',
            'Section 1 Intro',
            'Billing address',
            'Delivery address',
        ]

        deduped, skipped = _dedupe_neighboring_text_segments(segments)

        self.assertEqual(deduped, ['Section 1 - Intro', 'Billing address', 'Delivery address'])
        self.assertEqual(skipped, 1)

    def test_keeps_distinct_text_even_when_it_is_related(self) -> None:
        segments = [
            'Invoice date',
            'Delivery date',
            'Invoice number',
        ]

        deduped, skipped = _dedupe_neighboring_text_segments(segments)

        self.assertEqual(deduped, ['Invoice date', 'Delivery date', 'Invoice number'])
        self.assertEqual(skipped, 0)

    def test_dedupes_overlapping_crop_regions_but_keeps_separate_boxes(self) -> None:
        regions = [
            CropRegion(box=(0, 0, 100, 100), label='line'),
            CropRegion(box=(1, 1, 101, 101), label='line'),
            CropRegion(box=(120, 0, 220, 100), label='line'),
            CropRegion(box=(140, 0, 240, 100), label='line'),
        ]

        deduped, skipped = _dedupe_crop_regions(regions)

        self.assertEqual(deduped, [regions[0], regions[2], regions[3]])
        self.assertEqual(skipped, 1)

    def test_run_crop_first_ocr_truncates_overlong_segments(self) -> None:
        image = Image.new('RGB', (256, 128), 'white')
        plan = SegmentationPlan(
            crop_regions=[
                CropRegion(box=(0, 0, 256, 128), label='line'),
            ],
            used_full_page_fallback=False,
            strategy='line_only_crops',
            fallback_reason=None,
            original_crop_count=1,
            deduplicated_crop_count=1,
            duplicate_crop_count=0,
        )
        processor = _FakeProcessor([['ABCDEFGHIJKLMNO']])

        with patch('src.ocr_image_text.page_ocr.segment_page', return_value=plan):
            result = _run_crop_first_ocr(
                _FakeModel(),
                processor,
                image,
                segmentation_mode='line_only',
                max_new_tokens=16,
                num_beams=1,
                temperature=0.0,
                length_penalty=1.0,
                no_repeat_ngram_size=0,
                repetition_penalty=1.0,
                max_chars_per_segment=10,
                max_total_chars=100,
                max_invoice_markers_per_page=2,
                hard_truncate_segment_text=True,
                batch_size=4,
            )

        self.assertEqual(result['prediction'], 'ABCDEFGHIJ')
        self.assertEqual(result['normalized_output'], 'ABCDEFGHIJ')
        self.assertEqual(result['segment_truncation_count'], 1)
        self.assertFalse(result['used_full_page_fallback'])

    def test_run_crop_first_ocr_caps_total_chars_without_forcing_full_page_fallback(self) -> None:
        image = Image.new('RGB', (256, 128), 'white')
        plan = SegmentationPlan(
            crop_regions=[
                CropRegion(box=(0, 0, 256, 128), label='line'),
            ],
            used_full_page_fallback=False,
            strategy='line_only_crops',
            fallback_reason=None,
            original_crop_count=1,
            deduplicated_crop_count=1,
            duplicate_crop_count=0,
        )
        processor = _FakeProcessor([['ABCDEFGHIJ']])

        with patch('src.ocr_image_text.page_ocr.segment_page', return_value=plan):
            result = _run_crop_first_ocr(
                _FakeModel(),
                processor,
                image,
                segmentation_mode='line_only',
                max_new_tokens=16,
                num_beams=1,
                temperature=0.0,
                length_penalty=1.0,
                no_repeat_ngram_size=0,
                repetition_penalty=1.0,
                max_chars_per_segment=100,
                max_total_chars=5,
                max_invoice_markers_per_page=2,
                hard_truncate_segment_text=True,
                batch_size=4,
            )

        self.assertEqual(result['segmentation_strategy'], 'line_only_crops')
        self.assertFalse(result['used_full_page_fallback'])
        self.assertIsNone(result['fallback_reason'])
        self.assertEqual(result['prediction'], 'ABCDE')
        self.assertLessEqual(len(result['prediction']), 5)
        self.assertTrue(result['guardrail_char_cap_applied'])

    def test_run_crop_first_ocr_caps_invoice_markers_without_fallback(self) -> None:
        image = Image.new('RGB', (256, 128), 'white')
        plan = SegmentationPlan(
            crop_regions=[CropRegion(box=(0, 0, 256, 128), label='line')],
            used_full_page_fallback=False,
            strategy='line_only_crops',
            fallback_reason=None,
            original_crop_count=1,
            deduplicated_crop_count=1,
            duplicate_crop_count=0,
        )
        processor = _FakeProcessor([
            ['Invoice No: 1 Invoice No: 2 Invoice No: 3 Invoice No: 4 details']
        ])

        with patch('src.ocr_image_text.page_ocr.segment_page', return_value=plan):
            result = _run_crop_first_ocr(
                _FakeModel(),
                processor,
                image,
                max_new_tokens=16,
                num_beams=1,
                temperature=0.0,
                length_penalty=1.0,
                no_repeat_ngram_size=0,
                repetition_penalty=1.0,
                max_invoice_markers_per_page=2,
                batch_size=4,
            )

        self.assertFalse(result['used_full_page_fallback'])
        self.assertTrue(result['guardrail_marker_cap_applied'])
        self.assertLessEqual(result['invoice_marker_count'], 2)

    def test_run_crop_first_ocr_reports_dedup_metrics(self) -> None:
        image = Image.new('RGB', (256, 128), 'white')
        plan = SegmentationPlan(
            crop_regions=[
                CropRegion(box=(0, 0, 128, 128), label='line'),
                CropRegion(box=(128, 0, 256, 128), label='line'),
                CropRegion(box=(0, 0, 128, 128), label='line'),
            ],
            used_full_page_fallback=False,
            strategy='line_block_crops',
            fallback_reason=None,
            original_crop_count=3,
            deduplicated_crop_count=2,
            duplicate_crop_count=1,
        )
        processor = _FakeProcessor([['Hello world', 'Hello  world', 'Invoice 123']])

        with patch('src.ocr_image_text.page_ocr.segment_page', return_value=plan):
            result = _run_crop_first_ocr(
                _FakeModel(),
                processor,
                image,
                max_new_tokens=16,
                num_beams=1,
                temperature=0.0,
                length_penalty=1.0,
                no_repeat_ngram_size=0,
                repetition_penalty=1.0,
                batch_size=4,
            )

        self.assertEqual(result['raw_output'], 'Hello world\nInvoice 123')
        self.assertEqual(result['normalized_output'], 'Hello world\nInvoice 123')
        self.assertEqual(result['crop_count'], 3)
        self.assertEqual(result['original_crop_count'], 3)
        self.assertEqual(result['deduplicated_crop_count'], 2)
        self.assertEqual(result['duplicate_crop_count'], 1)
        self.assertEqual(result['segment_count'], 3)
        self.assertEqual(result['deduplicated_segment_count'], 2)
        self.assertEqual(result['duplicate_segment_count'], 1)
        self.assertFalse(result['used_full_page_fallback'])


if __name__ == '__main__':
    unittest.main()
