from __future__ import annotations

import unittest
from pathlib import Path

from src.ocr_image_text.data import OCRRecord
from src.ocr_image_text.evaluation import evaluate_records, summarize_predictions


class OCRImageTextEvaluationTest(unittest.TestCase):
    def test_supervised_metrics_include_text_summary(self) -> None:
        records = [
            OCRRecord(img_name="a.png", image_path=Path("a.png"), ocr_text="ABC"),
            OCRRecord(img_name="b.png", image_path=Path("b.png"), ocr_text="Hello world"),
        ]
        preds = {
            "a.png": "abc",
            "b.png": "Hello Hello",
        }

        metrics = evaluate_records(records, preds)

        self.assertEqual(metrics["mode"], "supervised")
        self.assertEqual(metrics["num_samples"], 2)
        self.assertIn("cer", metrics)
        self.assertIn("wer", metrics)
        self.assertIn("non_empty_rate", metrics)
        self.assertIn("prediction_reference_length_ratio", metrics)
        self.assertIn("repetition_rate", metrics)
        self.assertGreater(metrics["repetition_rate"], 0.0)
        self.assertAlmostEqual(metrics["non_empty_rate"], 1.0)
        self.assertAlmostEqual(metrics["prediction_reference_length_ratio"], 1.0)

    def test_supervised_metrics_can_skip_wer(self) -> None:
        records = [OCRRecord(img_name="a.png", image_path=Path("a.png"), ocr_text="Hello")]
        preds = {"a.png": "Hello"}

        metrics = evaluate_records(records, preds, compute_wer=False)

        self.assertEqual(metrics["mode"], "supervised")
        self.assertNotIn("wer", metrics)
        self.assertAlmostEqual(metrics["cer"], 0.0)

    def test_unlabeled_summary_keeps_lightweight_metrics(self) -> None:
        metrics = summarize_predictions(["foo foo", "", "bar"])

        self.assertEqual(metrics["mode"], "unlabeled")
        self.assertEqual(metrics["num_samples"], 3)
        self.assertIn("non_empty_rate", metrics)
        self.assertIn("avg_prediction_chars", metrics)
        self.assertIn("repetition_rate", metrics)
        self.assertAlmostEqual(metrics["non_empty_rate"], 2 / 3)
        self.assertGreater(metrics["repetition_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
