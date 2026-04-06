from __future__ import annotations

import unittest
from pathlib import Path

from src.ocr_image_text.data import OCRRecord
from src.ocr_image_text.evaluation import evaluate_records, summarize_predictions


class OCRImageTextEvaluationTest(unittest.TestCase):
    def test_supervised_metrics_include_text_summary_and_field_scores(self) -> None:
        records = [
            OCRRecord(
                img_name="a.png",
                image_path=Path("a.png"),
                ocr_text=(
                    "Invoice No: INV-001\n"
                    "Date of issue: 2026-04-03\n"
                    "Tax ID: FR123456789\n"
                    "IBAN: FR7630006000011234567890189"
                ),
            ),
            OCRRecord(
                img_name="b.png",
                image_path=Path("b.png"),
                ocr_text=(
                    "Invoice No: A-2\n"
                    "Date of issue: 03/04/2026\n"
                    "Tax ID: FR000111222\n"
                    "IBAN: BE68539007547034"
                ),
            ),
        ]
        preds = {
            "a.png": (
                "invoice number inv 001\n"
                "issue date 03/04/2026\n"
                "vat no fr123456789\n"
                "iban fr76 3000 6000 0112 3456 7890 189"
            ),
            "b.png": (
                "invoice no: A2\n"
                "date: 03-04-2026\n"
                "tax id: FR999111222\n"
                "iban: BE68 5390 0754 7034"
            ),
        }

        metrics = evaluate_records(records, preds)

        self.assertEqual(metrics["mode"], "supervised")
        self.assertEqual(metrics["num_samples"], 2)
        self.assertIn("cer", metrics)
        self.assertIn("wer", metrics)
        self.assertIn("field_exact_match_overall", metrics)
        self.assertIn("field_exact_match_by_name", metrics)
        self.assertIn("field_coverage_by_name", metrics)
        self.assertIn("non_empty_rate", metrics)
        self.assertIn("prediction_reference_length_ratio", metrics)
        self.assertIn("repetition_rate", metrics)
        self.assertGreater(metrics["cer"], 0.0)
        self.assertAlmostEqual(metrics["non_empty_rate"], 1.0)
        self.assertAlmostEqual(metrics["field_coverage_by_name"]["invoice_no"], 1.0)
        self.assertAlmostEqual(metrics["field_coverage_by_name"]["date_of_issue"], 1.0)
        self.assertAlmostEqual(metrics["field_coverage_by_name"]["tax_id"], 1.0)
        self.assertAlmostEqual(metrics["field_coverage_by_name"]["iban"], 1.0)
        self.assertAlmostEqual(metrics["field_exact_match_by_name"]["invoice_no"], 1.0)
        self.assertAlmostEqual(metrics["field_exact_match_by_name"]["date_of_issue"], 1.0)
        self.assertAlmostEqual(metrics["field_exact_match_by_name"]["tax_id"], 0.5)
        self.assertAlmostEqual(metrics["field_exact_match_by_name"]["iban"], 1.0)
        self.assertAlmostEqual(metrics["field_exact_match_overall"], 7 / 8)
        self.assertGreater(metrics["repetition_rate"], 0.0)

    def test_supervised_metrics_can_skip_wer(self) -> None:
        record_text = (
            "Invoice No: 123\n"
            "Date of issue: 2026-04-03\n"
            "Tax ID: FR123456789\n"
            "IBAN: FR7630006000011234567890189"
        )
        records = [
            OCRRecord(
                img_name="a.png",
                image_path=Path("a.png"),
                ocr_text=record_text,
            )
        ]
        preds = {"a.png": record_text}

        metrics = evaluate_records(records, preds, compute_wer=False)

        self.assertEqual(metrics["mode"], "supervised")
        self.assertNotIn("wer", metrics)
        self.assertAlmostEqual(metrics["cer"], 0.0)
        self.assertAlmostEqual(metrics["field_exact_match_overall"], 1.0)
        self.assertEqual(metrics["field_exact_match_by_name"]["iban"], 1.0)
        self.assertEqual(metrics["field_coverage_by_name"]["tax_id"], 1.0)

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
