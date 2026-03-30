from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.doc_understanding_cpu.data import load_cpu_records, records_to_text2text


class TestDataFormattingCPU(unittest.TestCase):
    def test_load_and_format_cpu_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "sample.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["File Name", "Json Data", "OCRed Text"])
                writer.writerow([
                    "img-1.jpg",
                    json.dumps({"invoice": {
                        "client_name": "ACME",
                        "client_address": "X",
                        "seller_name": "S",
                        "seller_address": "Y",
                        "invoice_number": "INV-1",
                        "invoice_date": "2026-01-01",
                    }}, ensure_ascii=False),
                    "OCR sample",
                ])

            records = load_cpu_records(csv_path)
            self.assertEqual(len(records), 1)
            ds = records_to_text2text(records)
            self.assertEqual(len(ds), 1)
            self.assertIn("OCR sample", ds[0]["input_text"])
            parsed = json.loads(ds[0]["target_text"])
            self.assertEqual(parsed["invoice_number"], "INV-1")


if __name__ == "__main__":
    unittest.main()
