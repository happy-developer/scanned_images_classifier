from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.doc_understanding_cpu.data import load_cpu_records, records_to_text2text


class TestCPUDataParsing(unittest.TestCase):
    def test_parse_and_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "sample.csv"
            payload = {
                "invoice": {
                    "client_name": "Alice",
                    "client_address": "A\nB",
                    "seller_name": "Store",
                    "seller_address": "C\nD",
                    "invoice_number": "INV-1",
                    "invoice_date": "2026-03-26",
                }
            }
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["File Name", "Json Data", "OCRed Text"])
                w.writeheader()
                w.writerow({
                    "File Name": "x.jpg",
                    "Json Data": json.dumps(payload),
                    "OCRed Text": "invoice text",
                })

            records = load_cpu_records(csv_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].target_dict["client_address"], "A, B")

            ds = records_to_text2text(records)
            self.assertEqual(len(ds), 1)
            self.assertIn("OCR text", ds[0]["input_text"])
            self.assertIn("invoice_number", ds[0]["target_text"])


if __name__ == "__main__":
    unittest.main()
