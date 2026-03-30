from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.doc_understanding.data import load_invoice_csv
from src.doc_understanding.formatting import build_target_invoice


class TestDocUnderstandingDataFormatting(unittest.TestCase):
    def test_load_and_format_invoice_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            img_path = image_dir / "invoice_001.jpg"
            Image.new("RGB", (64, 64), color=(255, 255, 255)).save(img_path)

            csv_path = root / "sample.csv"
            invoice_payload = {
                "invoice": {
                    "client_name": "Alice",
                    "client_address": "Street 1\nCity",
                    "seller_name": "Shop",
                    "seller_address": "Road 5\nTown",
                    "invoice_number": "INV-001",
                    "invoice_date": "2026-03-26",
                }
            }

            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["File Name", "Json Data", "OCRed Text"])
                writer.writeheader()
                writer.writerow(
                    {
                        "File Name": "invoice_001.jpg",
                        "Json Data": json.dumps(invoice_payload),
                        "OCRed Text": "dummy text",
                    }
                )

            records = load_invoice_csv(csv_path, image_dir)
            self.assertEqual(len(records), 1)
            rec = records[0]
            target = build_target_invoice(rec.invoice_data)

            self.assertEqual(target.client_name, "Alice")
            self.assertEqual(target.client_address, "Street 1, City")
            self.assertEqual(target.seller_address, "Road 5, Town")
            self.assertEqual(target.invoice_number, "INV-001")


if __name__ == "__main__":
    unittest.main()
