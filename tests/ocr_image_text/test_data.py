from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.ocr_image_text.data import load_ocr_csv


class OCRDataLoadingTest(unittest.TestCase):
    def _write_csv(self, path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_load_ocr_csv_legacy_file_name_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "batch_1" / "batch_1" / "batch1_1"
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / "invoice_001.png"
            image_path.write_bytes(b"fake")

            csv_path = root / "splits" / "legacy.csv"
            self._write_csv(
                csv_path,
                rows=[{"File Name": "invoice_001.png", "OCRed Text": "Invoice 001"}],
                fieldnames=["File Name", "OCRed Text"],
            )

            records = load_ocr_csv(csv_path, image_dir)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].img_name, "invoice_001.png")
            self.assertEqual(records[0].ocr_text, "Invoice 001")
            self.assertEqual(records[0].image_path, image_path.resolve())

    def test_load_ocr_csv_supports_optional_source_path_relative_to_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            image_dir = data_root / "batch_1" / "batch_1" / "batch1_1"
            image_dir.mkdir(parents=True, exist_ok=True)

            source_relative = Path("batch_1") / "batch_1" / "batch1_2" / "invoice_002.png"
            source_image_path = data_root / source_relative
            source_image_path.parent.mkdir(parents=True, exist_ok=True)
            source_image_path.write_bytes(b"fake")

            csv_path = data_root / "splits" / "global_batch_1.csv"
            self._write_csv(
                csv_path,
                rows=[
                    {
                        "File Name": "invoice_002.png",
                        "OCRed Text": "Invoice 002",
                        "source_path": str(source_relative).replace("\\", "/"),
                    }
                ],
                fieldnames=["File Name", "OCRed Text", "source_path"],
            )

            records = load_ocr_csv(csv_path, image_dir, data_root=data_root)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].img_name, "invoice_002.png")
            self.assertEqual(records[0].ocr_text, "Invoice 002")
            self.assertEqual(records[0].image_path, source_image_path.resolve())


if __name__ == "__main__":
    unittest.main()
