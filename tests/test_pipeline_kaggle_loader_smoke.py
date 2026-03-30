from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.scanned_images_pipeline import (
    build_dataloaders_from_records,
    load_kaggle_manifest_records,
    split_records,
)


class TestKaggleLoaderSmoke(unittest.TestCase):
    def _write_image(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color=(128, 128, 128)).save(path)

    def test_load_kaggle_like_batch_csv_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_dir = root / "batch_1"
            images_dir = batch_dir / "batch1_1"
            csv_path = batch_dir / "batch1_1.csv"

            rows = [
                ("img_a_001.png", "invoice"),
                ("img_a_002.png", "invoice"),
                ("img_b_001.png", "receipt"),
                ("img_b_002.png", "receipt"),
            ]
            for fname, _ in rows:
                self._write_image(images_dir / fname)

            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image_path", "label"])
                writer.writerows(rows)

            records = load_kaggle_manifest_records(root)
            train_records, val_records = split_records(records, val_split=0.5, seed=123)
            train_loader, val_loader, class_names = build_dataloaders_from_records(
                train_records=train_records,
                val_records=val_records,
                batch_size=2,
                img_size=32,
                num_workers=0,
            )

            self.assertEqual(sorted(class_names), ["invoice", "receipt"])
            self.assertGreaterEqual(len(train_loader.dataset), 2)
            self.assertGreaterEqual(len(val_loader.dataset), 2)


if __name__ == "__main__":
    unittest.main()
