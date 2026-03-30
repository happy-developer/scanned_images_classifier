from __future__ import annotations

import unittest
from pathlib import Path

from src.data_access.dataset_checks import validate_dataset_structure
from src.scanned_images_pipeline import load_kaggle_manifest_records


class LabelsCoherenceTest(unittest.TestCase):
    def test_dataset_context_matches_pipeline_labels(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        data_root = project_root / "data" / "kaggle_invoice_images"
        if not data_root.exists():
            self.skipTest(f"Dataset absent: {data_root}")

        dataset_context = validate_dataset_structure(data_root)
        records = load_kaggle_manifest_records(data_root)
        pipeline_labels = sorted({r.label_name for r in records})

        self.assertEqual(len(dataset_context.class_names), len(pipeline_labels))
        self.assertEqual(set(dataset_context.class_names), set(pipeline_labels))


if __name__ == "__main__":
    unittest.main()
