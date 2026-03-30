from __future__ import annotations

import csv
import unittest
from pathlib import Path

from src.inference.errors import InferenceExecutionError


class InferenceSmokeTest(unittest.TestCase):
    def _find_first_image_from_csv(self, data_root: Path) -> Path | None:
        csv_paths = sorted(data_root.rglob('*.csv'))
        for csv_path in csv_paths:
            with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                image_col = None
                for name in reader.fieldnames:
                    token = name.strip().lower()
                    if token in {'image_path', 'img_path', 'path', 'filepath', 'file_path', 'filename', 'file_name', 'image', 'img'}:
                        image_col = name
                        break
                if image_col is None:
                    continue
                for row in reader:
                    value = str(row.get(image_col, '')).strip()
                    if not value:
                        continue
                    ref = Path(value)
                    candidates = [
                        (csv_path.parent / ref).resolve(),
                        (csv_path.parent / csv_path.stem / ref).resolve(),
                        (csv_path.parent.parent / ref).resolve(),
                    ]
                    for candidate in candidates:
                        if candidate.exists() and candidate.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}:
                            return candidate
        return None

    def test_predict_smoke_kaggle(self) -> None:
        try:
            from PIL import Image
            from src.inference.service import load_model, predict_image
        except Exception as exc:
            self.skipTest(f"Dependances inference indisponibles: {exc}")
            return

        project_root = Path(__file__).resolve().parents[2]
        model_path = project_root / 'artifacts' / 'kaggle_faithful' / 'scanned_images_best_model.pth'
        data_root = project_root / 'data' / 'kaggle_invoice_images'
        logs_path = project_root / 'artifacts' / 'inference_runs.jsonl'

        if not model_path.exists():
            self.skipTest('Modele Kaggle absent pour smoke test.')
        if not data_root.exists():
            self.skipTest('Dataset Kaggle absent pour smoke test.')

        image_path = self._find_first_image_from_csv(data_root)
        if image_path is None:
            self.skipTest('Aucune image exploitable trouvee via CSV Kaggle.')

        try:
            loaded = load_model(model_path=model_path, data_root=data_root, strict_class_check=True)
        except InferenceExecutionError as exc:
            if 'timm' in str(exc).lower():
                self.skipTest(f"Dependance manquante pour checkpoint Kaggle: {exc}")
            raise

        with Image.open(image_path) as image:
            result = predict_image(loaded=loaded, image=image, run_id='smoke-test-kaggle', logs_path=logs_path)

        self.assertIn(result.label, loaded.class_names)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


if __name__ == '__main__':
    unittest.main()
