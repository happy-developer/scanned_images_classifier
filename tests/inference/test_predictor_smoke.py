from __future__ import annotations

import unittest
from pathlib import Path


class PredictorSmokeTest(unittest.TestCase):
    def test_predictor_smoke(self) -> None:
        try:
            from PIL import Image
            from src.data_access.dataset_checks import validate_dataset_structure
            from src.data_access.kagglehub_resolver import resolve_kaggle_dataset_root
            from src.inference.model_loader import load_model
            from src.inference.predictor import Predictor
        except Exception as exc:
            self.skipTest(f"Dependances absentes: {exc}")
            return

        project_root = Path(__file__).resolve().parents[2]
        model_path = project_root / "artifacts" / "scanned_images_resnet18.pt"
        meta_path = project_root / "artifacts" / "model_meta.json"
        logs_path = project_root / "artifacts" / "inference_runs.jsonl"

        if not model_path.exists() or not meta_path.exists():
            self.skipTest("Artifacts model/meta incomplets pour smoke test")

        preferred_root = project_root / "data" / "kaggle_invoice_images"
        data_root = resolve_kaggle_dataset_root(preferred_root)
        dataset_context = validate_dataset_structure(data_root)

        image_path = None
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            found = sorted(Path(data_root).rglob(ext))
            if found:
                image_path = found[0]
                break
        if image_path is None:
            self.skipTest("Aucune image disponible pour smoke test")

        loaded = load_model(model_path=model_path, model_meta_path=meta_path, dataset_context=dataset_context)
        predictor = Predictor(loaded_model=loaded, dataset_context=dataset_context, logs_path=logs_path)

        with Image.open(image_path) as image:
            result = predictor.predict(image=image, threshold=0.5, image_path=str(image_path))

        for key in (
            "run_id",
            "label",
            "confidence",
            "probs",
            "model_version",
            "latency_ms",
            "dataset_context",
        ):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
