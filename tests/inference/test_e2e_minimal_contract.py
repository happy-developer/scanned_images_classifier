from __future__ import annotations

import json
import unittest
from pathlib import Path


class E2EMinimalContractTest(unittest.TestCase):
    def _find_first_image(self, root: Path) -> Path | None:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            found = sorted(root.rglob(ext))
            if found:
                return found[0]
        return None

    def test_e2e_dataset_model_predict_contract(self) -> None:
        try:
            from PIL import Image
            import torch
            from src.data_access.dataset_checks import validate_dataset_structure
            from src.data_access.kagglehub_resolver import resolve_kaggle_dataset_root
            from src.inference.model_loader import load_model
            from src.inference.predictor import Predictor
        except Exception as exc:
            self.skipTest(f"Dependances indisponibles: {exc}")
            return

        project_root = Path(__file__).resolve().parents[2]
        model_path = project_root / "artifacts" / "scanned_images_resnet18.pt"
        if not model_path.exists():
            self.skipTest("Modele artefact absent: artifacts/scanned_images_resnet18.pt")

        data_root = resolve_kaggle_dataset_root(project_root / "data" / "scanned_images_kaggle" / "dataset")
        dataset_context = validate_dataset_structure(data_root)

        image_path = self._find_first_image(data_root)
        if image_path is None:
            self.skipTest("Aucune image trouvee pour le test E2E.")

        checkpoint = torch.load(model_path, map_location="cpu")
        checkpoint_labels = []
        img_size = 224
        if isinstance(checkpoint, dict):
            raw_labels = checkpoint.get("class_names")
            if isinstance(raw_labels, list):
                checkpoint_labels = [str(x) for x in raw_labels]
            raw_config = checkpoint.get("config") or {}
            if "img_size" in raw_config:
                img_size = int(raw_config["img_size"])

        if not checkpoint_labels:
            self.skipTest("Checkpoint sans class_names, impossible de verifier le contrat minimal.")

        model_meta_path = project_root / "artifacts" / "e2e_minimal_model_meta.json"
        model_meta_payload = {
            "labels": checkpoint_labels,
            "img_size": img_size,
            "model_name": "resnet18",
            "version": "e2e-minimal",
            "trained_at": "unknown",
            "source_dataset": str(data_root),
        }
        model_meta_path.write_text(json.dumps(model_meta_payload, ensure_ascii=True, indent=2), encoding="utf-8")

        loaded = load_model(
            model_path=model_path,
            model_meta_path=model_meta_path,
            dataset_context=dataset_context,
        )

        logs_path = project_root / "artifacts" / "inference_runs.e2e_minimal.jsonl"
        predictor = Predictor(loaded_model=loaded, dataset_context=dataset_context, logs_path=logs_path)

        with Image.open(image_path) as image:
            result = predictor.predict(image=image, threshold=0.0, image_path=str(image_path))

        required_keys = {
            "run_id",
            "label",
            "confidence",
            "probs",
            "model_version",
            "latency_ms",
            "dataset_context",
        }
        self.assertTrue(required_keys.issubset(result.keys()))
        self.assertIn(result["label"], loaded.class_names)
        self.assertGreaterEqual(float(result["confidence"]), 0.0)
        self.assertLessEqual(float(result["confidence"]), 1.0)
        self.assertIn("root", result["dataset_context"])
        self.assertIn("mode", result["dataset_context"])
        self.assertIn("num_classes", result["dataset_context"])

        # Contract must stay JSON-serializable for API/logging.
        json.dumps(result, ensure_ascii=True)


if __name__ == "__main__":
    unittest.main()
