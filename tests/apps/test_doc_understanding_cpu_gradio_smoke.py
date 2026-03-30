from __future__ import annotations

import unittest
from pathlib import Path

from src.apps.doc_understanding_cpu_gradio import AppError, _load_model_version


class DocUnderstandingCPUGradioSmokeTest(unittest.TestCase):
    def test_load_model_version_fallback(self) -> None:
        model_dir = Path("artifacts/doc_understanding_ocr_cpu/model")
        model_dir.mkdir(parents=True, exist_ok=True)
        version = _load_model_version(model_dir, None)
        self.assertEqual(version, "model-ocr-cpu")

    def test_app_error_payload_contract(self) -> None:
        payload = AppError("INVALID_INPUT", "Aucune image fournie.").to_dict()
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "INVALID_INPUT")
        self.assertIn("extracted_text", payload)


if __name__ == "__main__":
    unittest.main()
