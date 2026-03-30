from __future__ import annotations

import unittest
from pathlib import Path

from src.apps.doc_understanding_gradio import _extract_json_blob, _load_model_version


class DocUnderstandingGradioSmokeTest(unittest.TestCase):
    def test_extract_json_blob(self) -> None:
        txt = "answer: {\"client_name\":\"ACME\",\"invoice_number\":\"INV-001\"}"
        parsed = _extract_json_blob(txt)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["client_name"], "ACME")

    def test_load_model_version_fallback(self) -> None:
        model_dir = Path("artifacts/doc_understanding/model")
        model_dir.mkdir(parents=True, exist_ok=True)
        version = _load_model_version(model_dir, None)
        self.assertEqual(version, "model")


if __name__ == "__main__":
    unittest.main()
