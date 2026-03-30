from __future__ import annotations

import unittest

from src.inference.errors import InvalidImageError


class ContractIOTest(unittest.TestCase):
    def test_error_contract_code(self) -> None:
        exc = InvalidImageError(message="Image invalide", details={"field": "image"})
        payload = exc.to_dict()
        self.assertEqual(payload["error"]["code"], "INVALID_IMAGE")
        self.assertIn("message", payload["error"])
        self.assertIn("details", payload["error"])


if __name__ == "__main__":
    unittest.main()
