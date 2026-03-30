from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class InferenceError(Exception):
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class InvalidImageError(InferenceError):
    def __init__(self, message: str = "Image invalide", details: Dict[str, Any] | None = None) -> None:
        super().__init__(code="INVALID_IMAGE", message=message, details=details or {})


class ModelNotFoundError(InferenceError):
    def __init__(self, message: str = "Modele introuvable", details: Dict[str, Any] | None = None) -> None:
        super().__init__(code="MODEL_NOT_FOUND", message=message, details=details or {})


class DatasetUnavailableError(InferenceError):
    def __init__(self, message: str = "Dataset indisponible", details: Dict[str, Any] | None = None) -> None:
        super().__init__(code="DATASET_UNAVAILABLE", message=message, details=details or {})


class InferenceExecutionError(InferenceError):
    def __init__(self, message: str = "Echec inference", details: Dict[str, Any] | None = None) -> None:
        super().__init__(code="INFERENCE_FAILED", message=message, details=details or {})
