from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class InferenceResult:
    label: str
    confidence: float
    probs: Dict[str, float]
    model_version: str
    run_id: str
    latency_ms: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
