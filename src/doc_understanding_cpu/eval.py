from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from .config import TARGET_FIELDS
from .data import CPURecord


def _field_exact(pred: dict, target: dict) -> dict:
    out = {}
    for f in TARGET_FIELDS:
        out[f] = int(str(pred.get(f, "")).strip() == str(target.get(f, "")).strip())
    return out


def evaluate_cpu_predictions(records: Iterable[CPURecord], predictions: Dict[str, dict]) -> dict:
    rows: List[dict] = []
    for rec in records:
        pred = predictions.get(rec.img_name, {})
        if not isinstance(pred, dict):
            pred = {}
        match = _field_exact(pred, rec.target_dict)
        rows.append({"img_name": rec.img_name, **match})

    if not rows:
        return {"num_samples": 0, "field_accuracy": {}, "overall_exact_match": 0.0}

    fields = [f for f in rows[0].keys() if f != "img_name"]
    field_accuracy = {f: sum(r[f] for r in rows) / len(rows) for f in fields}
    overall = sum(1 for r in rows if all(r[f] == 1 for f in fields)) / len(rows)
    return {
        "num_samples": len(rows),
        "field_accuracy": field_accuracy,
        "overall_exact_match": overall,
    }


def write_eval_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
