from __future__ import annotations

import os
from pathlib import Path


def resolve_kaggle_dataset_root(explicit_path: str | Path | None = None) -> Path:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    env_path = os.environ.get("SCANNED_IMAGES_DATASET_ROOT", "").strip()
    if env_path:
        candidates.append(Path(env_path))

    project_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            project_root / "data" / "kaggle_invoice_images",
            project_root / "data" / "scanned_images_kaggle" / "dataset",
            project_root / "data" / "scanned_images" / "dataset",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Dataset Kaggle introuvable. Definir SCANNED_IMAGES_DATASET_ROOT ou fournir --data-root."
    )
