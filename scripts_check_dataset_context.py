from __future__ import annotations

import json
from pathlib import Path

from src.data_access.dataset_checks import validate_dataset_structure
from src.data_access.kagglehub_resolver import resolve_kaggle_dataset_root


def main() -> None:
    root = resolve_kaggle_dataset_root(Path(__file__).resolve().parents[0] / "data" / "kaggle_invoice_images")
    ctx = validate_dataset_structure(root)
    print(json.dumps(ctx.to_dict(), ensure_ascii=True))


if __name__ == "__main__":
    main()
