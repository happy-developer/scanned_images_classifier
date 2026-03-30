from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_COL_CANDIDATES = [
    "image_path",
    "img_path",
    "path",
    "filepath",
    "file_path",
    "filename",
    "file_name",
    "image",
    "img",
]
LABEL_COL_CANDIDATES = [
    "label",
    "class",
    "class_name",
    "category",
    "document_type",
    "type",
    "target",
]


@dataclass
class DatasetContext:
    root: str
    mode: str
    num_classes: int
    class_names: List[str]
    num_images: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _count_images_under(path: Path) -> int:
    count = 0
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        count += len(list(path.rglob(ext)))
    return count


def _canonical(value: str) -> str:
    return value.strip().lower()


def _find_column(fieldnames: List[str], candidates: List[str]) -> str | None:
    by_canonical = {_canonical(name): name for name in fieldnames}
    for candidate in candidates:
        found = by_canonical.get(_canonical(candidate))
        if found:
            return found
    return None


def _resolve_image_path(csv_path: Path, image_ref: str) -> Path | None:
    ref = Path(image_ref)
    candidates = []
    if ref.is_absolute():
        candidates.append(ref)
    else:
        candidates.append((csv_path.parent / ref).resolve())
        candidates.append((csv_path.parent / csv_path.stem / ref).resolve())
        candidates.append((csv_path.parent.parent / ref).resolve())

    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() in IMAGE_EXTS:
            return candidate
    return None


def _labels_from_batch_csv(root: Path) -> List[str]:
    labels = set()
    csv_paths = sorted(root.rglob("*.csv"))
    if not csv_paths:
        raise ValueError("Dataset batch+csv detecte mais aucun CSV n'a ete trouve.")

    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue

            label_col = _find_column(reader.fieldnames, LABEL_COL_CANDIDATES)
            image_col = _find_column(reader.fieldnames, IMAGE_COL_CANDIDATES)

            if image_col is None:
                for fn in reader.fieldnames:
                    token = _canonical(fn)
                    if "image" in token or "file" in token or "path" in token:
                        image_col = fn
                        break
            if label_col is None:
                for fn in reader.fieldnames:
                    token = _canonical(fn)
                    if fn != image_col and ("label" in token or "class" in token):
                        label_col = fn
                        break

            for row in reader:
                raw_label = str(row.get(label_col, "")).strip() if label_col else ""
                raw_image_ref = str(row.get(image_col, "")).strip() if image_col else ""

                if raw_label:
                    labels.add(raw_label)
                    continue

                if raw_image_ref:
                    image_path = _resolve_image_path(csv_path, raw_image_ref)
                    if image_path is not None:
                        labels.add(image_path.parent.name)

    if labels:
        return sorted(labels)

    return ["document"]


def validate_dataset_structure(root: Path) -> DatasetContext:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root absent: {root}")

    batch_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("batch_")])
    if batch_dirs:
        class_names = _labels_from_batch_csv(root)
        num_images = _count_images_under(root)
        if num_images <= 0:
            raise ValueError(f"Aucune image trouvee dans dataset batch: {root}")
        return DatasetContext(
            root=str(root),
            mode="batch_csv",
            num_classes=max(1, len(class_names)),
            class_names=class_names if class_names else ["document"],
            num_images=num_images,
        )

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"Aucun dossier classe trouve sous: {root}")

    class_names = [d.name for d in class_dirs]
    num_images = sum(_count_images_under(d) for d in class_dirs)
    if num_images <= 0:
        raise ValueError(f"Aucune image detectee sous: {root}")

    return DatasetContext(
        root=str(root),
        mode="class_folders",
        num_classes=len(class_names),
        class_names=class_names,
        num_images=num_images,
    )
