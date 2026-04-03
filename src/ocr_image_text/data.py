from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class OCRRecord:
    img_name: str
    image_path: Path
    ocr_text: str


def load_ocr_csv(csv_path: Path, image_dir: Path) -> List[OCRRecord]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    records: List[OCRRecord] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        expected_cols = {"File Name", "OCRed Text"}
        missing = expected_cols.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"CSV missing required columns {sorted(missing)}: {csv_path}")
        for row in reader:
            img_name = str(row.get("File Name", "")).strip()
            text = str(row.get("OCRed Text", "")).strip()
            if not img_name or not text:
                continue
            image_path = (image_dir / img_name).resolve()
            if not image_path.exists():
                continue
            records.append(OCRRecord(img_name=img_name, image_path=image_path, ocr_text=text))

    if not records:
        raise ValueError(f"No valid OCR records found in {csv_path}")
    return records


def load_multi_ocr_sources(
    data_root: Path,
    csv_paths: Sequence[str],
    image_subdirs: Sequence[str],
) -> List[OCRRecord]:
    if len(csv_paths) != len(image_subdirs):
        raise ValueError("csv_paths and image_subdirs must have the same length")

    merged: List[OCRRecord] = []
    seen: set[str] = set()
    for csv_rel, image_rel in zip(csv_paths, image_subdirs):
        records = load_ocr_csv(data_root / csv_rel, data_root / image_rel)
        for rec in records:
            key = str(rec.image_path).lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(rec)

    if not merged:
        raise ValueError("No valid OCR records found across provided train sources")
    return merged


def load_images_from_subdirs(data_root: Path, image_subdirs: Sequence[str]) -> List[Path]:
    images: List[Path] = []
    seen: set[str] = set()
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    for image_rel in image_subdirs:
        folder = data_root / image_rel
        if not folder.exists():
            continue
        for pattern in patterns:
            for img in folder.glob(pattern):
                key = str(img.resolve()).lower()
                if key in seen:
                    continue
                seen.add(key)
                images.append(img.resolve())
    return sorted(images)


def load_default_train_eval(
    data_root: Path,
    train_csvs: Sequence[str],
    eval_csv: str,
    image_subdirs_train: Sequence[str],
    image_subdir_eval: str,
) -> Tuple[List[OCRRecord], List[OCRRecord]]:
    train_records = load_multi_ocr_sources(
        data_root=data_root,
        csv_paths=train_csvs,
        image_subdirs=image_subdirs_train,
    )
    eval_records = load_ocr_csv(data_root / eval_csv, data_root / image_subdir_eval)
    return train_records, eval_records


def resolve_default_data_root(cli_data_root: str | None = None) -> Path:
    candidates = []
    if cli_data_root:
        candidates.append(Path(cli_data_root))
    candidates.extend([
        Path("data/kaggle_invoice_images"),
        Path("notebooks/data/kaggle_invoice_images"),
    ])
    for candidate in candidates:
        p = candidate.resolve()
        if p.exists():
            return p
    raise FileNotFoundError("No dataset root found. Tried data/kaggle_invoice_images and notebooks/data/kaggle_invoice_images.")
