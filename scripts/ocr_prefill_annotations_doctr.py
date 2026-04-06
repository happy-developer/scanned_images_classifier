from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefill OCRed Text in annotation CSV with docTR predictions.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--annotations-csv", type=str, default="outputs/real_val_to_annotate.csv")
    parser.add_argument("--output-csv", type=str, default="outputs/real_val_prefilled_doctr.csv")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional max rows to prefill; 0 means all")
    parser.add_argument("--cache-dir", type=str, default="outputs/doctr_cache")
    return parser.parse_args()


def _normalize_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _resolve_image_path(data_root: Path, row: dict[str, str], csv_path: Path) -> Path | None:
    source_path = str(row.get("source_path", "") or "").strip()
    file_name = str(row.get("File Name", "") or "").strip()

    candidates: list[Path] = []
    if source_path:
        src = Path(source_path)
        if src.is_absolute():
            candidates.append(src)
        else:
            candidates.append((data_root / src).resolve())
            candidates.append((csv_path.parent / src).resolve())
    if file_name:
        candidates.append((data_root / file_name).resolve())
        candidates.append((csv_path.parent / file_name).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _doctr_runner(cache_dir: Path):
    os.environ.setdefault("DOCTR_CACHE_DIR", str(cache_dir.resolve()))
    from doctr.io import DocumentFile  # type: ignore
    from doctr.models import ocr_predictor  # type: ignore

    predictor = ocr_predictor(pretrained=True)

    def run(image_path: Path) -> str:
        doc = DocumentFile.from_images(str(image_path))
        result = predictor(doc)
        export = result.export()
        lines: list[str] = []
        for page in export.get("pages", []):
            for block in page.get("blocks", []):
                for line in block.get("lines", []):
                    words = [str(word.get("value", "")).strip() for word in line.get("words", [])]
                    line_text = " ".join(word for word in words if word).strip()
                    if line_text:
                        lines.append(line_text)
        return _normalize_text("\n".join(lines))

    return run


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    annotations_csv = Path(args.annotations_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not annotations_csv.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {annotations_csv}")

    with annotations_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list((rows[0].keys() if rows else ["File Name", "OCRed Text", "split", "source_path", "notes"]))

    run_ocr = _doctr_runner(cache_dir)

    filled = 0
    skipped_existing = 0
    missing_image = 0
    processed = 0

    for row in rows:
        if args.max_rows and processed >= args.max_rows:
            break
        processed += 1

        text = str(row.get("OCRed Text", "") or "").strip()
        if text:
            skipped_existing += 1
            continue

        image_path = _resolve_image_path(data_root, row, annotations_csv)
        if image_path is None:
            missing_image += 1
            continue

        pred = run_ocr(image_path)
        row["OCRed Text"] = pred
        filled += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "output_csv": str(output_csv),
                "num_rows": len(rows),
                "processed_rows": processed,
                "filled_rows": filled,
                "skipped_existing": skipped_existing,
                "missing_image": missing_image,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
