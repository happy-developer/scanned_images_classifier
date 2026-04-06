from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _clean_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip().strip(".,;")
    return cleaned or None


def _parse_invoice_fields(text: str) -> dict[str, str | None]:
    normalized = _normalize_text(text)
    patterns: dict[str, list[re.Pattern[str]]] = {
        "invoice_no": [
            re.compile(r"(?:invoice\s*(?:no|number|#)\s*[:\-]?\s*)([A-Z0-9][A-Z0-9/\-]*)", re.I),
            re.compile(r"\binvoice\s*[:\-]?\s*([A-Z0-9][A-Z0-9/\-]*)", re.I),
        ],
        "date": [
            re.compile(
                r"(?:date(?:\s+of\s+issue)?|issue\s+date)\s*[:\-]?\s*"
                r"([0-9]{1,4}[/-][0-9]{1,2}[/-][0-9]{1,4}|[0-9]{2}/[0-9]{4}|[0-9]{4}-[0-9]{2}-[0-9]{2})",
                re.I,
            ),
        ],
        "tax_id": [
            re.compile(r"(?:tax\s*id|tax\s*ident(?:ification)?(?:\s*no)?|vat\s*(?:id|no)?)\s*[:\-]?\s*([A-Z0-9\- ]{4,})", re.I),
        ],
        "iban": [
            re.compile(r"\biban\s*[:\-]?\s*([A-Z0-9 ]{10,34})", re.I),
        ],
        "total": [
            re.compile(
                r"(?:grand\s+total|amount\s+due|invoice\s+total|total(?:\s+due)?|gross\s+worth)\s*[:\-]?\s*"
                r"([$€£]?\s*[0-9][0-9., ]{1,})",
                re.I,
            ),
        ],
    }

    parsed: dict[str, str | None] = {field: None for field in patterns}
    for field, field_patterns in patterns.items():
        for pattern in field_patterns:
            match = pattern.search(normalized)
            if not match:
                continue
            parsed[field] = _clean_value(match.group(1))
            if field == "iban" and parsed[field] is not None:
                parsed[field] = re.sub(r"\s+", "", parsed[field]).upper()
            if field == "tax_id" and parsed[field] is not None:
                parsed[field] = parsed[field].upper()
            break

    if parsed["total"] is None:
        total_match = re.search(r"(?:\btotal\b|\bgross worth\b).{0,30}?([0-9][0-9., ]{1,})", normalized, re.I)
        if total_match:
            parsed["total"] = _clean_value(total_match.group(1))

    return parsed


def _collect_image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    images: list[Path] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            images.append(path.resolve())
    return images


def _resolve_tests_ab_dir(data_root: Path | None, tests_dir: str) -> Path | None:
    candidates: list[Path] = []
    if tests_dir:
        raw = Path(tests_dir)
        if raw.is_absolute():
            candidates.append(raw)
        else:
            if data_root is not None:
                candidates.append(data_root / raw)
            candidates.append(PROJECT_ROOT / raw)
    if data_root is not None:
        candidates.append(data_root / "tests_AB")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    search_roots = [p for p in [data_root, PROJECT_ROOT] if p is not None and p.exists()]
    for root in search_roots:
        for found in root.rglob("tests_AB"):
            if found.is_dir():
                return found.resolve()
    return None


def _safe_import_doctr() -> tuple[Callable[[Path], str], str | None]:
    try:
        from doctr.io import DocumentFile  # type: ignore
        from doctr.models import ocr_predictor  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return (lambda _path: ""), f"{type(exc).__name__}: {exc}"

    try:
        predictor = ocr_predictor(pretrained=True)
    except Exception as exc:  # pragma: no cover - optional dependency
        return (lambda _path: ""), f"{type(exc).__name__}: {exc}"

    def _run(image_path: Path) -> str:
        document = DocumentFile.from_images(str(image_path))
        result = predictor(document)
        export = result.export()
        lines: list[str] = []
        for page in export.get("pages", []):
            for block in page.get("blocks", []):
                for line in block.get("lines", []):
                    words = [str(word.get("value", "")).strip() for word in line.get("words", [])]
                    line_text = " ".join(word for word in words if word).strip()
                    if line_text:
                        lines.append(line_text)
        if lines:
            return "\n".join(lines)
        return str(export.get("prediction", "") or export.get("text", "") or "")

    return _run, None


def _safe_import_paddleocr() -> tuple[Callable[[Path], str], str | None]:
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return (lambda _path: ""), f"{type(exc).__name__}: {exc}"

    last_error: Exception | None = None
    engine = None
    for kwargs in (
        {"lang": "en", "use_angle_cls": True, "use_gpu": False, "show_log": False},
        {"lang": "en", "use_gpu": False, "show_log": False},
        {"lang": "en"},
    ):
        try:
            engine = PaddleOCR(**kwargs)
            break
        except Exception as exc:  # pragma: no cover - optional dependency
            last_error = exc
            engine = None
    if engine is None:
        return (lambda _path: ""), f"{type(last_error).__name__}: {last_error}" if last_error else "Unknown initialization failure"

    def _run(image_path: Path) -> str:
        result = engine.ocr(str(image_path), cls=True)  # type: ignore[union-attr]
        if not result:
            return ""
        lines: list[str] = []
        rows = result[0] if len(result) == 1 and isinstance(result[0], list) else result
        for item in rows:
            try:
                if isinstance(item, list) and len(item) >= 2:
                    text = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
                elif isinstance(item, tuple) and len(item) >= 2:
                    text = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
                else:
                    text = ""
            except Exception:
                text = ""
            text = str(text).strip()
            if text:
                lines.append(text)
        return "\n".join(lines)

    return _run, None


def _safe_import_pytesseract() -> tuple[Callable[[Path], str], str | None]:
    try:
        import pytesseract  # type: ignore
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency
        return (lambda _path: ""), f"{type(exc).__name__}: {exc}"

    if shutil.which("tesseract") is None:
        return (lambda _path: ""), "tesseract executable not found on PATH"

    try:
        _ = pytesseract.get_tesseract_version()
    except Exception as exc:  # pragma: no cover - optional dependency
        return (lambda _path: ""), f"{type(exc).__name__}: {exc}"

    def _run(image_path: Path) -> str:
        image = Image.open(image_path).convert("RGB")
        return str(pytesseract.image_to_string(image, lang="eng", config="--psm 6"))

    return _run, None


@dataclass
class BackendRuntime:
    name: str
    runner: Callable[[Path], str] | None
    availability: bool
    error: str | None = None


def _build_backends(selected: list[str]) -> list[BackendRuntime]:
    builders: dict[str, Callable[[], tuple[Callable[[Path], str], str | None]]] = {
        "paddleocr": _safe_import_paddleocr,
        "doctr": _safe_import_doctr,
        "pytesseract": _safe_import_pytesseract,
    }

    backends: list[BackendRuntime] = []
    for name in selected:
        builder = builders[name]
        runner, error = builder()
        backends.append(
            BackendRuntime(
                name=name,
                runner=None if error else runner,
                availability=error is None,
                error=error,
            )
        )
    return backends


def _iter_backend_results(
    backend: BackendRuntime,
    image_paths: Iterable[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    latency_values: list[float] = []
    char_counts: list[int] = []
    successes = 0
    failures = 0

    if not backend.availability or backend.runner is None:
        for image_path in image_paths:
            rows.append(
                {
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                    "backend": backend.name,
                    "availability": False,
                    "prediction": None,
                    "latency_ms": None,
                    "parsed_fields": {field: None for field in ("invoice_no", "date", "tax_id", "iban", "total")},
                    "error": backend.error,
                }
            )
        return rows, {
            "backend": backend.name,
            "availability": False,
            "error": backend.error,
            "num_images": len(rows),
            "successes": 0,
            "failures": len(rows),
            "avg_latency_ms": None,
            "avg_chars": None,
        }

    for image_path in image_paths:
        started = time.perf_counter()
        prediction = ""
        error: str | None = None
        available = True
        try:
            prediction = str(backend.runner(image_path))  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover - optional runtime failure
            available = False
            error = f"{type(exc).__name__}: {exc}"
            prediction = ""
            failures += 1
        else:
            successes += 1
        latency_ms = (time.perf_counter() - started) * 1000.0
        if available:
            latency_values.append(latency_ms)
            char_counts.append(len(_normalize_text(prediction)))

        rows.append(
            {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "backend": backend.name,
                "availability": available,
                "prediction": _normalize_text(prediction),
                "latency_ms": round(latency_ms, 3) if available else None,
                "parsed_fields": _parse_invoice_fields(prediction),
                "error": error,
            }
        )

    summary = {
        "backend": backend.name,
        "availability": True,
        "error": backend.error,
        "num_images": len(rows),
        "successes": successes,
        "failures": failures,
        "avg_latency_ms": round(statistics.mean(latency_values), 3) if latency_values else None,
        "avg_chars": round(statistics.mean(char_counts), 2) if char_counts else None,
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark non-generative OCR backends on tests_AB with graceful fallback when optional deps are missing."
    )
    parser.add_argument("--data-root", type=str, default="", help="Dataset root containing tests_AB. Falls back to Kaggle cache resolution.")
    parser.add_argument("--tests-dir", type=str, default="tests_AB", help="Relative or absolute path to the tests_AB folder.")
    parser.add_argument(
        "--backends",
        type=str,
        default="auto",
        help="Comma-separated backend list: auto, paddleocr, doctr, pytesseract. Default runs every supported backend.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of images to benchmark. Use 0 for all images.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/ocr_baseline_benchmark.json",
        help="Where to write the full benchmark JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from src.data_access.kagglehub_resolver import resolve_kaggle_dataset_root
    except Exception as exc:  # pragma: no cover - repository import failure
        payload = {
            "schema_version": "ocr_baseline_benchmark.v1",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    data_root: Path | None = None
    try:
        data_root = resolve_kaggle_dataset_root(args.data_root or None)
    except Exception:
        if args.data_root:
            candidate = Path(args.data_root).expanduser().resolve()
            if candidate.exists():
                data_root = candidate

    tests_ab_dir = _resolve_tests_ab_dir(data_root, args.tests_dir)
    image_paths = _collect_image_files(tests_ab_dir) if tests_ab_dir is not None else []
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    backend_names = ["paddleocr", "doctr", "pytesseract"]
    if args.backends.strip().lower() != "auto":
        requested = {name.strip().lower() for name in _split_csv(args.backends)}
        backend_names = [name for name in backend_names if name in requested]

    backends = _build_backends(backend_names)

    all_rows: list[dict[str, Any]] = []
    backend_summaries: list[dict[str, Any]] = []
    for backend in backends:
        rows, summary = _iter_backend_results(backend, image_paths)
        all_rows.extend(rows)
        backend_summaries.append(summary)

    payload = {
        "schema_version": "ocr_baseline_benchmark.v1",
        "dataset_root": str(data_root) if data_root is not None else None,
        "tests_ab_dir": str(tests_ab_dir) if tests_ab_dir is not None else None,
        "num_images": len(image_paths),
        "requested_backends": backend_names,
        "backend_status": [
            {
                "backend": backend.name,
                "availability": backend.availability,
                "error": backend.error,
            }
            for backend in backends
        ],
        "backend_summaries": backend_summaries,
        "rows": all_rows,
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "num_images": len(image_paths),
                "backend_status": payload["backend_status"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
