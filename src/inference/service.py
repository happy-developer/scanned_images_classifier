from __future__ import annotations

import csv
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from .errors import InferenceExecutionError, InvalidImageError, ModelNotFoundError
from .schemas import InferenceResult

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
class LoadedModel:
    model: nn.Module
    class_names: List[str]
    img_size: int
    model_version: str
    model_path: Path


def _build_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_timm_model(model_name: str, num_classes: int) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise InferenceExecutionError(
            message="Le package timm est requis pour charger ce checkpoint Kaggle.",
            details={"dependency": "timm"},
        ) from exc

    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


def _resolve_class_names(data_root: Path | str | None, classes_csv: Path | str | None) -> List[str]:
    if classes_csv is not None:
        csv_path = Path(classes_csv)
        if not csv_path.exists():
            raise InferenceExecutionError(
                message="Fichier CSV des classes introuvable.",
                details={"classes_csv": str(csv_path)},
            )
        names: List[str] = []
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise InferenceExecutionError(message="CSV des classes vide.", details={"classes_csv": str(csv_path)})
            preferred = ["class_name", "label", "class", "name"]
            selected_column = next((c for c in preferred if c in reader.fieldnames), reader.fieldnames[0])
            for row in reader:
                value = str(row.get(selected_column, "")).strip()
                if value:
                    names.append(value)
        if not names:
            raise InferenceExecutionError(
                message="Impossible d'extraire les classes depuis le CSV.",
                details={"classes_csv": str(csv_path)},
            )
        return names

    if data_root is None:
        raise InferenceExecutionError(
            message="data_root requis pour resoudre les classes du mode Kaggle.",
            details={"data_root": None},
        )

    root = Path(data_root)
    if not root.exists():
        raise InferenceExecutionError(
            message="Dataset Kaggle introuvable.",
            details={"data_root": str(root)},
        )

    top_batch_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("batch_")])
    if top_batch_dirs:
        csv_paths = sorted([p for p in root.rglob("*.csv") if p.is_file()])
        if not csv_paths:
            raise InferenceExecutionError(
                message="Dataset Kaggle detecte (batch_*), mais aucun CSV manifeste trouve.",
                details={"data_root": str(root)},
            )
        labels = set()
        for csv_path in csv_paths:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                canonical_to_name = {name.strip().lower(): name for name in reader.fieldnames}
                label_col = None
                for candidate in LABEL_COL_CANDIDATES:
                    if candidate in canonical_to_name:
                        label_col = canonical_to_name[candidate]
                        break
                if label_col is None:
                    for fn in reader.fieldnames:
                        if "label" in fn.strip().lower() or "class" in fn.strip().lower():
                            label_col = fn
                            break
                image_col = None
                for name in reader.fieldnames:
                    token = name.strip().lower()
                    if token in {"image_path", "img_path", "path", "filepath", "file_path", "filename", "file_name", "image", "img", "file name"}:
                        image_col = name
                        break
                for row in reader:
                    label = str(row.get(label_col, "")).strip() if label_col else ""
                    if not label and image_col:
                        raw_ref = str(row.get(image_col, "")).strip()
                        if raw_ref:
                            ref = Path(raw_ref)
                            candidates = [
                                (csv_path.parent / ref),
                                (csv_path.parent / csv_path.stem / ref),
                                (csv_path.parent.parent / ref),
                            ]
                            for candidate in candidates:
                                if candidate.exists():
                                    label = candidate.parent.name
                                    break
                    if label:
                        labels.add(label)
        if not labels:
            raise InferenceExecutionError(
                message="Impossible d'extraire les classes depuis les CSV Kaggle.",
                details={"data_root": str(root)},
            )
        return sorted(labels)

    class_dirs = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise InferenceExecutionError(
            message="Aucune classe resolvable dans data_root.",
            details={"data_root": str(root)},
        )
    return class_dirs


def _infer_num_classes_from_state_dict(state_dict: Dict[str, Any]) -> int:
    if "head.fc.weight" in state_dict:
        return int(state_dict["head.fc.weight"].shape[0])
    if "fc.weight" in state_dict:
        return int(state_dict["fc.weight"].shape[0])
    raise InferenceExecutionError(message="Impossible d'inferer num_classes depuis le checkpoint.")


def _build_model_from_checkpoint(state_dict: Dict[str, Any], num_classes: int) -> nn.Module:
    if "head.fc.weight" in state_dict:
        model = _build_timm_model(model_name="rexnet_150", num_classes=num_classes)
    elif "fc.weight" in state_dict:
        model = _build_resnet18(num_classes=num_classes)
    else:
        raise InferenceExecutionError(
            message="Architecture de checkpoint non supportee.",
            details={"known_heads": ["head.fc.weight", "fc.weight"]},
        )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _as_state_dict(checkpoint_obj: Any) -> Dict[str, Any]:
    if isinstance(checkpoint_obj, OrderedDict):
        return dict(checkpoint_obj)
    if isinstance(checkpoint_obj, dict):
        if "state_dict" in checkpoint_obj:
            return checkpoint_obj["state_dict"]
        if "model_state_dict" in checkpoint_obj:
            return checkpoint_obj["model_state_dict"]
    raise InferenceExecutionError(message="Format de checkpoint non supporte.")


def load_model(
    model_path: Path | str,
    data_root: Path | str | None = None,
    classes_csv: Path | str | None = None,
    strict_class_check: bool = True,
) -> LoadedModel:
    path = Path(model_path)
    if not path.exists():
        raise ModelNotFoundError(details={"model_path": str(path)})

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = _as_state_dict(checkpoint)

    # Prefer explicit classes in checkpoint; otherwise resolve from Kaggle dataset root/csv.
    class_names = None
    model_version = f"{path.stem}-local"
    img_size = 224

    if isinstance(checkpoint, dict):
        ckpt_classes = checkpoint.get("class_names")
        if isinstance(ckpt_classes, list) and ckpt_classes:
            class_names = [str(c) for c in ckpt_classes]
        config = checkpoint.get("config") or {}
        if "img_size" in config:
            img_size = int(config["img_size"])
        if "seed" in config:
            model_version = f"resnet18-seed-{config['seed']}"

    if class_names is None:
        class_names = _resolve_class_names(data_root=data_root, classes_csv=classes_csv)
        model_version = f"{path.stem}-kaggle"

    expected_classes = _infer_num_classes_from_state_dict(state_dict)
    if strict_class_check and expected_classes != len(class_names):
        raise InferenceExecutionError(
            message="Incoherence checkpoint/classes detectee au demarrage.",
            details={
                "expected_num_classes": expected_classes,
                "resolved_num_classes": len(class_names),
                "model_path": str(path),
            },
        )

    model = _build_model_from_checkpoint(state_dict=state_dict, num_classes=len(class_names))

    return LoadedModel(
        model=model,
        class_names=class_names,
        img_size=img_size,
        model_version=model_version,
        model_path=path,
    )


def _build_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def predict_image(
    loaded: LoadedModel,
    image: Image.Image,
    run_id: str,
    logs_path: Path,
) -> InferenceResult:
    if image is None:
        raise InvalidImageError(message="Aucune image fournie.")

    try:
        image = _ensure_rgb(image)
    except Exception as exc:
        raise InvalidImageError(message="Impossible de lire l'image.", details={"reason": str(exc)}) from exc

    tfm = _build_eval_transform(loaded.img_size)

    start = time.perf_counter()
    try:
        with torch.no_grad():
            tensor = tfm(image).unsqueeze(0)
            logits = loaded.model(tensor)
            probs_tensor = torch.softmax(logits, dim=1)[0].cpu()
    except Exception as exc:
        raise InferenceExecutionError(details={"reason": str(exc)}) from exc
    latency_ms = (time.perf_counter() - start) * 1000.0

    probs = {
        class_name: float(probs_tensor[idx].item())
        for idx, class_name in enumerate(loaded.class_names)
    }
    label = max(probs, key=probs.get)

    result = InferenceResult(
        label=label,
        confidence=float(probs[label]),
        probs=probs,
        model_version=loaded.model_version,
        run_id=run_id,
        latency_ms=round(latency_ms, 3),
    )

    _append_jsonl(
        logs_path,
        {
            "event": "inference_run",
            "model_path": str(loaded.model_path),
            "run_id": run_id,
            "result": result.to_dict(),
            "timestamp_ms": int(time.time() * 1000),
        },
    )

    return result
