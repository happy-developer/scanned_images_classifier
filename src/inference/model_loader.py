from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn
from torchvision import models

from src.data_access.dataset_checks import DatasetContext

from .errors import InferenceExecutionError, ModelNotFoundError


@dataclass
class LoadedModel:
    model: nn.Module
    class_names: List[str]
    img_size: int
    model_version: str
    model_path: Path


def _infer_num_classes(state_dict: Dict[str, Any]) -> int:
    if "head.fc.weight" in state_dict:
        return int(state_dict["head.fc.weight"].shape[0])
    if "fc.weight" in state_dict:
        return int(state_dict["fc.weight"].shape[0])
    raise InferenceExecutionError(message="Checkpoint non supporte (head manquant)")


def _build_model(state_dict: Dict[str, Any], num_classes: int, model_name_hint: str | None = None) -> nn.Module:
    if "head.fc.weight" in state_dict:
        try:
            import timm
        except ImportError as exc:
            raise InferenceExecutionError(
                message="timm requis pour checkpoint Kaggle", details={"dependency": "timm"}
            ) from exc
        model_name = model_name_hint or "rexnet_150"
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    elif "fc.weight" in state_dict:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise InferenceExecutionError(message="Architecture checkpoint non supportee")

    model.load_state_dict(state_dict)
    model.eval()
    return model


def _load_model_meta(meta_path: Path, dataset_context: DatasetContext, checkpoint_num_classes: int) -> Dict[str, Any]:
    if not meta_path.exists():
        raise InferenceExecutionError(message="model_meta.json absent", details={"model_meta_path": str(meta_path)})

    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    labels = meta.get("labels") or []
    img_size = int(meta.get("img_size", 224))

    if not labels:
        labels = dataset_context.class_names

    if len(labels) != checkpoint_num_classes:
        raise InferenceExecutionError(
            message="Incoherence checkpoint vs labels metadata",
            details={
                "checkpoint_num_classes": checkpoint_num_classes,
                "labels_num_classes": len(labels),
                "model_meta_path": str(meta_path),
            },
        )

    if dataset_context.mode == "class_folders" and dataset_context.num_classes != len(labels):
        raise InferenceExecutionError(
            message="Incoherence dataset vs labels metadata",
            details={
                "dataset_num_classes": dataset_context.num_classes,
                "labels_num_classes": len(labels),
                "dataset_root": dataset_context.root,
            },
        )

    meta.setdefault("version", f"local-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    meta.setdefault("trained_at", "unknown")
    meta.setdefault("source_dataset", dataset_context.root)
    meta.setdefault("img_size", img_size)
    meta.setdefault("labels", labels)
    return meta


def load_model(model_path: Path, model_meta_path: Path, dataset_context: DatasetContext) -> LoadedModel:
    if not model_path.exists():
        raise ModelNotFoundError(details={"model_path": str(model_path)})

    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    checkpoint_num_classes = _infer_num_classes(state_dict)
    meta = _load_model_meta(model_meta_path, dataset_context, checkpoint_num_classes)

    model = _build_model(
        state_dict=state_dict,
        num_classes=len(meta["labels"]),
        model_name_hint=str(meta.get("model_name", "rexnet_150")),
    )

    return LoadedModel(
        model=model,
        class_names=[str(x) for x in meta["labels"]],
        img_size=int(meta["img_size"]),
        model_version=str(meta["version"]),
        model_path=model_path,
    )
