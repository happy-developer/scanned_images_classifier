from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from torchvision import transforms

from .errors import InvalidImageError


def ensure_valid_image(image: Any) -> Image.Image:
    if image is None:
        raise InvalidImageError(message="Aucune image fournie")
    if not isinstance(image, Image.Image):
        raise InvalidImageError(message="Type image non supporte", details={"type": str(type(image))})
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def build_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def infer_dataset_context_for_image(image_path: str | None, dataset_root: Path) -> str:
    if not image_path:
        return str(dataset_root)
    try:
        p = Path(image_path)
        if p.exists():
            return str(p.parent)
    except Exception:
        pass
    return str(dataset_root)
