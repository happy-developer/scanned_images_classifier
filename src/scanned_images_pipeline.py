from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


@dataclass
class PipelineConfig:
    project_root: Path
    data_root: Path
    artifacts_dir: Path
    batch_size: int = 16
    img_size: int = 224
    epochs: int = 2
    learning_rate: float = 1e-3
    seed: int = 42
    num_workers: int = 0
    use_pretrained: bool = True
    use_synthetic_if_missing: bool = True
    synthetic_train_per_class: int = 24
    synthetic_val_per_class: int = 8


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _draw_scanned_like_image(size: int, rng: np.random.Generator) -> Image.Image:
    image = Image.new("RGB", (size, size), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    for y in range(12, size - 8, 12):
        shade = int(rng.integers(120, 180))
        draw.line((8, y, size - 8, y), fill=(shade, shade, shade), width=1)
    for _ in range(10):
        x1 = int(rng.integers(5, size - 20))
        y1 = int(rng.integers(5, size - 20))
        w = int(rng.integers(8, 30))
        h = int(rng.integers(4, 14))
        shade = int(rng.integers(20, 90))
        draw.rectangle((x1, y1, x1 + w, y1 + h), fill=(shade, shade, shade))
    return image


def _draw_photo_like_image(size: int, rng: np.random.Generator) -> Image.Image:
    pixels = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    # Add smooth color blobs so this class looks less document-like.
    for _ in range(6):
        cx = int(rng.integers(0, size))
        cy = int(rng.integers(0, size))
        radius = int(rng.integers(size // 10, size // 4))
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        pixels[mask] = (pixels[mask] * 0.5 + color * 0.5).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def create_synthetic_dataset(
    target_root: Path, train_per_class: int, val_per_class: int, img_size: int, seed: int
) -> Tuple[Path, Path]:
    classes = ["scanned", "not_scanned"]
    rng = np.random.default_rng(seed)
    train_dir = target_root / "train"
    val_dir = target_root / "val"

    for split_dir, samples_per_class in [(train_dir, train_per_class), (val_dir, val_per_class)]:
        for class_name in classes:
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(samples_per_class):
                if class_name == "scanned":
                    image = _draw_scanned_like_image(img_size, rng)
                else:
                    image = _draw_photo_like_image(img_size, rng)
                image.save(class_dir / f"{class_name}_{idx:03d}.png")

    return train_dir, val_dir


def resolve_data_dirs(config: PipelineConfig) -> Tuple[Path, Path, bool]:
    train_candidates = ["train", "training"]
    val_candidates = ["val", "valid", "validation"]

    for train_name in train_candidates:
        for val_name in val_candidates:
            train_dir = config.data_root / train_name
            val_dir = config.data_root / val_name
            if train_dir.exists() and val_dir.exists():
                return train_dir, val_dir, False

    if not config.use_synthetic_if_missing:
        raise FileNotFoundError(
            f"No train/val dataset split found under: {config.data_root}. "
            "Expected e.g. data/scanned_images/train and data/scanned_images/val."
        )

    synthetic_root = config.data_root / "_synthetic"
    train_dir, val_dir = create_synthetic_dataset(
        target_root=synthetic_root,
        train_per_class=config.synthetic_train_per_class,
        val_per_class=config.synthetic_val_per_class,
        img_size=config.img_size,
        seed=config.seed,
    )
    return train_dir, val_dir, True


def build_dataloaders(
    train_dir: Path, val_dir: Path, batch_size: int, img_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_tfms)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=eval_tfms)

    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            f"Class mismatch between train and val splits: {train_dataset.classes} vs {val_dataset.classes}"
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    return train_loader, val_loader, train_dataset.classes


def build_model(num_classes: int, use_pretrained: bool) -> nn.Module:
    model_weights = None
    if use_pretrained:
        try:
            model_weights = models.ResNet18_Weights.DEFAULT
        except Exception:
            model_weights = None

    try:
        model = models.resnet18(weights=model_weights)
    except Exception:
        # Fallback when pre-trained weights cannot be downloaded in offline environments.
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    epoch_acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    running_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    epoch_acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    return epoch_loss, epoch_acc, y_true, y_pred


def run_training_pipeline(config: PipelineConfig) -> Dict[str, object]:
    seed_everything(config.seed)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir, val_dir, synthetic_used = resolve_data_dirs(config)
    train_loader, val_loader, class_names = build_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=config.batch_size,
        img_size=config.img_size,
        num_workers=config.num_workers,
    )

    model = build_model(num_classes=len(class_names), use_pretrained=config.use_pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state_dict = None
    best_val_acc = -1.0

    for _ in range(config.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    _, final_val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    model_path = config.artifacts_dir / "scanned_images_resnet18.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        },
        model_path,
    )

    summary = {
        "device": str(device),
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "synthetic_data_used": synthetic_used,
        "class_names": class_names,
        "history": history,
        "best_val_acc": best_val_acc,
        "final_val_acc": final_val_acc,
        "classification_report": report,
        "model_path": str(model_path),
    }
    return summary


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a scanned images classifier with PyTorch.")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root containing train/val splits.")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Directory to write model artifacts.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-pretrained",
        action="store_true",
        help="Disable pretrained initialization for offline or deterministic smoke runs.",
    )
    parser.add_argument(
        "--no-synthetic-fallback",
        action="store_true",
        help="Fail if train/val splits are missing instead of generating synthetic data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = _default_project_root()
    data_root = Path(args.data_root) if args.data_root else (project_root / "data" / "scanned_images")
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else (project_root / "artifacts")

    config = PipelineConfig(
        project_root=project_root,
        data_root=data_root,
        artifacts_dir=artifacts_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        use_pretrained=not args.disable_pretrained,
        use_synthetic_if_missing=not args.no_synthetic_fallback,
    )

    summary = run_training_pipeline(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
