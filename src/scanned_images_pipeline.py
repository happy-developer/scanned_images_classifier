from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
try:
    from src.data_access.dataset_checks import validate_dataset_structure
except ModuleNotFoundError:
    from data_access.dataset_checks import validate_dataset_structure

KAGGLE_DATASET_ID = "osamahosamabdellatif/high-quality-invoice-images-for-ocr"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_COL_CANDIDATES = [
    "label",
    "class",
    "class_name",
    "category",
    "document_type",
    "type",
    "target",
]
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
SPLIT_COL_CANDIDATES = ["split", "subset", "set"]


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
    val_split: float = 0.2


@dataclass
class ManifestRecord:
    image_path: Path
    label_name: str
    split_hint: Optional[str] = None
    label_source: str = "csv_label"


class KaggleManifestDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[Path, int]], transform: transforms.Compose) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_kaggle_dataset() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub est requis pour telecharger le dataset. Installez: pip install kagglehub"
        ) from exc

    path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
    data_root = Path(path).resolve()
    print(f"Path to dataset files: {data_root}")
    return data_root


def _canonical(value: str) -> str:
    return value.strip().lower()


def _discover_manifest_csvs(data_root: Path) -> List[Path]:
    csvs = sorted([p for p in data_root.rglob("*.csv") if p.is_file()])
    if not csvs:
        raise FileNotFoundError(
            f"No CSV manifest found under {data_root}. Expected files like batch_1/batch1_1.csv."
        )
    return csvs


def _discover_batch_dirs(data_root: Path) -> List[Path]:
    batch_dirs = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("batch_")])
    if not batch_dirs:
        raise FileNotFoundError(
            f"No top-level batch_* directories found under {data_root}. "
            "Expected e.g. batch_1, batch_2, batch_3."
        )
    return batch_dirs


def _find_column(fieldnames: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    by_canonical = {_canonical(name): name for name in fieldnames}
    for candidate in candidates:
        found = by_canonical.get(_canonical(candidate))
        if found:
            return found
    return None


def _looks_like_image_ref(value: str) -> bool:
    return Path(value).suffix.lower() in IMAGE_EXTS


def _resolve_image_path(csv_path: Path, image_ref: str) -> Path:
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
    raise FileNotFoundError(f"Unable to resolve image path '{image_ref}' referenced in {csv_path}")


def _parse_manifest_csv(csv_path: Path) -> List[ManifestRecord]:
    records: List[ManifestRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")

        label_col = _find_column(reader.fieldnames, LABEL_COL_CANDIDATES)
        image_col = _find_column(reader.fieldnames, IMAGE_COL_CANDIDATES)
        split_col = _find_column(reader.fieldnames, SPLIT_COL_CANDIDATES)

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

        if image_col is None:
            raise ValueError(
                f"Unable to infer image column in {csv_path}. Found columns: {reader.fieldnames}"
            )

        for row in reader:
            raw_image_ref = (row.get(image_col) or "").strip()
            raw_label = (row.get(label_col) or "").strip() if label_col else ""
            label_source = "csv_label"
            if not raw_image_ref:
                continue
            if not _looks_like_image_ref(raw_image_ref):
                continue
            image_path = _resolve_image_path(csv_path, raw_image_ref)
            if not raw_label:
                raw_label = image_path.parent.name
                label_source = "parent_folder_fallback"
            split_hint = (row.get(split_col) or "").strip() if split_col else ""
            records.append(
                ManifestRecord(
                    image_path=image_path,
                    label_name=raw_label,
                    split_hint=split_hint or None,
                    label_source=label_source,
                )
            )

    if not records:
        raise ValueError(f"No valid image records parsed from {csv_path}")
    return records


def _normalize_split(split_hint: Optional[str]) -> Optional[str]:
    if not split_hint:
        return None
    token = _canonical(split_hint)
    if token in {"train", "training"}:
        return "train"
    if token in {"val", "valid", "validation", "dev", "test"}:
        return "val"
    return None


def load_kaggle_manifest_records(data_root: Path) -> List[ManifestRecord]:
    _discover_batch_dirs(data_root)
    csv_paths = _discover_manifest_csvs(data_root)
    records: List[ManifestRecord] = []
    for csv_path in csv_paths:
        records.extend(_parse_manifest_csv(csv_path))
    if not records:
        raise ValueError(f"No valid records loaded from CSV manifests under {data_root}")
    return records


def split_records(
    records: Sequence[ManifestRecord], val_split: float, seed: int
) -> Tuple[List[ManifestRecord], List[ManifestRecord]]:
    if not (0.0 < val_split < 1.0):
        raise ValueError(f"val_split must be in (0,1), got {val_split}")

    train_records = [r for r in records if _normalize_split(r.split_hint) == "train"]
    val_records = [r for r in records if _normalize_split(r.split_hint) == "val"]
    unknown_records = [r for r in records if _normalize_split(r.split_hint) is None]

    if unknown_records:
        rng = random.Random(seed)
        grouped: Dict[str, List[ManifestRecord]] = defaultdict(list)
        for record in unknown_records:
            grouped[record.label_name].append(record)
        for _, group in grouped.items():
            group_sorted = sorted(group, key=lambda x: str(x.image_path))
            rng.shuffle(group_sorted)
            if len(group_sorted) <= 1:
                train_records.extend(group_sorted)
                continue
            n_val = max(1, int(round(len(group_sorted) * val_split)))
            n_val = min(n_val, len(group_sorted) - 1)
            val_records.extend(group_sorted[:n_val])
            train_records.extend(group_sorted[n_val:])

    if not train_records or not val_records:
        raise ValueError(
            "Train/val split is empty after processing CSV manifests. "
            "Please verify labels/split values in Kaggle CSV files."
        )
    return train_records, val_records


def build_dataloaders_from_records(
    train_records: Sequence[ManifestRecord],
    val_records: Sequence[ManifestRecord],
    batch_size: int,
    img_size: int,
    num_workers: int,
    class_names_override: Optional[Sequence[str]] = None,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    labels_from_records = {r.label_name for r in train_records + val_records}
    if class_names_override is not None:
        class_names = list(class_names_override)
        missing = sorted(labels_from_records.difference(class_names))
        if missing:
            raise ValueError(
                f"Labels from records missing in dataset_context.class_names: {missing}"
            )
    else:
        class_names = sorted(labels_from_records)
    if len(class_names) < 2:
        raise ValueError(f"Need at least 2 classes for classification, got {class_names}")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    train_samples = [(r.image_path, class_to_idx[r.label_name]) for r in train_records]
    val_samples = [(r.image_path, class_to_idx[r.label_name]) for r in val_records]

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

    train_dataset = KaggleManifestDataset(samples=train_samples, transform=train_tfms)
    val_dataset = KaggleManifestDataset(samples=val_samples, transform=eval_tfms)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    return train_loader, val_loader, class_names


def summarize_label_sources(records: Sequence[ManifestRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for record in records:
        counts[record.label_source] += 1
    return dict(sorted(counts.items()))


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
    dataset_context = validate_dataset_structure(config.data_root)
    records = load_kaggle_manifest_records(config.data_root)
    label_source_counts = summarize_label_sources(records)
    train_records, val_records = split_records(records, val_split=config.val_split, seed=config.seed)
    labels_from_records = sorted({r.label_name for r in train_records + val_records})
    if (
        len(labels_from_records) != dataset_context.num_classes
        or set(labels_from_records) != set(dataset_context.class_names)
    ):
        raise ValueError(
            "Dataset context labels mismatch with pipeline labels. "
            f"context={dataset_context.class_names}, pipeline={labels_from_records}"
        )
    train_loader, val_loader, class_names = build_dataloaders_from_records(
        train_records=train_records,
        val_records=val_records,
        batch_size=config.batch_size,
        img_size=config.img_size,
        num_workers=config.num_workers,
        class_names_override=dataset_context.class_names,
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
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )

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
        "data_root": str(config.data_root),
        "num_manifest_records": len(records),
        "label_policy": (
            "Prefer CSV label/class columns; fallback to image parent folder name when label is missing."
        ),
        "label_source_counts": label_source_counts,
        "num_train_samples": len(train_records),
        "num_val_samples": len(val_records),
        "dataset_context": dataset_context.to_dict(),
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
    parser = argparse.ArgumentParser(
        description="Train a scanned images classifier from Kaggle manifests (batch_* + csv)."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Kaggle dataset root (batch_* + csv). If omitted with --download-latest, kagglehub is used.",
    )
    parser.add_argument(
        "--download-latest",
        action="store_true",
        help="Download latest Kaggle dataset version via kagglehub before training.",
    )
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Directory to write model artifacts.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio when CSV has no split.")
    parser.add_argument(
        "--disable-pretrained",
        action="store_true",
        help="Disable pretrained initialization for offline or deterministic smoke runs.",
    )
    return parser.parse_args()


def resolve_data_root(args: argparse.Namespace) -> Path:
    if args.download_latest:
        return download_kaggle_dataset()

    if args.data_root:
        return Path(args.data_root).expanduser().resolve()

    raise ValueError("Provide --data-root or use --download-latest to fetch via kagglehub.")


def main() -> None:
    args = parse_args()
    project_root = _default_project_root()
    data_root = resolve_data_root(args)
    artifacts_dir = Path(args.artifacts_dir).resolve() if args.artifacts_dir else (project_root / "artifacts")

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
        val_split=args.val_split,
    )

    summary = run_training_pipeline(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
