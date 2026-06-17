from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms

from main import load_config
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.models.encoder import VisionTransformerEncoder, build_encoder

DEFAULT_DATA_ROOT = Path("data/imagenet50-200")
DEFAULT_OUTPUT_DIR = Path("outputs/imagenet50-200-vit-small-original-mask-comparison")
DEFAULT_JEPA_CONFIG = Path("configs/imagenet50_200_vit_small_original_mask_jepa.yaml")
DEFAULT_JEPA_CHECKPOINT = Path(
    "outputs/imagenet50-200-vit-small-original-mask-jepa/checkpoints/"
    "imagenet50-200-vit-small-original-mask-jepa_latest.pt"
)


class JepaFeatureExtractor(nn.Module):
    def __init__(self, encoder: VisionTransformerEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, images: Tensor) -> Tensor:
        return self.encoder(images).mean(dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare random, JEPA, supervised scratch, fine-tuned and torchvision "
            "features with the same supervised linear probe."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-folder", default="train")
    parser.add_argument("--val-folder", default="val")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        choices=(
            "random-frozen-vit-b16",
            "jepa-checkpoint",
            "jepa-full-finetune",
            "supervised-vit-b16-scratch",
            "torchvision-vit-b16",
        ),
        help="Model to evaluate. Repeat to select a subset. Defaults to all four.",
    )
    parser.add_argument("--jepa-config", type=Path, default=DEFAULT_JEPA_CONFIG)
    parser.add_argument("--jepa-checkpoint", type=Path, default=DEFAULT_JEPA_CHECKPOINT)
    parser.add_argument("--torchvision-weights", default="IMAGENET1K_V1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--linear-epochs", type=int, default=20)
    parser.add_argument("--linear-lr", type=float, default=0.05)
    parser.add_argument("--linear-batch-size", type=int, default=512)
    parser.add_argument("--supervised-epochs", type=int, default=20)
    parser.add_argument("--supervised-lr", type=float, default=3e-4)
    parser.add_argument("--supervised-weight-decay", type=float, default=0.05)
    parser.add_argument("--supervised-batch-size", type=int, default=64)
    parser.add_argument("--fine-tune-epochs", type=int, default=20)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser.add_argument("--fine-tune-weight-decay", type=float, default=0.05)
    parser.add_argument("--fine-tune-batch-size", type=int, default=64)
    parser.add_argument(
        "--supervised-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint for the supervised-from-scratch ViT baseline.",
    )
    parser.add_argument(
        "--refresh-supervised",
        action="store_true",
        help=(
            "Retrain the supervised-from-scratch baseline even if a checkpoint exists."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument(
        "--refresh-features",
        action="store_true",
        help="Ignore cached feature tensors and extract features again.",
    )
    return parser.parse_args()


def eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_loader(
    root: Path,
    transform: Any,
    batch_size: int,
    num_workers: int,
    device: str,
    shuffle: bool = False,
) -> DataLoader:
    dataset = datasets.ImageFolder(root, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )


def get_vit_b16_weights(name: str) -> models.ViT_B_16_Weights | None:
    if name.lower() in {"none", "random"}:
        return None
    try:
        return getattr(models.ViT_B_16_Weights, name)
    except AttributeError as exc:
        available = [weight.name for weight in models.ViT_B_16_Weights]
        raise ValueError(
            f"Unknown vit_b_16 weights {name!r}; choose one of {available}"
        ) from exc


def build_torchvision_feature_model(weights: str) -> nn.Module:
    weights_obj = get_vit_b16_weights(weights)
    model = models.vit_b_16(weights=weights_obj)
    model.heads = nn.Identity()
    return model


def build_random_feature_model(config_path: Path, seed: int) -> nn.Module:
    config = load_config(config_path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = build_encoder(config.get("model", {}).get("encoder", {}))
    return JepaFeatureExtractor(encoder)


def build_jepa_feature_model(config_path: Path, checkpoint_path: Path) -> nn.Module:
    config = load_config(config_path)
    encoder = build_encoder(config.get("model", {}).get("encoder", {}))
    encoder_state = load_jepa_encoder_state(checkpoint_path)
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected JEPA encoder checkpoint keys: {unexpected}")
    if missing:
        raise RuntimeError(f"Missing JEPA encoder checkpoint keys: {missing}")
    return JepaFeatureExtractor(encoder)


class SupervisedVitClassifier(nn.Module):
    def __init__(self, config_path: Path, num_classes: int, seed: int) -> None:
        super().__init__()
        config = load_config(config_path)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = build_encoder(config.get("model", {}).get("encoder", {}))
            self.head = nn.Linear(self.encoder.config.embed_dim, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        features = self.encoder(images).mean(dim=1)
        return self.head(features)


def supervised_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.supervised_checkpoint is not None:
        return args.supervised_checkpoint
    return args.output_dir / "supervised" / "supervised_vit_b16_scratch.pt"


def jepa_finetune_checkpoint_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "finetune" / "jepa_full_finetune.pt"


def load_jepa_encoder_state(checkpoint_path: Path) -> dict[str, Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint.get("model_state", checkpoint)
    encoder_state = {
        key.removeprefix("context_encoder."): value
        for key, value in model_state.items()
        if key.startswith("context_encoder.")
    }
    if not encoder_state:
        encoder_state = model_state
    return encoder_state


def supervised_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    return correct / max(1, total)


def train_supervised_vit(args: argparse.Namespace, num_classes: int) -> Path:
    checkpoint_path = supervised_checkpoint_path(args)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    image_size = 224
    train_loader = build_loader(
        args.data_root / args.train_folder,
        train_transform(image_size),
        args.supervised_batch_size,
        args.num_workers,
        args.device,
        shuffle=True,
    )
    val_loader = build_loader(
        args.data_root / args.val_folder,
        eval_transform(image_size),
        args.batch_size,
        args.num_workers,
        args.device,
    )

    torch.manual_seed(args.seed)
    model = SupervisedVitClassifier(args.jepa_config, num_classes, args.seed).to(
        args.device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.supervised_lr,
        weight_decay=args.supervised_weight_decay,
    )
    autocast_enabled = args.amp and args.device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_enabled)

    start_epoch = 0
    if checkpoint_path.exists() and not args.refresh_supervised:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        completed_epochs = int(checkpoint.get("epochs", 0))
        if completed_epochs >= args.supervised_epochs:
            return checkpoint_path
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = completed_epochs
        print(
            f"resuming supervised baseline from epoch {start_epoch}",
            flush=True,
        )

    for epoch in range(start_epoch, args.supervised_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for images, labels in train_loader:
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=autocast_enabled):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().cpu()) * images.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_seen += int(labels.numel())

        train_acc = total_correct / max(1, total_seen)
        val_acc = supervised_accuracy(model, val_loader, args.device)
        print(
            f"supervised epoch {epoch}: "
            f"loss={total_loss / max(1, total_seen):.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "encoder_state": model.encoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "num_classes": num_classes,
                "epochs": epoch + 1,
                "seed": args.seed,
                "lr": args.supervised_lr,
                "weight_decay": args.supervised_weight_decay,
                "last_train_acc": train_acc,
                "last_val_acc": val_acc,
                "best_val_acc": max(
                    val_acc,
                    float(
                        torch.load(checkpoint_path, map_location="cpu").get(
                            "best_val_acc", 0.0
                        )
                    )
                    if checkpoint_path.exists()
                    else val_acc,
                ),
            },
            checkpoint_path,
        )

    return checkpoint_path


def train_jepa_finetune_vit(args: argparse.Namespace, num_classes: int) -> Path:
    checkpoint_path = jepa_finetune_checkpoint_path(args)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader = build_loader(
        args.data_root / args.train_folder,
        train_transform(224),
        args.fine_tune_batch_size,
        args.num_workers,
        args.device,
        shuffle=True,
    )
    val_loader = build_loader(
        args.data_root / args.val_folder,
        eval_transform(224),
        args.batch_size,
        args.num_workers,
        args.device,
    )

    torch.manual_seed(args.seed)
    model = SupervisedVitClassifier(args.jepa_config, num_classes, args.seed).to(
        args.device
    )
    missing, unexpected = model.encoder.load_state_dict(
        load_jepa_encoder_state(args.jepa_checkpoint), strict=False
    )
    if unexpected:
        raise RuntimeError(f"Unexpected JEPA fine-tune encoder keys: {unexpected}")
    if missing:
        raise RuntimeError(f"Missing JEPA fine-tune encoder keys: {missing}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.fine_tune_lr,
        weight_decay=args.fine_tune_weight_decay,
    )
    autocast_enabled = args.amp and args.device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_enabled)

    start_epoch = 0
    if checkpoint_path.exists() and not args.refresh_supervised:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        completed_epochs = int(checkpoint.get("epochs", 0))
        if completed_epochs >= args.fine_tune_epochs:
            return checkpoint_path
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = completed_epochs
        print(f"resuming JEPA fine-tune from epoch {start_epoch}", flush=True)

    best_val = 0.0
    if checkpoint_path.exists():
        best_val = float(
            torch.load(checkpoint_path, map_location="cpu").get("best_val_acc", 0.0)
        )

    for epoch in range(start_epoch, args.fine_tune_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for images, labels in train_loader:
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=autocast_enabled):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().cpu()) * images.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_seen += int(labels.numel())

        train_acc = total_correct / max(1, total_seen)
        val_acc = supervised_accuracy(model, val_loader, args.device)
        best_val = max(best_val, val_acc)
        print(
            f"jepa fine-tune epoch {epoch}: "
            f"loss={total_loss / max(1, total_seen):.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "encoder_state": model.encoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "num_classes": num_classes,
                "epochs": epoch + 1,
                "seed": args.seed,
                "lr": args.fine_tune_lr,
                "weight_decay": args.fine_tune_weight_decay,
                "last_train_acc": train_acc,
                "last_val_acc": val_acc,
                "best_val_acc": best_val,
            },
            checkpoint_path,
        )

    return checkpoint_path


def build_supervised_feature_model(
    config_path: Path,
    checkpoint_path: Path,
    seed: int,
) -> nn.Module:
    config = load_config(config_path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = build_encoder(config.get("model", {}).get("encoder", {}))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder_state = checkpoint.get("encoder_state")
    if encoder_state is None:
        model_state = checkpoint.get("model_state", checkpoint)
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in model_state.items()
            if key.startswith("encoder.")
        }
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected supervised encoder keys: {unexpected}")
    if missing:
        raise RuntimeError(f"Missing supervised encoder keys: {missing}")
    return JepaFeatureExtractor(encoder)


def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    use_amp: bool,
) -> tuple[Tensor, Tensor]:
    features: list[Tensor] = []
    labels: list[Tensor] = []
    model.to(device).eval()
    autocast_enabled = use_amp and device.startswith("cuda")
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=autocast_enabled):
                batch_features = model(images)
            features.append(batch_features.float().cpu())
            labels.append(batch_labels.cpu())
    return torch.cat(features), torch.cat(labels)


def load_or_extract_features(
    model: nn.Module,
    model_key: str,
    split: str,
    loader: DataLoader,
    output_dir: Path,
    device: str,
    use_amp: bool,
    refresh: bool,
) -> tuple[Tensor, Tensor]:
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    cache_path = feature_dir / f"{model_key}_{split}.pt"
    if cache_path.exists() and not refresh:
        cache = torch.load(cache_path, map_location="cpu")
        return cache["features"], cache["labels"]

    features, labels = extract_features(model, loader, device, use_amp)
    torch.save(
        {
            "features": features,
            "labels": labels,
            "classes": list(loader.dataset.classes),
            "class_to_idx": dict(loader.dataset.class_to_idx),
        },
        cache_path,
    )
    return features, labels


def linear_probe(
    train_x: Tensor,
    train_y: Tensor,
    val_x: Tensor,
    val_y: Tensor,
    num_classes: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    classifier = nn.Linear(train_x.size(1), num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = classifier(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        print(
            f"linear epoch {epoch}: loss={total_loss / max(1, len(loader)):.4f}",
            flush=True,
        )
    return accuracy(classifier, train_x, train_y, device), accuracy(
        classifier, val_x, val_y, device
    )


def accuracy(
    classifier: nn.Module, features: Tensor, labels: Tensor, device: str
) -> float:
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, features.size(0), 2048):
            batch_x = features[start : start + 2048].to(device)
            batch_y = labels[start : start + 2048].to(device)
            logits = classifier(batch_x)
            correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            total += int(batch_y.numel())
    return correct / max(1, total)


def checkpoint_epoch(path: Path) -> int | None:
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu")
    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    return int(epoch) if epoch is not None else None


def safe_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    if Path(path).exists():
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_comparison(summary: dict[str, Any], output: Path) -> None:
    results = summary["results"]
    height = 260 + 92 * len(results)
    canvas = Image.new("RGB", (1400, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (40, 32),
        "ImageNet-50 frozen linear probe",
        fill=(31, 35, 40),
        font=font(32, True),
    )
    draw.text(
        (42, 78),
        f"{summary['train_images']} train / {summary['val_images']} val / "
        f"{summary['num_classes']} classes",
        fill=(80, 86, 94),
        font=font(20),
    )

    x0, y0, bar_w, bar_h = 660, 150, 560, 36
    colors = [(36, 112, 166), (47, 133, 90)]
    for i, result in enumerate(results):
        y = y0 + i * 92
        draw.text((60, y + 4), result["name"], fill=(31, 35, 40), font=font(20, True))
        draw.rectangle((x0, y, x0 + bar_w, y + bar_h), fill=(235, 238, 242))
        fill_w = int(bar_w * min(max(result["linear_val_acc"], 0.0), 1.0))
        draw.rectangle((x0, y, x0 + fill_w, y + bar_h), fill=colors[i % len(colors)])
        draw.text(
            (x0 + bar_w + 24, y + 4),
            f"{result['linear_val_acc'] * 100:.2f}%",
            fill=(31, 35, 40),
            font=font(20),
        )
        summary_text = (
            f"train {result['linear_train_acc'] * 100:.2f}% | "
            f"dim {result['feature_dim']}"
        )
        draw.text(
            (60, y + 42),
            summary_text,
            fill=(80, 86, 94),
            font=font(16),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def evaluate_model(args: argparse.Namespace, model_name: str) -> dict[str, Any]:
    image_size = 224
    transform = eval_transform(image_size)
    train_loader = build_loader(
        args.data_root / args.train_folder,
        transform,
        args.batch_size,
        args.num_workers,
        args.device,
    )
    val_loader = build_loader(
        args.data_root / args.val_folder,
        transform,
        args.batch_size,
        args.num_workers,
        args.device,
    )
    num_classes = len(train_loader.dataset.classes)

    if model_name == "random-frozen-vit-b16":
        model = build_random_feature_model(args.jepa_config, args.seed)
        result_name = "random_frozen_vit_b16_linear_probe"
        model_key = safe_key(f"random_custom_vit_b16_seed{args.seed}")
    elif model_name == "jepa-checkpoint":
        if not args.jepa_checkpoint.exists():
            raise FileNotFoundError(
                f"JEPA checkpoint not found: {args.jepa_checkpoint}"
            )
        model = build_jepa_feature_model(args.jepa_config, args.jepa_checkpoint)
        result_name = "jepa_linear_probe"
        model_key = safe_key(f"jepa_{args.jepa_checkpoint.stem}")
    elif model_name == "jepa-full-finetune":
        checkpoint = train_jepa_finetune_vit(args, num_classes)
        model = build_supervised_feature_model(args.jepa_config, checkpoint, args.seed)
        result_name = "jepa_full_finetune_linear_probe"
        model_key = safe_key(f"jepa_full_finetune_{checkpoint.stem}")
    elif model_name == "supervised-vit-b16-scratch":
        checkpoint = train_supervised_vit(args, num_classes)
        model = build_supervised_feature_model(args.jepa_config, checkpoint, args.seed)
        result_name = "supervised_scratch_vit_b16_linear_probe"
        model_key = safe_key(f"supervised_custom_vit_b16_{checkpoint.stem}")
    elif model_name == "torchvision-vit-b16":
        model = build_torchvision_feature_model(args.torchvision_weights)
        result_name = "vit_b16_imagenet_linear_probe"
        model_key = safe_key(f"vit_b16_{args.torchvision_weights}")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"extracting {model_name} train features", flush=True)
    train_x, train_y = load_or_extract_features(
        model,
        model_key,
        "train",
        train_loader,
        args.output_dir,
        args.device,
        args.amp,
        args.refresh_features,
    )
    print(f"extracting {model_name} val features", flush=True)
    val_x, val_y = load_or_extract_features(
        model,
        model_key,
        "val",
        val_loader,
        args.output_dir,
        args.device,
        args.amp,
        args.refresh_features,
    )
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    print(f"running {model_name} linear probe", flush=True)
    linear_train, linear_val = linear_probe(
        train_x,
        train_y,
        val_x,
        val_y,
        num_classes=num_classes,
        epochs=args.linear_epochs,
        lr=args.linear_lr,
        batch_size=args.linear_batch_size,
        device=args.device,
        seed=args.seed,
    )
    result: dict[str, Any] = {
        "name": result_name,
        "model": model_name,
        "feature_dim": int(train_x.size(1)),
        "linear_train_acc": linear_train,
        "linear_val_acc": linear_val,
    }
    if model_name == "jepa-checkpoint":
        result["checkpoint_path"] = str(args.jepa_checkpoint)
        result["checkpoint_epoch"] = checkpoint_epoch(args.jepa_checkpoint)
    if model_name == "jepa-full-finetune":
        checkpoint_path = jepa_finetune_checkpoint_path(args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        result["checkpoint_path"] = str(checkpoint_path)
        result["fine_tune_epochs"] = args.fine_tune_epochs
        result["fine_tune_lr"] = args.fine_tune_lr
        result["full_finetune_last_val_acc"] = checkpoint.get("last_val_acc")
        result["full_finetune_best_val_acc"] = checkpoint.get("best_val_acc")
    if model_name == "supervised-vit-b16-scratch":
        checkpoint_path = supervised_checkpoint_path(args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        result["checkpoint_path"] = str(checkpoint_path)
        result["supervised_epochs"] = args.supervised_epochs
        result["supervised_lr"] = args.supervised_lr
        result["full_finetune_last_val_acc"] = checkpoint.get("last_val_acc")
        result["full_finetune_best_val_acc"] = checkpoint.get("best_val_acc")
    if model_name == "torchvision-vit-b16":
        result["weights"] = args.torchvision_weights
    return result


def main() -> None:
    args = parse_args()
    models_to_run = args.models or [
        "random-frozen-vit-b16",
        "jepa-checkpoint",
        "jepa-full-finetune",
        "supervised-vit-b16-scratch",
        "torchvision-vit-b16",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.ImageFolder(args.data_root / args.train_folder)
    val_dataset = datasets.ImageFolder(args.data_root / args.val_folder)
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Train and val class folders do not match.")

    results = [evaluate_model(args, model_name) for model_name in models_to_run]
    summary = {
        "dataset": str(args.data_root),
        "train_folder": args.train_folder,
        "val_folder": args.val_folder,
        "num_classes": len(train_dataset.classes),
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "linear_epochs": args.linear_epochs,
        "linear_lr": args.linear_lr,
        "linear_batch_size": args.linear_batch_size,
        "seed": args.seed,
        "results": results,
    }
    comparison_path = args.output_dir / "comparison.json"
    figure_path = args.output_dir / "comparison.png"
    comparison_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    draw_comparison(summary, figure_path)
    print(comparison_path)
    print(figure_path)
    print(summary)


if __name__ == "__main__":
    main()
