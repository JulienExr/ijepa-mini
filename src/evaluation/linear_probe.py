from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.models.encoder import build_encoder


@dataclass(frozen=True)
class LinearProbeConfig:
    """Configuration for supervised linear probing."""

    checkpoint_path: str
    num_classes: int
    epochs: int = 50
    lr: float = 0.1
    weight_decay: float = 0.0
    batch_size: int = 256


class LinearClassifier(nn.Module):
    """Linear classifier trained on frozen encoder features."""

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, features: Tensor) -> Tensor:
        return self.fc(features)


class LinearProbeEvaluator:
    """Train and evaluate a linear probe on frozen I-JEPA features."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.probe_config: LinearProbeConfig | None = None
        self.encoder: nn.Module | None = None
        self.probe: LinearClassifier | None = None
        self.optimizer: Optimizer | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.device = str(config.get("runtime", {}).get("device", "cpu"))

    def setup(self) -> None:
        probe_config = self._build_probe_config()
        self.probe_config = probe_config
        self.encoder = self.load_encoder().to(self.device)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        feature_dim = self.encoder.config.embed_dim
        self.probe = LinearClassifier(feature_dim, probe_config.num_classes).to(
            self.device
        )
        self.optimizer = torch.optim.SGD(
            self.probe.parameters(),
            lr=probe_config.lr,
            momentum=0.9,
            weight_decay=probe_config.weight_decay,
        )
        self.train_loader = self._build_loader("train", shuffle=True)
        self.val_loader = self._build_loader("val", shuffle=False)

    def load_encoder(self) -> nn.Module:
        assert self.probe_config is not None
        encoder = build_encoder(self.config.get("model", {}).get("encoder", {}))
        checkpoint = torch.load(self.probe_config.checkpoint_path, map_location="cpu")
        model_state = checkpoint.get("model_state", checkpoint)
        encoder_state = {
            key.removeprefix("context_encoder."): value
            for key, value in model_state.items()
            if key.startswith("context_encoder.")
        }
        if not encoder_state:
            encoder_state = model_state
        encoder.load_state_dict(encoder_state, strict=False)
        return encoder

    def train(self) -> None:
        assert self.probe_config is not None
        for epoch in range(self.probe_config.epochs):
            metrics = self.train_epoch(epoch)
            tqdm.write(f"Linear probe epoch {epoch}: {metrics}")

    def train_epoch(self, epoch: int) -> dict[str, float]:
        assert self.encoder is not None
        assert self.probe is not None
        assert self.optimizer is not None
        assert self.train_loader is not None

        self.probe.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for images, labels in tqdm(
            self.train_loader,
            desc=f"Probe {epoch}",
            leave=False,
        ):
            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                features = self.encoder(images).mean(dim=1)
            logits = self.probe(features)
            loss = nn.functional.cross_entropy(logits, labels)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.detach().cpu()) * images.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_seen += images.size(0)

        return {
            "loss": total_loss / max(1, total_seen),
            "accuracy": total_correct / max(1, total_seen),
        }

    def evaluate(self) -> dict[str, float]:
        assert self.encoder is not None
        assert self.probe is not None
        assert self.val_loader is not None

        self.probe.eval()
        total_correct = 0
        total_seen = 0
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Probe eval", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                features = self.encoder(images).mean(dim=1)
                logits = self.probe(features)
                total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                total_seen += images.size(0)

        metrics = {"accuracy": total_correct / max(1, total_seen)}
        tqdm.write(f"Linear probe eval: {metrics}")
        return metrics

    def _build_probe_config(self) -> LinearProbeConfig:
        raw = self.config.get("evaluation", {}).get(
            "linear_probe",
            self.config.get("linear_probe", {}),
        )
        missing = {"checkpoint_path", "num_classes"} - set(raw)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"linear probe config missing required field(s): {names}")
        return LinearProbeConfig(**raw)

    def _build_loader(self, split: str, shuffle: bool) -> DataLoader:
        assert self.probe_config is not None
        data_config = self.config.get("data", {})
        root = Path(data_config.get("root_path", "data"))
        folder = data_config.get(f"{split}_folder", split)
        dataset = datasets.ImageFolder(root / folder, transform=self._eval_transform())
        return DataLoader(
            dataset,
            batch_size=self.probe_config.batch_size,
            shuffle=shuffle,
            num_workers=int(data_config.get("num_workers", 4)),
            pin_memory=bool(data_config.get("pin_memory", True)),
        )

    def _eval_transform(self) -> transforms.Compose:
        image_size = int(self.config.get("data", {}).get("image_size", 224))
        return transforms.Compose(
            [
                transforms.Resize(image_size + 32, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


def main(config: dict[str, Any]) -> None:
    """Run linear probing evaluation."""
    evaluator = LinearProbeEvaluator(config)
    evaluator.setup()
    evaluator.train()
    evaluator.evaluate()
