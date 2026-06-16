from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.models.encoder import build_encoder


@dataclass(frozen=True)
class KNNConfig:
    """Configuration for k-NN evaluation on frozen features."""

    checkpoint_path: str
    k: int = 20
    temperature: float = 0.07
    batch_size: int = 256
    normalize_features: bool = True


class FeatureBank:
    """Store train-set features and labels for k-NN classification."""

    def __init__(self) -> None:
        self.features: Tensor | None = None
        self.labels: Tensor | None = None

    def build(self, encoder: nn.Module, loader: DataLoader) -> None:
        features = []
        labels = []
        device = next(encoder.parameters()).device
        encoder.eval()
        with torch.no_grad():
            for images, batch_labels in tqdm(loader, desc="Feature bank", leave=False):
                images = images.to(device)
                batch_features = encoder(images).mean(dim=1)
                features.append(F.normalize(batch_features.cpu(), dim=1))
                labels.append(batch_labels.cpu())
        self.features = torch.cat(features, dim=0)
        self.labels = torch.cat(labels, dim=0)

    def query(self, features: Tensor, k: int, temperature: float) -> Tensor:
        if self.features is None or self.labels is None:
            raise RuntimeError("Feature bank has not been built")

        k = min(k, self.features.size(0))
        features = F.normalize(features.cpu(), dim=1)
        similarities = features @ self.features.T
        scores, indices = similarities.topk(k=k, dim=1)
        neighbor_labels = self.labels[indices]
        weights = (scores / temperature).softmax(dim=1)

        num_classes = int(self.labels.max().item()) + 1
        votes = torch.zeros(features.size(0), num_classes)
        votes.scatter_add_(1, neighbor_labels, weights)
        return votes


class KNNEvaluator:
    """Evaluate frozen I-JEPA features with weighted k-NN."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.knn_config: KNNConfig | None = None
        self.encoder: nn.Module | None = None
        self.feature_bank = FeatureBank()
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.device = str(config.get("runtime", {}).get("device", "cpu"))

    def setup(self) -> None:
        self.knn_config = self._build_knn_config()
        self.encoder = self.load_encoder().to(self.device)
        self.train_loader = self._build_loader("train")
        self.val_loader = self._build_loader("val")
        self.feature_bank.build(self.encoder, self.train_loader)

    def load_encoder(self) -> nn.Module:
        assert self.knn_config is not None
        encoder = build_encoder(self.config.get("model", {}).get("encoder", {}))
        checkpoint = torch.load(self.knn_config.checkpoint_path, map_location="cpu")
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

    def evaluate(self) -> dict[str, float]:
        assert self.encoder is not None
        assert self.knn_config is not None
        assert self.val_loader is not None

        total_correct = 0
        total_seen = 0
        self.encoder.eval()
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="k-NN eval", leave=False):
                images = images.to(self.device)
                features = self.encoder(images).mean(dim=1)
                predictions = self.feature_bank.query(
                    features,
                    k=self.knn_config.k,
                    temperature=self.knn_config.temperature,
                )
                total_correct += int((predictions.argmax(dim=1) == labels).sum().item())
                total_seen += images.size(0)

        metrics = {"accuracy": total_correct / max(1, total_seen)}
        tqdm.write(f"k-NN eval: {metrics}")
        return metrics

    def _build_knn_config(self) -> KNNConfig:
        raw = self.config.get("evaluation", {}).get("knn", self.config.get("knn", {}))
        missing = {"checkpoint_path"} - set(raw)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"k-NN config missing required field(s): {names}")
        return KNNConfig(**raw)

    def _build_loader(self, split: str) -> DataLoader:
        assert self.knn_config is not None
        data_config = self.config.get("data", {})
        root = Path(data_config.get("root_path", "data"))
        folder = data_config.get(f"{split}_folder", split)
        dataset = datasets.ImageFolder(root / folder, transform=self._eval_transform())
        return DataLoader(
            dataset,
            batch_size=self.knn_config.batch_size,
            shuffle=False,
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
    """Run k-NN evaluation."""
    evaluator = KNNEvaluator(config)
    evaluator.setup()
    evaluator.evaluate()
