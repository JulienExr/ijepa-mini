from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor, nn
    from torch.utils.data import DataLoader
else:
    Tensor = Any
    DataLoader = Any

    class _Module:
        pass

    class _NN:
        Module = _Module

    nn = _NN()


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
        raise NotImplementedError("Feature bank construction is not implemented.")

    def query(self, features: Tensor, k: int, temperature: float) -> Tensor:
        raise NotImplementedError("Feature bank query is not implemented.")


class KNNEvaluator:
    """Evaluate frozen I-JEPA features with weighted k-NN."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.encoder: nn.Module | None = None
        self.feature_bank = FeatureBank()
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None

    def setup(self) -> None:
        raise NotImplementedError("k-NN evaluator setup is not implemented.")

    def load_encoder(self) -> nn.Module:
        raise NotImplementedError("Encoder checkpoint loading is not implemented.")

    def evaluate(self) -> dict[str, float]:
        raise NotImplementedError("k-NN evaluation is not implemented.")


def main(config: dict[str, Any]) -> None:
    """Run k-NN evaluation."""
    evaluator = KNNEvaluator(config)
    evaluator.setup()
    evaluator.evaluate()
