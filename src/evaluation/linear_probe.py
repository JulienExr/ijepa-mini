from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor, nn
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader
else:
    Tensor = Any
    DataLoader = Any
    Optimizer = Any

    class _Module:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    class _NN:
        Module = _Module

    nn = _NN()


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

    def forward(self, features: Tensor) -> Tensor:
        raise NotImplementedError("Linear classifier forward pass is not implemented.")


class LinearProbeEvaluator:
    """Train and evaluate a linear probe on frozen I-JEPA features."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.probe: LinearClassifier | None = None
        self.optimizer: Optimizer | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None

    def setup(self) -> None:
        raise NotImplementedError("Linear probe setup is not implemented.")

    def load_encoder(self) -> nn.Module:
        raise NotImplementedError("Encoder checkpoint loading is not implemented.")

    def train(self) -> None:
        raise NotImplementedError("Linear probe training is not implemented.")

    def train_epoch(self, epoch: int) -> dict[str, float]:
        raise NotImplementedError("Linear probe epoch is not implemented.")

    def evaluate(self) -> dict[str, float]:
        raise NotImplementedError("Linear probe evaluation is not implemented.")


def main(config: dict[str, Any]) -> None:
    """Run linear probing evaluation."""
    evaluator = LinearProbeEvaluator(config)
    evaluator.setup()
    evaluator.train()
    evaluator.evaluate()
