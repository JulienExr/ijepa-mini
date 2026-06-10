from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor, nn
else:
    Tensor = Any

    class _Module:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    class _NN:
        Module = _Module

    nn = _NN()


@dataclass(frozen=True)
class LossConfig:
    """Configuration for I-JEPA latent prediction loss."""

    name: str = "smooth_l1"
    beta: float = 1.0
    normalize_targets: bool = True


class IJEPALoss(nn.Module):
    """Compute the loss between predicted and target patch representations."""

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError("I-JEPA loss computation is not implemented.")


def normalize_targets(targets: Tensor) -> Tensor:
    raise NotImplementedError("Target normalization is not implemented.")


def smooth_l1_latent_loss(predictions: Tensor, targets: Tensor, beta: float = 1.0) -> Tensor:
    raise NotImplementedError("Smooth L1 latent loss is not implemented.")


def build_loss(config: dict | LossConfig) -> IJEPALoss:
    if isinstance(config, dict):
        config = LossConfig(**config)
    return IJEPALoss(config)
