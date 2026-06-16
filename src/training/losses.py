from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
        if predictions.shape != targets.shape:
            raise ValueError(
                "predictions and targets must have the same shape, got "
                f"{tuple(predictions.shape)} and {tuple(targets.shape)}"
            )

        if self.config.normalize_targets:
            targets = normalize_targets(targets)

        if self.config.name == "smooth_l1":
            return smooth_l1_latent_loss(predictions, targets, beta=self.config.beta)
        if self.config.name == "mse":
            return F.mse_loss(predictions, targets)
        raise ValueError(f"Unsupported loss: {self.config.name}")


def normalize_targets(targets: Tensor) -> Tensor:
    mean = targets.mean(dim=-1, keepdim=True)
    variance = targets.var(dim=-1, keepdim=True, unbiased=False)
    return (targets - mean) * torch.rsqrt(variance + 1e-6)


def smooth_l1_latent_loss(
    predictions: Tensor,
    targets: Tensor,
    beta: float = 1.0,
) -> Tensor:
    return F.smooth_l1_loss(predictions, targets.detach(), beta=beta)


def build_loss(config: dict | LossConfig) -> IJEPALoss:
    if isinstance(config, dict):
        config = LossConfig(**config)
    return IJEPALoss(config)
