from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EMAConfig:
    """Momentum schedule for the target encoder."""

    start: float = 0.996
    end: float = 1.0
    total_steps: int = 100_000


class EMAScheduler:
    """Yield momentum values between ``start`` and ``end``."""

    def __init__(self, config: EMAConfig) -> None:
        self.config = config
        self.step_index = 0

    def __iter__(self) -> Iterator[float]:
        return self

    def __next__(self) -> float:
        total = max(self.config.total_steps, 1)
        t = min(self.step_index, self.config.total_steps)
        span = self.config.end - self.config.start
        momentum = self.config.start + span * (t / total)
        self.step_index += 1
        return momentum

    def state_dict(self) -> dict[str, int]:
        return {"step_index": self.step_index}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.step_index = state["step_index"]


@torch.no_grad()
def update_ema(source: nn.Module, target: nn.Module, momentum: float) -> None:
    """Update target encoder parameters from context encoder parameters."""
    for p_src, p_tgt in zip(source.parameters(), target.parameters(), strict=True):
        p_tgt.mul_(momentum).add_(p_src.detach(), alpha=1.0 - momentum)


def copy_model_for_ema(model: nn.Module) -> nn.Module:
    """Create a frozen target model initialized from a source model."""
    target = copy.deepcopy(model)
    for parameter in target.parameters():
        parameter.requires_grad = False
    target.eval()
    return target
