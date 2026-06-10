from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from torch import nn
else:
    class _Module:
        def parameters(self) -> list[Any]:
            return []

    class _NN:
        Module = _Module

    nn = _NN()


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
        raise NotImplementedError("EMA momentum scheduling is not implemented.")

    def state_dict(self) -> dict[str, int]:
        return {"step_index": self.step_index}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.step_index = state["step_index"]


def update_ema(source: nn.Module, target: nn.Module, momentum: float) -> None:
    """Update target encoder parameters from context encoder parameters."""
    raise NotImplementedError("EMA parameter update is not implemented.")


def copy_model_for_ema(model: nn.Module) -> nn.Module:
    """Create a frozen target model initialized from a source model."""
    raise NotImplementedError("EMA target model copy is not implemented.")
