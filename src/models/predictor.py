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

        def parameters(self) -> list[Any]:
            return []

    class _NN:
        Module = _Module

    nn = _NN()


@dataclass(frozen=True)
class PredictorConfig:
    """Configuration for the latent-space predictor."""

    embed_dim: int = 192
    predictor_embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0


class PredictorBlock(nn.Module):
    """Transformer block used inside the I-JEPA predictor."""

    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, tokens: Tensor) -> Tensor:
        raise NotImplementedError("Predictor block forward is not implemented.")


class IJEPAPredictor(nn.Module):
    """Predict target patch representations from context representations."""

    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        context_tokens: Tensor,
        context_masks: list[Tensor],
        target_masks: list[Tensor],
    ) -> Tensor:
        raise NotImplementedError("Predictor forward pass is not implemented.")

    def add_mask_tokens(self, context_tokens: Tensor, target_masks: list[Tensor]) -> Tensor:
        raise NotImplementedError("Target mask token insertion is not implemented.")


def build_predictor(config: dict[str, Any] | PredictorConfig) -> IJEPAPredictor:
    """Factory used by the training loop."""
    if isinstance(config, dict):
        config = PredictorConfig(**config)
    return IJEPAPredictor(config)
