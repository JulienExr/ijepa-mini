from __future__ import annotations

import copy
from dataclasses import dataclass
from dataclasses import field
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

from src.models.encoder import EncoderConfig, VisionTransformerEncoder
from src.models.predictor import IJEPAPredictor, PredictorConfig


@dataclass(frozen=True)
class IJEPAConfig:
    """Top-level model configuration."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)


class IJEPA(nn.Module):
    """Container for context encoder, target encoder, and predictor."""

    def __init__(self, config: IJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = VisionTransformerEncoder(config.encoder)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.predictor = IJEPAPredictor(config.predictor)
        self.freeze_target_encoder()

    def freeze_target_encoder(self) -> None:
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

    def forward_context(
        self,
        images: Tensor,
        context_masks: list[Tensor],
        target_masks: list[Tensor],
    ) -> Tensor:
        raise NotImplementedError("Context/predictor forward pass is not implemented.")

    def forward_target(self, images: Tensor, target_masks: list[Tensor]) -> Tensor:
        raise NotImplementedError("Target encoder forward pass is not implemented.")

    def forward(
        self,
        images: Tensor,
        context_masks: list[Tensor],
        target_masks: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        predictions = self.forward_context(images, context_masks, target_masks)
        targets = self.forward_target(images, target_masks)
        return predictions, targets


def build_ijepa(config: dict[str, Any] | IJEPAConfig) -> IJEPA:
    """Factory for the full I-JEPA model."""
    if isinstance(config, dict):
        encoder_config = EncoderConfig(**config.get("encoder", {}))
        predictor_config = PredictorConfig(**config.get("predictor", {}))
        config = IJEPAConfig(encoder=encoder_config, predictor=predictor_config)
    return IJEPA(config)
