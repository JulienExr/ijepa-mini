from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

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
        context_tokens = self.context_encoder(images, context_masks)
        return self.predictor(context_tokens, context_masks, target_masks)

    def forward_target(self, images: Tensor, target_masks: list[Tensor]) -> Tensor:
        with torch.no_grad():
            target_tokens = self.target_encoder(images)

        gathered_targets = []
        for mask in target_masks:
            mask = mask.to(device=target_tokens.device, dtype=torch.long)
            idx = mask.unsqueeze(-1).expand(-1, -1, target_tokens.size(-1))
            gathered_targets.append(target_tokens.gather(dim=1, index=idx))
        return torch.cat(gathered_targets, dim=1)

    def forward(
        self,
        images: Tensor,
        context_masks: list[Tensor],
        target_masks: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        predictions = self.forward_context(images, context_masks, target_masks)
        targets = self.forward_target(images, target_masks)
        if predictions.size(0) != targets.size(0):
            if predictions.size(0) % targets.size(0) != 0:
                raise ValueError(
                    "Prediction and target batch sizes are incompatible: "
                    f"{predictions.size(0)} vs {targets.size(0)}"
                )
            targets = targets.repeat(predictions.size(0) // targets.size(0), 1, 1)
        return predictions, targets


def build_ijepa(config: dict[str, Any] | IJEPAConfig) -> IJEPA:
    """Factory for the full I-JEPA model."""
    if isinstance(config, dict):
        encoder_config = EncoderConfig(**config.get("encoder", {}))
        predictor_kwargs = {
            "image_size": encoder_config.image_size,
            "patch_size": encoder_config.patch_size,
            **config.get("predictor", {}),
        }
        predictor_config = PredictorConfig(**predictor_kwargs)
        config = IJEPAConfig(encoder=encoder_config, predictor=predictor_config)
    return IJEPA(config)
