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
class EncoderConfig:
    """Configuration for the context and target image encoders."""

    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_cls_token: bool = False


class PatchEmbedding(nn.Module):
    """Convert an image batch into a sequence of patch embeddings."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, images: Tensor) -> Tensor:
        raise NotImplementedError("Patch embedding forward pass is not implemented.")


class TransformerEncoderBlock(nn.Module):
    """Single ViT encoder block used by the I-JEPA encoder."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, tokens: Tensor) -> Tensor:
        raise NotImplementedError("Transformer encoder block forward is not implemented.")


class VisionTransformerEncoder(nn.Module):
    """Context or target encoder for I-JEPA."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, images: Tensor, masks: list[Tensor] | None = None) -> Tensor:
        """Encode images, optionally keeping only context patch tokens."""
        raise NotImplementedError("Encoder forward pass is not implemented.")

    def encode_patches(self, images: Tensor) -> Tensor:
        raise NotImplementedError("Patch-level encoding is not implemented.")

    def apply_context_masks(self, tokens: Tensor, masks: list[Tensor]) -> Tensor:
        raise NotImplementedError("Context mask application is not implemented.")


def build_encoder(config: dict[str, Any] | EncoderConfig) -> VisionTransformerEncoder:
    """Factory used by training and evaluation code."""
    if isinstance(config, dict):
        config = EncoderConfig(**config)
    return VisionTransformerEncoder(config)
