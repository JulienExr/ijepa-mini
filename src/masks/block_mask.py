from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


@dataclass(frozen=True)
class BlockMaskConfig:
    """Configuration for I-JEPA block masking."""

    image_size: int = 224
    patch_size: int = 16
    allow_overlap: bool = False
    num_context_masks: int = 1
    num_target_masks: int = 4
    context_scale: tuple[float, float] = (0.85, 1.0)
    target_scale: tuple[float, float] = (0.15, 0.2)
    aspect_ratio: tuple[float, float] = (0.75, 1.5)
    min_context_patches: int = 4

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size * self.grid_size


@dataclass(frozen=True)
class MaskBatch:
    """Masks sampled for one batch."""

    context_masks: list[Tensor]
    target_masks: list[Tensor]


class BlockMaskSampler:
    """Sample rectangular context and target blocks over patch indices."""

    def __init__(self, config: BlockMaskConfig) -> None:
        self.config = config

    def sample_context_masks(self, batch_size: int) -> list[Tensor]:
        raise NotImplementedError("Context block mask sampling is not implemented.")

    def sample_target_masks(self, batch_size: int) -> list[Tensor]:
        raise NotImplementedError("Target block mask sampling is not implemented.")

    def sample_block(self, batch_size: int, scale: tuple[float, float]) -> Tensor:
        raise NotImplementedError("Single block sampling is not implemented.")


class BlockMaskCollator:
    """Collator compatible with a PyTorch DataLoader."""

    def __init__(self, config: BlockMaskConfig) -> None:
        self.config = config
        self.sampler = BlockMaskSampler(config)

    def __call__(self, images: list[Tensor]) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        raise NotImplementedError("Mask collation is not implemented.")

    def step(self) -> None:
        """Hook for mask schedules across epochs/iterations."""
        raise NotImplementedError("Mask scheduler step is not implemented.")


def build_mask_collator(config: dict | BlockMaskConfig) -> BlockMaskCollator:
    if isinstance(config, dict):
        config = BlockMaskConfig(**config)
    return BlockMaskCollator(config)
