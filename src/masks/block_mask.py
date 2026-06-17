from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import default_collate


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

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.num_context_masks <= 0 or self.num_target_masks <= 0:
            raise ValueError("At least one context and target mask are required")
        if not 1 <= self.min_context_patches <= self.num_patches:
            raise ValueError("min_context_patches is outside the patch grid")

        self._validate_range("context_scale", self.context_scale, upper_bound=1.0)
        self._validate_range("target_scale", self.target_scale, upper_bound=1.0)
        self._validate_range("aspect_ratio", self.aspect_ratio)

    @staticmethod
    def _validate_range(
        name: str,
        values: tuple[float, float],
        upper_bound: float | None = None,
    ) -> None:
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values")
        lower, upper = values
        if lower <= 0 or lower > upper:
            raise ValueError(f"Invalid {name}: {values}")
        if upper_bound is not None and upper > upper_bound:
            raise ValueError(f"{name} cannot exceed {upper_bound}")

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

    max_sampling_attempts = 100

    def __init__(self, config: BlockMaskConfig) -> None:
        self.config = config

    def sample_context_masks(self, batch_size: int) -> list[Tensor]:
        self._validate_batch_size(batch_size)
        block_size = self._sample_block_size(
            self.config.context_scale,
            aspect_ratio=(1.0, 1.0),
        )
        return [
            self._sample_block_at_random_locations(batch_size, block_size)
            for _ in range(self.config.num_context_masks)
        ]

    def sample_target_masks(self, batch_size: int) -> list[Tensor]:
        self._validate_batch_size(batch_size)
        block_size = self._sample_block_size(
            self.config.target_scale,
            aspect_ratio=self.config.aspect_ratio,
        )
        return [
            self._sample_block_at_random_locations(batch_size, block_size)
            for _ in range(self.config.num_target_masks)
        ]

    def sample_block(self, batch_size: int, scale: tuple[float, float]) -> Tensor:
        self._validate_batch_size(batch_size)
        BlockMaskConfig._validate_range("scale", scale, upper_bound=1.0)
        block_size = self._sample_block_size(scale, self.config.aspect_ratio)
        return self._sample_block_at_random_locations(batch_size, block_size)

    def sample_masks(self, batch_size: int) -> MaskBatch:
        """Sample target blocks, then context blocks for the same batch."""
        target_masks = self.sample_target_masks(batch_size)
        context_masks = self._sample_context_masks(batch_size, target_masks)
        return MaskBatch(
            context_masks=context_masks,
            target_masks=target_masks,
        )

    def _sample_context_masks(
        self,
        batch_size: int,
        target_masks: list[Tensor],
    ) -> list[Tensor]:
        block_size = self._sample_block_size(
            self.config.context_scale,
            aspect_ratio=(1.0, 1.0),
        )
        if self.config.allow_overlap:
            return [
                self._sample_block_at_random_locations(batch_size, block_size)
                for _ in range(self.config.num_context_masks)
            ]

        context_masks = []
        for _ in range(self.config.num_context_masks):
            per_image_masks = [
                self._sample_context_for_image(
                    block_size,
                    [mask[batch_index] for mask in target_masks],
                )
                for batch_index in range(batch_size)
            ]
            min_keep = min(mask.numel() for mask in per_image_masks)
            context_masks.append(
                torch.stack([mask[:min_keep] for mask in per_image_masks])
            )
        return context_masks

    def _sample_context_for_image(
        self,
        block_size: tuple[int, int],
        target_masks: list[Tensor],
    ) -> Tensor:
        excluded = torch.zeros(self.config.num_patches, dtype=torch.bool)
        for target_mask in target_masks:
            excluded[target_mask] = True

        best_candidate = torch.empty(0, dtype=torch.long)
        for _ in range(self.max_sampling_attempts):
            candidate = self._sample_single_block(block_size)
            candidate = candidate[~excluded[candidate]]
            if candidate.numel() > best_candidate.numel():
                best_candidate = candidate
            if candidate.numel() >= self.config.min_context_patches:
                return candidate

        if best_candidate.numel() >= self.config.min_context_patches:
            return best_candidate

        fallback = torch.arange(self.config.num_patches, dtype=torch.long)
        fallback = fallback[~excluded]
        if fallback.numel() >= self.config.min_context_patches:
            return fallback

        raise RuntimeError(
            "Unable to sample enough context patches outside target blocks"
        )

    def _sample_block_size(
        self,
        scale: tuple[float, float],
        aspect_ratio: tuple[float, float],
    ) -> tuple[int, int]:
        scale_value = torch.empty(1).uniform_(float(scale[0]), float(scale[1])).item()
        log_ratio = (
            torch.empty(1)
            .uniform_(
                math.log(float(aspect_ratio[0])),
                math.log(float(aspect_ratio[1])),
            )
            .item()
        )
        ratio = math.exp(log_ratio)
        target_area = self.config.num_patches * scale_value

        height = round(math.sqrt(target_area * ratio))
        width = round(math.sqrt(target_area / ratio))
        height = min(self.config.grid_size, max(1, height))
        width = min(self.config.grid_size, max(1, width))

        max_area = max(1, min(self.config.num_patches, math.ceil(target_area)))
        while height * width > max_area:
            if height >= width and height > 1:
                height -= 1
            elif width > 1:
                width -= 1
            else:
                break
        return height, width

    def _sample_block_at_random_locations(
        self,
        batch_size: int,
        block_size: tuple[int, int],
    ) -> Tensor:
        return torch.stack(
            [self._sample_single_block(block_size) for _ in range(batch_size)]
        )

    def _sample_single_block(self, block_size: tuple[int, int]) -> Tensor:
        height, width = block_size
        max_top = self.config.grid_size - height
        max_left = self.config.grid_size - width
        top = int(torch.randint(max_top + 1, size=()).item())
        left = int(torch.randint(max_left + 1, size=()).item())

        rows = torch.arange(top, top + height)
        columns = torch.arange(left, left + width)
        return (rows[:, None] * self.config.grid_size + columns).reshape(-1)

    @staticmethod
    def _validate_batch_size(batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")


class BlockMaskCollator:
    """Collator compatible with a PyTorch DataLoader."""

    def __init__(self, config: BlockMaskConfig) -> None:
        self.config = config
        self.sampler = BlockMaskSampler(config)
        self.iteration = 0

    def __call__(
        self,
        images: list[Tensor],
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        if not images:
            raise ValueError("Cannot collate an empty image batch")

        image_batch = default_collate(images)
        masks = self.sampler.sample_masks(image_batch.size(0))
        self.step()
        return image_batch, masks.context_masks, masks.target_masks

    def step(self) -> None:
        """Hook for mask schedules across epochs/iterations."""
        self.iteration += 1


def build_mask_collator(config: dict[str, Any] | BlockMaskConfig) -> BlockMaskCollator:
    if isinstance(config, dict):
        config = BlockMaskConfig(**config)
    return BlockMaskCollator(config)
