from __future__ import annotations

import torch

from src.masks.block_mask import BlockMaskConfig, BlockMaskSampler


def test_no_overlap_sampler_falls_back_to_available_context_patches() -> None:
    config = BlockMaskConfig(
        image_size=64,
        patch_size=8,
        allow_overlap=False,
        num_context_masks=1,
        num_target_masks=4,
        context_scale=(0.85, 1.0),
        target_scale=(0.15, 0.2),
        min_context_patches=4,
    )
    sampler = BlockMaskSampler(config)
    sampler.max_sampling_attempts = 1
    sampler._sample_single_block = lambda block_size: torch.arange(0, 8)
    target_masks = [
        torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]),
        torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]]),
        torch.tensor([[16, 17, 18, 19, 20, 21, 22, 23]]),
        torch.tensor([[24, 25, 26, 27, 28, 29, 30, 31]]),
    ]

    context_masks = sampler._sample_context_masks(
        batch_size=1,
        target_masks=target_masks,
    )

    context = set(context_masks[0][0].tolist())
    targets = {index for mask in target_masks for index in mask[0].tolist()}
    assert len(context) >= config.min_context_patches
    assert context.isdisjoint(targets)


def test_sampled_block_size_does_not_exceed_requested_area() -> None:
    torch.manual_seed(0)
    config = BlockMaskConfig(
        image_size=64,
        patch_size=8,
        target_scale=(0.2, 0.2),
        aspect_ratio=(1.0, 1.0),
    )
    sampler = BlockMaskSampler(config)

    height, width = sampler._sample_block_size(config.target_scale, config.aspect_ratio)

    assert height * width <= 13
