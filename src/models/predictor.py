from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PredictorConfig:
    """Configuration for the latent-space predictor."""

    image_size: int = 224
    patch_size: int = 16

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

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
        dim = config.predictor_embed_dim
        hidden_dim = int(dim * config.mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.norm1(tokens)
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        x = tokens + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class IJEPAPredictor(nn.Module):
    """Predict target patch representations from context representations."""

    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.context_proj = nn.Linear(config.embed_dim, config.predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.predictor_embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches, config.predictor_embed_dim)
        )
        self.blocks = nn.ModuleList(
            [PredictorBlock(config) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(config.predictor_embed_dim)
        self.output_proj = nn.Linear(config.predictor_embed_dim, config.embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def gather_pos_embed(self, masks: list[Tensor]) -> Tensor:
        if not masks:
            raise ValueError("masks cannot be empty.")

        pos = []
        for mask in masks:
            if mask.ndim != 2:
                raise ValueError(f"mask must be [B, N], got {tuple(mask.shape)}")

            mask = mask.to(device=self.pos_embed.device, dtype=torch.long)
            if mask.numel() > 0:
                min_index = int(mask.min().item())
                max_index = int(mask.max().item())
                if min_index < 0 or max_index >= self.config.num_patches:
                    raise ValueError(
                        "mask indices must be in "
                        f"[0, {self.config.num_patches - 1}], got "
                        f"[{min_index}, {max_index}]"
                    )

            expanded = mask.unsqueeze(-1).expand(
                -1,
                -1,
                self.config.predictor_embed_dim,
            )
            pos_embed = self.pos_embed.expand(mask.size(0), -1, -1)
            pos.append(pos_embed.gather(1, expanded))
        return torch.cat(pos, dim=1)

    def gather_context_pos_embed(self, masks: list[Tensor], batch_size: int) -> Tensor:
        base_batch_size = masks[0].size(0)
        if batch_size == base_batch_size:
            return self.gather_pos_embed(masks)
        if batch_size == base_batch_size * len(masks):
            return torch.cat([self.gather_pos_embed([mask]) for mask in masks], dim=0)
        raise ValueError(
            "context token batch size must match the image batch or "
            f"image batch * num context masks, got {batch_size}"
        )

    def gather_target_pos_embed(self, masks: list[Tensor], batch_size: int) -> Tensor:
        target_pos = self.gather_pos_embed(masks)
        if batch_size == target_pos.size(0):
            return target_pos
        if batch_size % target_pos.size(0) != 0:
            raise ValueError(
                "target position batch size is incompatible with context tokens: "
                f"{target_pos.size(0)} vs {batch_size}"
            )
        return target_pos.repeat(batch_size // target_pos.size(0), 1, 1)

    def add_mask_tokens(
        self,
        context_tokens: Tensor,
        target_masks: list[Tensor],
    ) -> Tensor:
        batch_size = context_tokens.size(0)
        target_pos = self.gather_target_pos_embed(target_masks, batch_size)
        mask_tokens = self.mask_token.expand(batch_size, target_pos.size(1), -1)
        return mask_tokens + target_pos

    def forward(
        self,
        context_tokens: Tensor,
        context_masks: list[Tensor],
        target_masks: list[Tensor],
    ) -> Tensor:
        context_tokens = self.context_proj(context_tokens)

        context_pos = self.gather_context_pos_embed(
            context_masks,
            context_tokens.size(0),
        )
        context_tokens = context_tokens + context_pos

        target_tokens = self.add_mask_tokens(context_tokens, target_masks)

        tokens = torch.cat([context_tokens, target_tokens], dim=1)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        target_tokens = tokens[:, context_tokens.size(1) :, :]
        predictions = self.output_proj(target_tokens)
        return predictions


def build_predictor(config: dict[str, Any] | PredictorConfig) -> IJEPAPredictor:
    """Factory used by the training loop."""
    if isinstance(config, dict):
        config = PredictorConfig(**config)
    return IJEPAPredictor(config)
