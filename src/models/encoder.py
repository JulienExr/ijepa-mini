from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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


def sincos_2d_pos_embed(embed_dim: int, grid_size: int) -> Tensor:
    """Return a (1, grid_size**2, embed_dim) 2D sin-cos position embedding."""
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos"

    coords = torch.arange(grid_size, dtype=torch.float32)
    grid_row, grid_col = torch.meshgrid(coords, coords, indexing="ij")
    grid_row = grid_row.reshape(-1)
    grid_col = grid_col.reshape(-1)

    half_dim = embed_dim // 2

    def sincos_1d(positions: Tensor) -> Tensor:
        quarter = half_dim // 2
        freq_idx = torch.arange(quarter, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (freq_idx / quarter))
        angles = positions[:, None] * inv_freq[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    row_emb = sincos_1d(grid_row)
    col_emb = sincos_1d(grid_col)
    pos = torch.cat([row_emb, col_emb], dim=1)
    return pos.unsqueeze(0)


class PatchEmbedding(nn.Module):
    """Convert an image batch into a sequence of patch embeddings."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        assert config.image_size % config.patch_size == 0
        self.grid_size = config.image_size // config.patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, images: Tensor) -> Tensor:
        x = self.proj(images)
        return x.flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_drop = nn.Dropout(config.dropout)
        self.proj_drop = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj_drop(self.proj(out))


class MLP(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        hidden = int(config.embed_dim * config.mlp_ratio)
        self.fc1 = nn.Linear(config.embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class TransformerEncoderBlock(nn.Module):
    """Single ViT encoder block used by the I-JEPA encoder."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.attn(self.norm1(tokens))
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class VisionTransformerEncoder(nn.Module):
    """Context or target encoder for I-JEPA."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)
        self.register_buffer(
            "pos_embed",
            sincos_2d_pos_embed(config.embed_dim, self.patch_embed.grid_size),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(config) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode_patches(self, images: Tensor) -> Tensor:
        return self.patch_embed(images) + self.pos_embed

    def apply_context_masks(self, tokens: Tensor, masks: list[Tensor]) -> Tensor:
        if len(masks) == 1:
            mask = masks[0]
            idx = mask.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
            return tokens.gather(dim=1, index=idx)

        outs: list[Tensor] = []
        for mask in masks:
            idx = mask.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
            outs.append(tokens.gather(dim=1, index=idx))
        return torch.cat(outs, dim=0)

    def forward(self, images: Tensor, masks: list[Tensor] | None = None) -> Tensor:
        tokens = self.encode_patches(images)
        if masks is not None:
            tokens = self.apply_context_masks(tokens, masks)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


def build_encoder(config: dict[str, Any] | EncoderConfig) -> VisionTransformerEncoder:
    """Factory used by training and evaluation code."""
    if isinstance(config, dict):
        config = EncoderConfig(**config)
    return VisionTransformerEncoder(config)
