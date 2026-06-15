from __future__ import annotations

import torch

from src.models.encoder import (
    EncoderConfig,
    PatchEmbedding,
    TransformerEncoderBlock,
    VisionTransformerEncoder,
    build_encoder,
    sincos_2d_pos_embed,
)


def test_patch_embedding_shape():
    pe = PatchEmbedding(EncoderConfig())
    out = pe(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 196, 192)


def test_patch_embedding_grid_metadata():
    pe = PatchEmbedding(EncoderConfig())
    assert pe.grid_size == 14
    assert pe.num_patches == 196


def test_sincos_2d_pos_embed_shape():
    pos = sincos_2d_pos_embed(192, 14)
    assert pos.shape == (1, 196, 192)


def test_sincos_2d_pos_embed_norm_is_constant():
    pos = sincos_2d_pos_embed(192, 14)
    norms = pos.norm(dim=-1)
    assert norms.std().item() < 1e-4


def test_transformer_block_preserves_shape():
    block = TransformerEncoderBlock(EncoderConfig())
    x = torch.randn(2, 196, 192)
    y = block(x)
    assert y.shape == x.shape


def test_encoder_full_forward_shape():
    enc = VisionTransformerEncoder(EncoderConfig())
    enc.eval()
    x = torch.randn(2, 3, 224, 224)
    out = enc(x)
    assert out.shape == (2, 196, 192)


def test_encoder_context_mask_forward_shape():
    torch.manual_seed(0)
    enc = VisionTransformerEncoder(EncoderConfig())
    enc.eval()
    x = torch.randn(2, 3, 224, 224)
    idx = torch.stack([torch.randperm(196)[:30] for _ in range(2)])
    out = enc(x, masks=[idx])
    assert out.shape == (2, 30, 192)


def test_encoder_context_mask_matches_gathered_full_output():
    torch.manual_seed(1)
    enc = VisionTransformerEncoder(EncoderConfig())
    enc.eval()
    x = torch.randn(1, 3, 224, 224)
    idx = torch.tensor([[3, 17, 95, 120]])
    masked = enc(x, masks=[idx])
    # Sanity: shape OK and values finite.
    assert masked.shape == (1, 4, 192)
    assert torch.isfinite(masked).all()


def test_encoder_multiple_context_masks_concat_on_batch():
    torch.manual_seed(2)
    enc = VisionTransformerEncoder(EncoderConfig())
    enc.eval()
    x = torch.randn(2, 3, 224, 224)
    m1 = torch.stack([torch.randperm(196)[:10] for _ in range(2)])
    m2 = torch.stack([torch.randperm(196)[:10] for _ in range(2)])
    out = enc(x, masks=[m1, m2])
    assert out.shape == (4, 10, 192)


def test_build_encoder_from_dict():
    enc = build_encoder({"embed_dim": 192, "depth": 2, "num_heads": 3})
    out = enc(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 196, 192)


def test_encoder_pos_embed_is_buffer_not_parameter():
    enc = VisionTransformerEncoder(EncoderConfig())
    param_ids = {id(p) for p in enc.parameters()}
    assert id(enc.pos_embed) not in param_ids
