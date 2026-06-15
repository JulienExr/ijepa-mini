"""Device-parametrized smoke tests (cpu + mps when available)."""

from __future__ import annotations

import torch

from src.models.encoder import EncoderConfig, VisionTransformerEncoder
from src.training.ema import (
    EMAConfig,
    EMAScheduler,
    copy_model_for_ema,
    update_ema,
)


def test_encoder_forward_on_device(device):
    torch.manual_seed(0)
    enc = VisionTransformerEncoder(EncoderConfig()).to(device)
    enc.eval()
    x = torch.randn(2, 3, 224, 224, device=device)
    out = enc(x)
    assert out.shape == (2, 196, 192)
    assert out.device.type == device.type
    assert torch.isfinite(out).all()


def test_encoder_masked_forward_on_device(device):
    torch.manual_seed(1)
    enc = VisionTransformerEncoder(EncoderConfig()).to(device)
    enc.eval()
    x = torch.randn(2, 3, 224, 224, device=device)
    idx = torch.stack(
        [torch.randperm(196)[:30] for _ in range(2)]
    ).to(device)
    out = enc(x, masks=[idx])
    assert out.shape == (2, 30, 192)
    assert out.device.type == device.type
    assert torch.isfinite(out).all()


def test_pos_embed_moves_with_to_device(device):
    enc = VisionTransformerEncoder(EncoderConfig()).to(device)
    assert enc.pos_embed.device.type == device.type


def test_update_ema_on_device(device):
    torch.manual_seed(0)
    src = VisionTransformerEncoder(EncoderConfig()).to(device)
    tgt = copy_model_for_ema(src)
    with torch.no_grad():
        for p in src.parameters():
            p.add_(0.01)
    before = next(tgt.parameters()).detach().clone()
    src_first = next(src.parameters()).detach().clone()
    update_ema(src, tgt, momentum=0.9)
    after = next(tgt.parameters()).detach().clone()
    expected = before * 0.9 + src_first * 0.1
    assert after.device.type == device.type
    assert torch.allclose(after, expected, atol=1e-5)


def test_ema_scheduler_is_device_agnostic(device):
    sched = EMAScheduler(EMAConfig(start=0.5, end=1.0, total_steps=4))
    ms = [next(sched) for _ in range(5)]
    assert ms[0] == 0.5
    assert ms[-1] == 1.0
