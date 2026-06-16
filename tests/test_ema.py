from __future__ import annotations

import torch
import torch.nn as nn

from src.training.ema import (
    EMAConfig,
    EMAScheduler,
    copy_model_for_ema,
    update_ema,
)


def test_copy_model_for_ema_freezes_params():
    src = nn.Linear(4, 4)
    tgt = copy_model_for_ema(src)
    assert all(not p.requires_grad for p in tgt.parameters())


def test_copy_model_for_ema_copies_weights():
    src = nn.Linear(4, 4)
    tgt = copy_model_for_ema(src)
    assert torch.allclose(tgt.weight, src.weight)
    assert torch.allclose(tgt.bias, src.bias)


def test_copy_model_for_ema_is_independent_object():
    src = nn.Linear(4, 4)
    tgt = copy_model_for_ema(src)
    with torch.no_grad():
        src.weight.add_(1.0)
    assert not torch.allclose(tgt.weight, src.weight)


def test_update_ema_moves_target_toward_source():
    torch.manual_seed(0)
    src = nn.Linear(4, 4)
    tgt = copy_model_for_ema(src)
    with torch.no_grad():
        for p in src.parameters():
            p.add_(1.0)
    before_w = tgt.weight.detach().clone()
    src_w = src.weight.detach().clone()
    update_ema(src, tgt, momentum=0.9)
    expected = before_w * 0.9 + src_w * 0.1
    assert torch.allclose(tgt.weight, expected, atol=1e-6)


def test_update_ema_momentum_one_is_noop():
    torch.manual_seed(0)
    src = nn.Linear(4, 4)
    tgt = copy_model_for_ema(src)
    with torch.no_grad():
        for p in src.parameters():
            p.add_(5.0)
    before_w = tgt.weight.detach().clone()
    update_ema(src, tgt, momentum=1.0)
    assert torch.allclose(tgt.weight, before_w)


def test_update_ema_momentum_zero_copies_source():
    torch.manual_seed(0)
    src = nn.Linear(4, 4)
    tgt = copy_model_for_ema(src)
    with torch.no_grad():
        for p in src.parameters():
            p.add_(5.0)
    update_ema(src, tgt, momentum=0.0)
    assert torch.allclose(tgt.weight, src.weight)


def test_update_ema_runs_under_no_grad():
    src = nn.Linear(4, 4)
    tgt = copy_model_for_ema(src)
    update_ema(src, tgt, momentum=0.5)
    for p in tgt.parameters():
        assert p.grad is None
        assert not p.requires_grad


def test_ema_scheduler_anneals_linearly():
    sched = EMAScheduler(EMAConfig(start=0.5, end=1.0, total_steps=10))
    ms = [next(sched) for _ in range(12)]
    assert abs(ms[0] - 0.5) < 1e-6
    assert abs(ms[5] - 0.75) < 1e-6
    assert abs(ms[10] - 1.0) < 1e-6
    assert abs(ms[11] - 1.0) < 1e-6


def test_ema_scheduler_state_roundtrip():
    sched = EMAScheduler(EMAConfig(start=0.5, end=1.0, total_steps=10))
    [next(sched) for _ in range(3)]
    state = sched.state_dict()
    sched2 = EMAScheduler(EMAConfig(start=0.5, end=1.0, total_steps=10))
    sched2.load_state_dict(state)
    assert next(sched) == next(sched2)
