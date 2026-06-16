from __future__ import annotations

import torch

from src.models.ijepa import build_ijepa
from src.training.losses import build_loss, normalize_targets
from src.training.train import OptimizerFactory, TrainingConfig


def _tiny_model_config() -> dict:
    return {
        "encoder": {
            "image_size": 32,
            "patch_size": 16,
            "embed_dim": 32,
            "depth": 1,
            "num_heads": 4,
        },
        "predictor": {
            "embed_dim": 32,
            "predictor_embed_dim": 64,
            "depth": 1,
            "num_heads": 4,
        },
    }


def test_ijepa_is_torch_module():
    model = build_ijepa(_tiny_model_config())
    assert hasattr(model, "to")
    assert hasattr(model, "state_dict")


def test_ijepa_forward_shapes_single_context_mask():
    torch.manual_seed(0)
    model = build_ijepa(_tiny_model_config())
    images = torch.randn(2, 3, 32, 32)
    context_masks = [torch.tensor([[0, 1], [1, 2]])]
    target_masks = [torch.tensor([[2, 3], [0, 3]])]

    predictions, targets = model(images, context_masks, target_masks)

    assert predictions.shape == (2, 2, 32)
    assert targets.shape == predictions.shape
    assert torch.isfinite(predictions).all()
    assert torch.isfinite(targets).all()


def test_ijepa_forward_shapes_multiple_context_masks():
    torch.manual_seed(1)
    model = build_ijepa(_tiny_model_config())
    images = torch.randn(2, 3, 32, 32)
    context_masks = [
        torch.tensor([[0, 1], [1, 2]]),
        torch.tensor([[2, 3], [0, 3]]),
    ]
    target_masks = [torch.tensor([[2, 3], [0, 3]])]

    predictions, targets = model(images, context_masks, target_masks)

    assert predictions.shape == (4, 2, 32)
    assert targets.shape == predictions.shape


def test_loss_computes_smooth_l1():
    loss_fn = build_loss({"name": "smooth_l1", "normalize_targets": True})
    predictions = torch.zeros(2, 3, 4)
    targets = torch.randn(2, 3, 4)
    loss = loss_fn(predictions, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_normalize_targets_normalizes_last_dimension():
    targets = torch.randn(2, 3, 8)
    normalized = normalize_targets(targets)
    assert torch.allclose(normalized.mean(dim=-1), torch.zeros(2, 3), atol=1e-5)
    assert torch.allclose(
        normalized.var(dim=-1, unbiased=False),
        torch.ones(2, 3),
        atol=1e-5,
    )


def test_optimizer_factory_schedules_per_step():
    model = build_ijepa(_tiny_model_config())
    config = TrainingConfig(
        epochs=2,
        warmup_epochs=1,
        lr=1.0,
        start_lr=0.1,
        final_lr=0.01,
        weight_decay=0.1,
        final_weight_decay=0.2,
    )
    optimizer, lr_scheduler, wd_scheduler = OptimizerFactory(config).build(
        model,
        steps_per_epoch=2,
    )

    assert optimizer.param_groups[0]["lr"] == 0.1
    optimizer.step()
    lr_scheduler.step()
    wd_scheduler.step()
    assert optimizer.param_groups[0]["lr"] > 0.1
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
