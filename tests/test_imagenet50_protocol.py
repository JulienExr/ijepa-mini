from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch import nn

from scripts.compare_imagenet50_linear_probe import (
    build_jepa_feature_model,
    build_torchvision_feature_model,
    linear_probe,
)
from scripts.prepare_imagenet50_subset import ImageNet50Config, create_subset
from src.models.encoder import build_encoder


def _fake_imagenet_stream(num_classes: int, images_per_class: int) -> list[dict]:
    samples = []
    for rank in range(images_per_class):
        for label in range(num_classes):
            color = (label * 40 % 255, rank * 20 % 255, 128)
            image = Image.new("RGB", (24, 24), color)
            samples.append({"label": label, "image": image})
    return samples


def test_prepare_imagenet50_subset_writes_exact_stratified_split(
    tmp_path: Path,
) -> None:
    config = ImageNet50Config(
        output_dir=tmp_path / "imagenet50-smoke",
        num_classes=3,
        images_per_class=10,
        train_per_class=8,
        seed=123,
    )

    create_subset(config, dataset=_fake_imagenet_stream(3, 10))

    manifest_path = config.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["num_classes"] == 3
    assert manifest["images_per_class"] == 10
    assert manifest["train_per_class"] == 8
    assert manifest["val_per_class"] == 2
    assert manifest["counts"] == {"train": 24, "val": 6}

    for class_info in manifest["classes"]:
        assert class_info["total"] == 10
        assert class_info["train"] == 8
        assert class_info["val"] == 2
        class_name = class_info["name"]
        assert len(list((config.output_dir / "train" / class_name).glob("*.jpg"))) == 8
        assert len(list((config.output_dir / "val" / class_name).glob("*.jpg"))) == 2


def test_prepare_imagenet50_subset_rejects_incomplete_classes(tmp_path: Path) -> None:
    config = ImageNet50Config(
        output_dir=tmp_path / "imagenet50-smoke",
        num_classes=3,
        images_per_class=10,
        train_per_class=8,
    )

    try:
        create_subset(config, dataset=_fake_imagenet_stream(3, 9))
    except ValueError as exc:
        assert "Could not collect exactly 10 images per class" in str(exc)
    else:
        raise AssertionError("Expected incomplete class counts to fail")


def test_jepa_feature_model_loads_context_encoder_checkpoint(tmp_path: Path) -> None:
    model_config = {
        "model": {
            "encoder": {
                "image_size": 32,
                "patch_size": 16,
                "in_channels": 3,
                "embed_dim": 32,
                "depth": 1,
                "num_heads": 4,
                "mlp_ratio": 4.0,
                "dropout": 0.0,
                "use_cls_token": False,
            }
        }
    }
    config_path = tmp_path / "small_jepa.yaml"
    config_path.write_text(yaml.safe_dump(model_config), encoding="utf-8")

    encoder = build_encoder(model_config["model"]["encoder"])
    checkpoint_path = tmp_path / "small_jepa.pt"
    torch.save(
        {
            "epoch": 2,
            "model_state": {
                f"context_encoder.{key}": value
                for key, value in encoder.state_dict().items()
            },
        },
        checkpoint_path,
    )

    feature_model = build_jepa_feature_model(config_path, checkpoint_path)
    features = feature_model(torch.randn(2, 3, 32, 32))
    assert features.shape == (2, 32)


def test_torchvision_feature_model_builder_accepts_random_weights(monkeypatch) -> None:
    class SmallVit(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = nn.Linear(3, 3)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return images.mean(dim=(2, 3))

    captured = {}

    def fake_vit_b_16(weights=None):
        captured["weights"] = weights
        return SmallVit()

    monkeypatch.setattr(
        "scripts.compare_imagenet50_linear_probe.models.vit_b_16",
        fake_vit_b_16,
    )

    model = build_torchvision_feature_model("none")
    assert captured["weights"] is None
    assert isinstance(model.heads, nn.Identity)
    assert model(torch.randn(2, 3, 8, 8)).shape == (2, 3)


def test_linear_probe_runs_on_cached_feature_tensors() -> None:
    train_x = torch.tensor(
        [
            [2.0, 0.0],
            [1.5, 0.2],
            [0.0, 2.0],
            [0.2, 1.5],
        ]
    )
    train_y = torch.tensor([0, 0, 1, 1])
    val_x = torch.tensor([[1.8, 0.1], [0.1, 1.8]])
    val_y = torch.tensor([0, 1])

    train_acc, val_acc = linear_probe(
        train_x,
        train_y,
        val_x,
        val_y,
        num_classes=2,
        epochs=5,
        lr=0.2,
        batch_size=2,
        device="cpu",
        seed=0,
    )

    assert 0.0 <= train_acc <= 1.0
    assert 0.0 <= val_acc <= 1.0
