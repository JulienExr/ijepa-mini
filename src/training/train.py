from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor, nn
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader
else:
    Tensor = Any
    DataLoader = Any
    Optimizer = Any

    class _Module:
        pass

    class _NN:
        Module = _Module

    nn = _NN()

from src.data.dataset import build_dataloader
from src.masks.block_mask import build_mask_collator
from src.models.ijepa import IJEPA, build_ijepa
from src.training.losses import IJEPALoss, build_loss


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization parameters for I-JEPA pretraining."""

    epochs: int = 100
    warmup_epochs: int = 10
    lr: float = 0.000625
    start_lr: float = 0.0002
    final_lr: float = 0.000001
    weight_decay: float = 0.04
    final_weight_decay: float = 0.4
    use_bfloat16: bool = False


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint paths and cadence."""

    folder: str = "outputs"
    write_tag: str = "ijepa-mini"
    save_every_epochs: int = 1
    load_checkpoint: bool = False
    read_checkpoint: str | None = None

    @property
    def output_dir(self) -> Path:
        return Path(self.folder)


class OptimizerFactory:
    """Build optimizer and schedulers for the I-JEPA modules."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def build(self, model: IJEPA) -> tuple[Optimizer, Any, Any]:
        raise NotImplementedError("Optimizer and schedulers are not implemented.")


class CheckpointManager:
    """Save and restore training state."""

    def __init__(self, config: CheckpointConfig) -> None:
        self.config = config

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError("Checkpoint saving is not implemented.")

    def load(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("Checkpoint loading is not implemented.")


class IJEPATrainer:
    """Own the full pretraining lifecycle."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model: IJEPA | None = None
        self.loss_fn: IJEPALoss | None = None
        self.optimizer: Optimizer | None = None
        self.train_loader: DataLoader | None = None
        self.checkpoints: CheckpointManager | None = None

    def setup(self) -> None:
        raise NotImplementedError("Training setup is not implemented.")

    def build_model(self) -> IJEPA:
        return build_ijepa(self.config.get("model", {}))

    def build_loss(self) -> IJEPALoss:
        return build_loss(self.config.get("loss", {}))

    def build_dataloader(self) -> DataLoader:
        mask_collator = build_mask_collator(self.config.get("mask", {}))
        return build_dataloader(
            self.config.get("data", {}),
            collate_fn=mask_collator,
        )

    def train(self) -> None:
        raise NotImplementedError("Training loop is not implemented.")

    def train_epoch(self, epoch: int) -> dict[str, float]:
        raise NotImplementedError("Training epoch is not implemented.")

    def train_step(
        self,
        images: Tensor,
        context_masks: list[Tensor],
        target_masks: list[Tensor],
    ) -> dict[str, float]:
        raise NotImplementedError("Training step is not implemented.")

    def validate_batch(
        self,
        images: Tensor,
        context_masks: list[Tensor],
        target_masks: list[Tensor],
    ) -> None:
        raise NotImplementedError("Batch validation is not implemented.")


def main(config: dict[str, Any]) -> None:
    """Run I-JEPA pretraining.

    The root ``main.py`` loads the YAML config and dispatches here for the
    training loop, matching the organization of the official I-JEPA project.
    """
    trainer = IJEPATrainer(config)
    trainer.setup()
    trainer.train()
