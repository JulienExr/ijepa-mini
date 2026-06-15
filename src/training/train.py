from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import build_dataloader
from src.masks.block_mask import build_mask_collator
from src.models.ijepa import IJEPA, build_ijepa
from src.training.ema import EMAConfig, EMAScheduler, update_ema
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
    use_amp: bool = True
    amp_dtype: str = "float16"


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
        params = list(model.context_encoder.parameters())
        params += list(model.predictor.parameters())
        t_max = max(1, self.config.epochs - self.config.warmup_epochs)

        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=self.config.final_lr,
        )
        wd_scheduler = None

        return optimizer, lr_scheduler, wd_scheduler


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
        lr_scheduler: Any | None = None,
        wd_scheduler: Any | None = None,
        ema_scheduler: Any | None = None,
        scaler: Any | None = None,
    ) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = (
            self.config.output_dir / f"{self.config.write_tag}_epoch_{epoch:04d}.pt"
        )
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "extra_state": extra_state or {},
        }

        if lr_scheduler is not None:
            checkpoint["lr_scheduler_state"] = lr_scheduler.state_dict()
        if wd_scheduler is not None:
            checkpoint["wd_scheduler_state"] = wd_scheduler.state_dict()
        if ema_scheduler is not None:
            checkpoint["ema_scheduler_state"] = ema_scheduler.state_dict()
        if scaler is not None:
            checkpoint["scaler_state"] = scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        latest_path = self.config.output_dir / f"{self.config.write_tag}_latest.pt"
        torch.save(checkpoint, latest_path)

    def load(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        lr_scheduler: Any | None = None,
        wd_scheduler: Any | None = None,
        ema_scheduler: Any | None = None,
        scaler: Any | None = None,
    ) -> dict[str, Any]:
        if self.config.read_checkpoint is None:
            path = self.config.output_dir / f"{self.config.write_tag}_latest.pt"
        else:
            path = Path(self.config.read_checkpoint)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])

        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])

        if lr_scheduler is not None and "lr_scheduler_state" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])

        if wd_scheduler is not None and "wd_scheduler_state" in checkpoint:
            wd_scheduler.load_state_dict(checkpoint["wd_scheduler_state"])

        if ema_scheduler is not None and "ema_scheduler_state" in checkpoint:
            ema_scheduler.load_state_dict(checkpoint["ema_scheduler_state"])

        if scaler is not None and "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])

        return {
            "epoch": checkpoint.get("epoch", 0),
            "extra_state": checkpoint.get("extra_state", {}),
            "path": str(path),
        }


class IJEPATrainer:
    """Own the full pretraining lifecycle."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model: IJEPA | None = None
        self.loss_fn: IJEPALoss | None = None
        self.optimizer: Optimizer | None = None
        self.train_loader: DataLoader | None = None
        self.checkpoints: CheckpointManager | None = None
        self.lr_scheduler: Any | None = None
        self.wd_scheduler: Any | None = None
        self.ema_scheduler: EMAScheduler | None = None
        self.scaler: Any | None = None
        self.use_amp = False
        self.amp_dtype = torch.float16
        self.device: str = "cpu"
        self.start_epoch = 0

    def setup(self) -> None:
        self.device = self.config["runtime"]["device"]

        self.model = self.build_model().to(self.device)
        self.loss_fn = self.build_loss().to(self.device)
        self.train_loader = self.build_dataloader()

        optimization_config = self.config.get("optimization", {})
        training_config = self._build_training_config(optimization_config)
        self.optimizer, self.lr_scheduler, self.wd_scheduler = OptimizerFactory(
            training_config
        ).build(self.model)
        self.ema_scheduler = self._build_ema_scheduler(
            optimization_config,
            training_config,
        )
        self.use_amp = self._should_use_amp(training_config)
        self.amp_dtype = self._get_amp_dtype(training_config)
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and self.amp_dtype is torch.float16,
        )

        checkpoint_config = CheckpointConfig(
            **self.config.get("logging", {}),
            load_checkpoint=self.config.get("meta", {}).get("load_checkpoint", False),
            read_checkpoint=self.config.get("meta", {}).get("read_checkpoint"),
        )
        self.checkpoints = CheckpointManager(checkpoint_config)

        if checkpoint_config.load_checkpoint:
            state = self.checkpoints.load(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                wd_scheduler=self.wd_scheduler,
                ema_scheduler=self.ema_scheduler,
                scaler=self.scaler,
            )
            self.start_epoch = int(state["epoch"]) + 1

    def _build_training_config(self, config: dict[str, Any]) -> TrainingConfig:
        valid_fields = {field.name for field in fields(TrainingConfig)}
        kwargs = {key: value for key, value in config.items() if key in valid_fields}
        return TrainingConfig(**kwargs)

    def _should_use_amp(self, config: TrainingConfig) -> bool:
        return config.use_amp and self.device.startswith("cuda")

    def _get_amp_dtype(self, config: TrainingConfig) -> torch.dtype:
        if config.amp_dtype != "float16":
            raise ValueError("Only float16 AMP is currently supported.")
        return torch.float16

    def _build_ema_scheduler(
        self,
        optimization_config: dict[str, Any],
        training_config: TrainingConfig,
    ) -> EMAScheduler:
        ema = optimization_config.get("ema", (0.996, 1.0))
        if len(ema) != 2:
            raise ValueError("optimization.ema must contain [start, end].")

        try:
            steps_per_epoch = len(self.train_loader)
        except (NotImplementedError, TypeError):
            steps_per_epoch = 1

        total_steps = max(1, training_config.epochs * steps_per_epoch)
        return EMAScheduler(
            EMAConfig(
                start=float(ema[0]),
                end=float(ema[1]),
                total_steps=total_steps,
            )
        )

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
        assert self.model is not None
        assert self.train_loader is not None

        epochs = self.config.get("optimization", {}).get("epochs", 100)

        for epoch in range(self.start_epoch, epochs):
            train_metrics = self.train_epoch(epoch)
            tqdm.write(f"Epoch {epoch}: {train_metrics}")

            should_save = (
                self.checkpoints is not None
                and (epoch + 1) % self.checkpoints.config.save_every_epochs == 0
            )
            if should_save:
                assert self.optimizer is not None
                assert self.checkpoints is not None
                self.checkpoints.save(
                    epoch,
                    self.model,
                    self.optimizer,
                    extra_state=train_metrics,
                    lr_scheduler=self.lr_scheduler,
                    wd_scheduler=self.wd_scheduler,
                    ema_scheduler=self.ema_scheduler,
                    scaler=self.scaler,
                )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        assert self.model is not None
        assert self.train_loader is not None

        self.model.train()

        total_loss = 0.0
        num_batches = 0
        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in progress:
            images, context_masks, target_masks = batch

            metrics = self.train_step(images, context_masks, target_masks)
            total_loss += metrics["loss"]
            num_batches += 1
            progress.set_postfix(loss=f"{metrics['loss']:.4f}")

        return {"loss": total_loss / num_batches if num_batches > 0 else 0.0}

    def train_step(
        self,
        images: Tensor,
        context_masks: list[Tensor],
        target_masks: list[Tensor],
    ) -> dict[str, float]:
        assert self.model is not None
        assert self.loss_fn is not None
        assert self.optimizer is not None
        assert self.scaler is not None

        images = images.to(self.device)
        context_masks = [mask.to(self.device) for mask in context_masks]
        target_masks = [mask.to(self.device) for mask in target_masks]

        self.validate_batch(images, context_masks, target_masks)

        with torch.autocast(
            device_type="cuda",
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            predictions, targets = self.model(images, context_masks, target_masks)
            loss = self.loss_fn(predictions, targets)

        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.ema_scheduler is not None:
            momentum = next(self.ema_scheduler)
            update_ema(self.model.context_encoder, self.model.target_encoder, momentum)

        return {"loss": float(loss.detach().cpu())}

    def validate_batch(
        self,
        images: Tensor,
        context_masks: list[Tensor],
        target_masks: list[Tensor],
    ) -> None:
        if images.ndim != 4:
            raise ValueError(f"images must be [B, C, H, W], got {images.shape}")

        if not context_masks:
            raise ValueError("context_masks cannot be empty")

        if not target_masks:
            raise ValueError("target_masks cannot be empty")

        batch_size = images.size(0)

        for mask in context_masks + target_masks:
            if mask.ndim != 2:
                raise ValueError(f"mask must be [B, N], got {mask.shape}")

            if mask.size(0) != batch_size:
                raise ValueError(
                    f"mask batch size {mask.size(0)} != image batch size {batch_size}"
                )


def main(config: dict[str, Any]) -> None:
    """Run I-JEPA pretraining.

    The root ``main.py`` loads the YAML config and dispatches here for the
    training loop, matching the organization of the official I-JEPA project.
    """
    trainer = IJEPATrainer(config)
    trainer.setup()
    trainer.train()
