from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset
else:
    Tensor = Any
    DataLoader = Any

    class Dataset:
        def __class_getitem__(cls, _item: Any) -> type["Dataset"]:
            return cls


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for pretraining image data."""

    root_path: str = "data"
    image_folder: str = "train"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True

    @property
    def image_root(self) -> Path:
        return Path(self.root_path) / self.image_folder


class ImageFolderDataset(Dataset[Tensor]):
    """Image dataset used for self-supervised I-JEPA pretraining."""

    def __init__(
        self,
        root: str | Path,
        transform: Callable[[Any], Tensor] | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        raise NotImplementedError("Dataset length is not implemented.")

    def __getitem__(self, index: int) -> Tensor:
        raise NotImplementedError("Image loading is not implemented.")


class IJEPABatchCollator:
    """Build image batches and attach I-JEPA context/target masks."""

    def __init__(self, mask_collator: Callable[[list[Tensor]], Any] | None = None) -> None:
        self.mask_collator = mask_collator

    def __call__(self, samples: list[Tensor]) -> Any:
        raise NotImplementedError("Batch collation is not implemented.")


def build_transforms(config: dict[str, Any] | DatasetConfig) -> Callable[[Any], Tensor]:
    """Create image transforms for pretraining."""
    raise NotImplementedError("Image transforms are not implemented.")


def build_dataset(config: dict[str, Any] | DatasetConfig) -> ImageFolderDataset:
    if isinstance(config, dict):
        config = DatasetConfig(**config)
    return ImageFolderDataset(root=config.image_root, transform=build_transforms(config))


def build_dataloader(
    config: dict[str, Any] | DatasetConfig,
    collate_fn: Callable[[list[Tensor]], Any] | None = None,
) -> DataLoader[Tensor]:
    """Create the pretraining dataloader."""
    raise NotImplementedError("Dataloader construction is not implemented.")
