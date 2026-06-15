from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import transforms

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
        if not self.root.exists():
            raise FileNotFoundError(f"Image directory not found: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Image root is not a directory: {self.root}")

        self.image_paths = tuple(
            path
            for path in sorted(self.root.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise ValueError(f"No supported images found in {self.root}")

        self.transform = transform or transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tensor:
        with Image.open(self.image_paths[index]) as image:
            image = image.convert("RGB")
            return self.transform(image)


class IJEPABatchCollator:
    """Build image batches and attach I-JEPA context/target masks."""

    def __init__(
        self,
        mask_collator: Callable[[list[Tensor]], Any] | None = None,
    ) -> None:
        self.mask_collator = mask_collator

    def __call__(self, samples: list[Tensor]) -> Any:
        if self.mask_collator is not None:
            return self.mask_collator(samples)
        return default_collate(samples)


def build_transforms(config: dict[str, Any] | DatasetConfig) -> Callable[[Any], Tensor]:
    """Create image transforms for pretraining."""
    if isinstance(config, dict):
        config = DatasetConfig(**config)
    if config.image_size <= 0:
        raise ValueError("image_size must be positive")

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config.image_size,
                scale=(0.3, 1.0),
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_dataset(config: dict[str, Any] | DatasetConfig) -> ImageFolderDataset:
    if isinstance(config, dict):
        config = DatasetConfig(**config)
    return ImageFolderDataset(
        root=config.image_root,
        transform=build_transforms(config),
    )


def build_dataloader(
    config: dict[str, Any] | DatasetConfig,
    collate_fn: Callable[[list[Tensor]], Any] | None = None,
) -> DataLoader[Any]:
    """Create the pretraining dataloader."""
    if isinstance(config, dict):
        config = DatasetConfig(**config)
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    dataset = build_dataset(config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        collate_fn=IJEPABatchCollator(collate_fn),
    )
