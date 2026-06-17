from __future__ import annotations

import argparse
import json
import random
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

DEFAULT_HF_DATASET = "ILSVRC/imagenet-1k"
DEFAULT_HF_SPLIT = "train"
DEFAULT_OUTPUT_DIR = Path("data/imagenet50-200")


@dataclass(frozen=True)
class ImageNet50Config:
    hf_dataset: str = DEFAULT_HF_DATASET
    hf_split: str = DEFAULT_HF_SPLIT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    num_classes: int = 50
    images_per_class: int = 200
    train_per_class: int = 160
    classes: tuple[int, ...] | None = None
    seed: int = 0
    image_quality: int = 95
    overwrite: bool = False

    @property
    def val_per_class(self) -> int:
        return self.images_per_class - self.train_per_class


@dataclass(frozen=True)
class CollectedSample:
    label: int
    rank: int
    image: Image.Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic ImageNet-50 subset from Hugging Face streaming "
            "with a local stratified train/val split."
        )
    )
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--hf-split", default=DEFAULT_HF_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-classes", type=int, default=50)
    parser.add_argument("--images-per-class", type=int, default=200)
    parser.add_argument("--train-per-class", type=int, default=160)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Numeric ImageNet labels to keep. Defaults to labels 0..num_classes-1.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-quality", type=int, default=95)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before writing the subset.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ImageNet50Config:
    classes = None
    if args.classes is not None:
        classes = tuple(parse_label(value) for value in args.classes)

    config = ImageNet50Config(
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        output_dir=args.output_dir.expanduser().resolve(),
        num_classes=args.num_classes,
        images_per_class=args.images_per_class,
        train_per_class=args.train_per_class,
        classes=classes,
        seed=args.seed,
        image_quality=args.image_quality,
        overwrite=args.overwrite,
    )
    validate_config(config)
    return config


def validate_config(config: ImageNet50Config) -> None:
    if config.num_classes <= 0:
        raise ValueError("--num-classes must be positive")
    if config.images_per_class <= 0:
        raise ValueError("--images-per-class must be positive")
    if not 0 < config.train_per_class < config.images_per_class:
        raise ValueError("--train-per-class must be between 1 and images_per_class - 1")
    if not 1 <= config.image_quality <= 100:
        raise ValueError("--image-quality must be between 1 and 100")
    if config.classes is not None:
        if len(config.classes) != config.num_classes:
            raise ValueError(
                f"--classes must contain exactly {config.num_classes} labels, got "
                f"{len(config.classes)}"
            )
        if len(set(config.classes)) != len(config.classes):
            raise ValueError("--classes must not contain duplicates")


def parse_label(value: str) -> int:
    try:
        label = int(value)
    except ValueError as exc:
        raise ValueError(f"ImageNet labels must be numeric, got {value!r}") from exc
    if label < 0:
        raise ValueError(f"ImageNet labels must be non-negative, got {label}")
    return label


def load_streaming_dataset(config: ImageNet50Config) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required. Install project dependencies with "
            "`uv sync`."
        ) from exc

    try:
        return load_dataset(
            config.hf_dataset,
            split=config.hf_split,
            streaming=True,
            token=True,
        )
    except Exception as exc:  # pragma: no cover - depends on remote auth state.
        raise RuntimeError(hf_auth_message(config)) from exc


def hf_auth_message(config: ImageNet50Config) -> str:
    return (
        f"Could not open Hugging Face dataset {config.hf_dataset!r}. "
        "Make sure you have accepted the gated ImageNet terms and are logged in "
        "locally with `huggingface-cli login`."
    )


def selected_labels(config: ImageNet50Config) -> tuple[int, ...]:
    if config.classes is not None:
        return config.classes
    return tuple(range(config.num_classes))


def create_subset(
    config: ImageNet50Config, dataset: Iterable[Any] | None = None
) -> None:
    validate_config(config)
    if config.overwrite and config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {config.output_dir}. "
            "Pass --overwrite to replace it."
        )

    dataset = dataset if dataset is not None else load_streaming_dataset(config)
    label_names = resolve_label_names(dataset, selected_labels(config))
    samples = collect_samples(config, dataset)
    write_split(config, samples, label_names)


def resolve_label_names(dataset: Any, labels: tuple[int, ...]) -> dict[int, str]:
    names: dict[int, str] = {}
    features = getattr(dataset, "features", None)
    label_feature = features.get("label") if isinstance(features, dict) else None
    for label in labels:
        if label_feature is not None and hasattr(label_feature, "int2str"):
            raw_name = label_feature.int2str(label)
        else:
            raw_name = f"class_{label:03d}"
        names[label] = sanitize_folder_name(raw_name, fallback=f"class_{label:03d}")
    return names


def sanitize_folder_name(name: str, fallback: str = "class") -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    safe = safe.strip("._")
    return safe or fallback


def collect_samples(
    config: ImageNet50Config,
    dataset: Iterable[Any],
) -> dict[int, list[CollectedSample]]:
    labels = selected_labels(config)
    wanted = set(labels)
    samples: dict[int, list[CollectedSample]] = {label: [] for label in labels}
    progress = tqdm(desc="streaming ImageNet", unit="sample")

    try:
        for sample in dataset:
            progress.update(1)
            label = int(sample["label"])
            if label not in wanted:
                continue
            class_samples = samples[label]
            if len(class_samples) >= config.images_per_class:
                if all(
                    len(items) >= config.images_per_class for items in samples.values()
                ):
                    break
                continue

            image = sample["image"].convert("RGB")
            class_samples.append(
                CollectedSample(
                    label=label,
                    rank=len(class_samples),
                    image=image.copy(),
                )
            )

            if all(len(items) >= config.images_per_class for items in samples.values()):
                break
    except Exception as exc:  # pragma: no cover - depends on remote streaming state.
        raise RuntimeError(hf_auth_message(config)) from exc
    finally:
        progress.close()

    missing = {
        label: len(items)
        for label, items in samples.items()
        if len(items) != config.images_per_class
    }
    if missing:
        raise ValueError(
            f"Could not collect exactly {config.images_per_class} images per class. "
            f"Counts: {missing}"
        )
    return samples


def write_split(
    config: ImageNet50Config,
    samples: dict[int, list[CollectedSample]],
    label_names: dict[int, str],
) -> None:
    rng = random.Random(config.seed)
    manifest_classes: list[dict[str, Any]] = []
    split_counts = {"train": 0, "val": 0}

    for label in selected_labels(config):
        class_samples = list(samples[label])
        rng.shuffle(class_samples)
        train_samples = class_samples[: config.train_per_class]
        val_samples = class_samples[config.train_per_class :]
        folder_name = label_names[label]

        for split, split_samples in (
            ("train", train_samples),
            ("val", val_samples),
        ):
            split_counts[split] += len(split_samples)
            class_dir = config.output_dir / split / folder_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for split_rank, sample in enumerate(split_samples):
                filename = (
                    f"label{label:04d}_source{sample.rank:04d}_{split_rank:04d}.jpg"
                )
                sample.image.save(class_dir / filename, quality=config.image_quality)

        manifest_classes.append(
            {
                "label": label,
                "name": folder_name,
                "total": len(class_samples),
                "train": len(train_samples),
                "val": len(val_samples),
            }
        )

    manifest = {
        "source": config.hf_dataset,
        "split": config.hf_split,
        "seed": config.seed,
        "num_classes": config.num_classes,
        "images_per_class": config.images_per_class,
        "train_per_class": config.train_per_class,
        "val_per_class": config.val_per_class,
        "counts": split_counts,
        "classes": manifest_classes,
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "classes": list(config.classes) if config.classes is not None else None,
        },
    }
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    create_subset(build_config(parse_args()))


if __name__ == "__main__":
    main()
