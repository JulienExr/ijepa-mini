


import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

DEFAULT_IMAGENET10_CLASSES = (
    "n01440764",  # tench
    "n02102040",  # English springer
    "n02979186",  # cassette player
    "n03000684",  # chain saw
    "n03028079",  # church
    "n03394916",  # French horn
    "n03417042",  # garbage truck
    "n03425413",  # gas pump
    "n03445777",  # golf ball
    "n03888257",  # parachute
)
DEFAULT_KAGGLE_IMAGENET_COMPETITION = "imagenet-object-localization-challenge"
DEFAULT_HF_IMAGENET_DATASET = "ILSVRC/imagenet-1k"
DEFAULT_HF_SPLIT = "train"


@dataclass(frozen=True)
class SubsetConfig:
    source: Path | None
    destination: Path
    classes: tuple[str, ...] | None
    num_classes: int
    max_images_per_class: int | None
    seed: int
    mode: str
    kaggle_dataset: str | None
    kaggle_competition: str | None
    kaggle_output_dir: Path
    hf_dataset: str | None
    hf_split: str
    hf_streaming: bool
    force_download: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download or reuse an ImageNet-style dataset, then export class folders."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Source ImageNet split folder, e.g. /datasets/imagenet/train.",
    )
    parser.add_argument(
        "--kaggle-dataset",
        default=None,
        help="Kaggle dataset handle, e.g. owner/dataset-name.",
    )
    parser.add_argument(
        "--kaggle-competition",
        default=None,
        help=(
            "Kaggle competition name, e.g. "
            f"{DEFAULT_KAGGLE_IMAGENET_COMPETITION!r}."
        ),
    )
    parser.add_argument(
        "--kaggle-output-dir",
        type=Path,
        default=Path("data/downloads/kaggle"),
        help="Where kagglehub should store downloaded files.",
    )
    parser.add_argument(
        "--hf-dataset",
        default=DEFAULT_HF_IMAGENET_DATASET,
        help=(
            "Hugging Face dataset id. Defaults to "
            f"{DEFAULT_HF_IMAGENET_DATASET!r}."
        ),
    )
    parser.add_argument(
        "--hf-split",
        default=DEFAULT_HF_SPLIT,
        help="Hugging Face dataset split to export.",
    )
    parser.add_argument(
        "--hf-streaming",
        action="store_true",
        help="Stream Hugging Face samples instead of downloading the split first.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ask kagglehub to redownload even if files already exist.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/imagenet/train"),
        help="Destination folder to create.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help=(
            "Class folder names or numeric HF label ids to keep. Defaults to the "
            "first --num-classes labels for Hugging Face datasets."
        ),
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=200,
        help="Number of classes to keep from --classes or from source folders.",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Optional cap per class for faster smoke runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used when sampling images or fallback classes.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy", "hardlink"),
        default="symlink",
        help="How to materialize images in the destination.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SubsetConfig:
    uses_local_or_kaggle = (
        args.source is not None
        or args.kaggle_dataset is not None
        or args.kaggle_competition is not None
    )
    sources = [
        args.source is not None,
        args.kaggle_dataset,
        args.kaggle_competition,
    ]
    if args.hf_dataset != DEFAULT_HF_IMAGENET_DATASET:
        sources.append(args.hf_dataset)
    if sum(bool(source) for source in sources) > 1:
        raise ValueError(
            "Provide at most one of --source, --kaggle-dataset, "
            "--kaggle-competition, or --hf-dataset"
        )
    if args.num_classes <= 0:
        raise ValueError("--num-classes must be positive")
    if args.max_images_per_class is not None and args.max_images_per_class <= 0:
        raise ValueError("--max-images-per-class must be positive")
    return SubsetConfig(
        source=args.source.expanduser().resolve() if args.source is not None else None,
        destination=args.destination.expanduser().resolve(),
        classes=tuple(args.classes) if args.classes is not None else None,
        num_classes=args.num_classes,
        max_images_per_class=args.max_images_per_class,
        seed=args.seed,
        mode=args.mode,
        kaggle_dataset=args.kaggle_dataset,
        kaggle_competition=args.kaggle_competition,
        kaggle_output_dir=args.kaggle_output_dir.expanduser().resolve(),
        hf_dataset=None if uses_local_or_kaggle else args.hf_dataset,
        hf_split=args.hf_split,
        hf_streaming=args.hf_streaming,
        force_download=args.force_download,
    )


def list_class_dirs(source: Path) -> dict[str, Path]:
    if not source.exists():
        raise FileNotFoundError(f"Source folder not found: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source is not a folder: {source}")

    class_dirs = {
        path.name: path
        for path in sorted(source.iterdir())
        if path.is_dir() and any(iter_image_files(path))
    }
    if not class_dirs:
        raise ValueError(f"No class folders with images found in {source}")
    return class_dirs


def resolve_source(config: SubsetConfig) -> Path:
    if config.source is not None:
        return config.source

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required for Kaggle downloads. Install dependencies with "
            "`uv sync`."
        ) from exc

    if config.kaggle_dataset is not None:
        downloaded = Path(
            kagglehub.dataset_download(
                config.kaggle_dataset,
                output_dir=str(config.kaggle_output_dir),
                force_download=config.force_download,
            )
        )
    elif config.kaggle_competition is not None:
        downloaded = Path(
            kagglehub.competition_download(
                config.kaggle_competition,
                output_dir=str(config.kaggle_output_dir),
                force_download=config.force_download,
            )
        )
    else:
        raise ValueError("No source configured")

    return find_imagenet_train_folder(downloaded)


def find_imagenet_train_folder(root: Path) -> Path:
    candidates = [
        root,
        root / "train",
        root / "ILSVRC" / "Data" / "CLS-LOC" / "train",
        root / "Data" / "CLS-LOC" / "train",
    ]
    candidates.extend(path for path in root.rglob("train") if path.is_dir())

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            list_class_dirs(candidate)
        except ValueError:
            continue
        else:
            return candidate

    raise FileNotFoundError(
        "Could not find an ImageNet-style train folder under "
        f"{root}. Expected class folders like n01440764/*.JPEG."
    )


def select_class_dirs(config: SubsetConfig) -> list[Path]:
    source = resolve_source(config)
    print(f"Using source: {source}")
    class_dirs = list_class_dirs(source)
    requested_classes = config.classes or tuple(class_dirs)
    selected_names = [name for name in requested_classes if name in class_dirs]

    if len(selected_names) < config.num_classes:
        missing = [name for name in requested_classes if name not in class_dirs]
        if missing:
            print(f"Missing requested classes: {', '.join(missing)}")
        remaining = [name for name in class_dirs if name not in selected_names]
        selected_names.extend(remaining[: config.num_classes - len(selected_names)])

    selected_names = selected_names[: config.num_classes]
    if len(selected_names) != config.num_classes:
        raise ValueError(
            f"Requested {config.num_classes} classes, found {len(selected_names)}."
        )
    return [class_dirs[name] for name in selected_names]


def iter_image_files(folder: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def materialize_image(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return

    if mode == "symlink":
        destination.symlink_to(source)
    elif mode == "hardlink":
        destination.hardlink_to(source)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def create_subset(config: SubsetConfig) -> None:
    if config.hf_dataset is not None:
        create_huggingface_subset(config)
        return

    rng = random.Random(config.seed)
    selected_dirs = select_class_dirs(config)
    total_images = 0

    config.destination.mkdir(parents=True, exist_ok=True)
    for class_dir in selected_dirs:
        image_paths = list(iter_image_files(class_dir))
        if config.max_images_per_class is not None:
            rng.shuffle(image_paths)
            image_paths = sorted(image_paths[: config.max_images_per_class])

        destination_class_dir = config.destination / class_dir.name
        for image_path in image_paths:
            materialize_image(
                image_path,
                destination_class_dir / image_path.name,
                config.mode,
            )
        total_images += len(image_paths)
        print(f"{class_dir.name}: {len(image_paths)} images")

    print(
        f"Created {total_images} images across {len(selected_dirs)} classes at "
        f"{config.destination}"
    )


def create_huggingface_subset(config: SubsetConfig) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for Hugging Face downloads. Install dependencies "
            "with `uv sync`."
        ) from exc

    dataset = load_dataset(
        config.hf_dataset,
        split=config.hf_split,
        streaming=config.hf_streaming,
        token=True if config.hf_dataset == DEFAULT_HF_IMAGENET_DATASET else None,
    )
    label_to_folder = resolve_hf_label_folders(config, dataset)
    counts = dict.fromkeys(label_to_folder, 0)

    config.destination.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(dataset):
        label = int(sample["label"])
        if label not in label_to_folder:
            continue
        if (
            config.max_images_per_class is not None
            and counts[label] >= config.max_images_per_class
        ):
            if all(count >= config.max_images_per_class for count in counts.values()):
                break
            continue

        destination_class_dir = config.destination / label_to_folder[label]
        destination_class_dir.mkdir(parents=True, exist_ok=True)
        image = sample["image"].convert("RGB")
        image.save(destination_class_dir / f"{index:09d}.jpg", quality=95)
        counts[label] += 1

    for label, count in counts.items():
        print(f"{label_to_folder[label]}: {count} images")
    print(
        f"Created {sum(counts.values())} images across {len(counts)} classes at "
        f"{config.destination}"
    )


def resolve_hf_label_folders(config: SubsetConfig, dataset: object) -> dict[int, str]:
    if config.classes is None:
        labels = tuple(range(config.num_classes))
    else:
        labels = tuple(parse_hf_label(class_name) for class_name in config.classes)
    labels = labels[: config.num_classes]

    label_to_folder = {label: get_hf_label_name(dataset, label) for label in labels}
    if len(label_to_folder) != config.num_classes:
        raise ValueError(
            f"Requested {config.num_classes} classes, resolved "
            f"{len(label_to_folder)} unique Hugging Face labels."
        )
    return label_to_folder


def parse_hf_label(class_name: str) -> int:
    class_text = str(class_name)
    if class_text.isdigit():
        return int(class_text)
    raise ValueError(
        "Hugging Face export accepts numeric label ids in --classes, got "
        f"{class_text!r}."
    )


def get_hf_label_name(dataset: object, label: int) -> str:
    features = getattr(dataset, "features", None)
    if features is not None and "label" in features:
        label_feature = features["label"]
        if hasattr(label_feature, "int2str"):
            return sanitize_folder_name(label_feature.int2str(label))
    return f"class_{label:03d}"


def sanitize_folder_name(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    return safe.strip("._") or "class"


def main() -> None:
    create_subset(build_config(parse_args()))


if __name__ == "__main__":
    main()
