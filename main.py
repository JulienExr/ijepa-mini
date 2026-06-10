"""Project entrypoint for local I-JEPA experiments.

This follows the official I-JEPA layout: configuration lives in YAML files,
the root ``main.py`` loads that configuration, selects devices, and dispatches
the actual work to the modules under ``src``.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import multiprocessing as mp
import os
import pprint
import random
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

ENTRYPOINTS = {
    "train": "src.training.train:main",
    "linear-probe": "src.evaluation.linear_probe:main",
    "knn": "src.evaluation.knnn:main",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local entrypoint for I-JEPA mini experiments."
    )
    parser.add_argument(
        "task",
        nargs="?",
        choices=sorted(ENTRYPOINTS),
        default="train",
        help="Pipeline step to run.",
    )
    parser.add_argument(
        "--fname",
        "--config",
        dest="fname",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML config file to load.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0"],
        help="Devices to use locally, e.g. cuda:0 cuda:1 or cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed. Each worker adds its rank to this value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and print the resolved run setup without calling the task.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load YAML configs. Install it with "
            "`pip install pyyaml`."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {path}")

    return config


def import_entrypoint(task: str) -> Callable[..., Any]:
    module_name, function_name = ENTRYPOINTS[task].split(":", maxsplit=1)
    module = importlib.import_module(module_name)

    try:
        entrypoint = getattr(module, function_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Task '{task}' expects `{ENTRYPOINTS[task]}`, but `{function_name}` "
            f"does not exist yet."
        ) from exc

    if not callable(entrypoint):
        raise TypeError(f"`{ENTRYPOINTS[task]}` is not callable.")

    return entrypoint


def configure_logging(rank: int) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger("ijepa-mini")


def configure_worker_device(device: str) -> None:
    if device.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]


def seed_everything(seed: int) -> None:
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def process_main(
    rank: int,
    world_size: int,
    task: str,
    config_path: Path,
    devices: list[str],
    seed: int,
    dry_run: bool,
) -> None:
    device = devices[rank]
    configure_worker_device(device)
    logger = configure_logging(rank)
    worker_seed = seed + rank
    seed_everything(worker_seed)

    config = load_config(config_path)
    run_context = {
        "task": task,
        "rank": rank,
        "world_size": world_size,
        "device": device,
        "seed": worker_seed,
        "config_path": str(config_path),
    }
    config.setdefault("runtime", {}).update(run_context)

    if rank == 0:
        logger.info("Resolved run context:")
        pprint.pprint(run_context)
        logger.info("Loaded config:")
        pprint.pprint(config)

    if dry_run:
        return

    entrypoint = import_entrypoint(task)
    entrypoint(config)


def main() -> None:
    args = parse_args()
    config_path = args.fname.expanduser().resolve()
    devices = list(args.devices)
    world_size = len(devices)

    if world_size < 1:
        raise ValueError("At least one device must be provided.")

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    if world_size == 1:
        process_main(
            rank=0,
            world_size=world_size,
            task=args.task,
            config_path=config_path,
            devices=devices,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        return

    processes = []
    for rank in range(world_size):
        process = mp.Process(
            target=process_main,
            args=(
                rank,
                world_size,
                args.task,
                config_path,
                devices,
                args.seed,
                args.dry_run,
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise SystemExit(process.exitcode)


if __name__ == "__main__":
    main()
