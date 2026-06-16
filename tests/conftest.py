from __future__ import annotations

import pytest
import torch


def _available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.fixture(params=_available_devices())
def device(request) -> torch.device:
    return torch.device(request.param)
