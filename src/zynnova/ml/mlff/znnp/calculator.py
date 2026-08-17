from __future__ import annotations

from pathlib import Path
from typing import Any

from ....dynamics import TorchPotentialCalculator
from ...common import load_checkpoint, resolve_device
from .config import ZNNPModelConfig
from .model import ZNNP


def znnp_calculator(
    model: ZNNP,
    *,
    device: str = "auto",
    dtype: str = "float32",
    compile_model: bool = False,
) -> TorchPotentialCalculator:
    return TorchPotentialCalculator(
        model,
        device=resolve_device(device),
        dtype=dtype,
        stress_mode="none",
        compile_model=compile_model,
    )


def load_znnp(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> ZNNP:
    payload = load_checkpoint(checkpoint, map_location=device)
    config = ZNNPModelConfig(**payload["model_config"])
    model = ZNNP(config)
    model.load_state_dict(payload["model_state"])
    model.to(resolve_device(device))
    model.eval()
    return model


def load_znnp_calculator(
    checkpoint: str | Path,
    *,
    device: str = "auto",
    dtype: str = "float32",
) -> TorchPotentialCalculator:
    resolved = resolve_device(device)
    model = load_znnp(checkpoint, device=str(resolved))
    return znnp_calculator(model, device=str(resolved), dtype=dtype)


__all__ = ["load_znnp", "load_znnp_calculator", "znnp_calculator"]
