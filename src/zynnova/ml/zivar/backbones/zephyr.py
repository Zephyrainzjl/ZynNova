"""Exact-source-parity alternative registration.

Zephyr deliberately constructs the same reviewed upstream graph, contraction,
readout and acceleration classes as the default implementation.  It exists to
exercise architecture selection and checkpoint identity without introducing a
mathematically approximate reimplementation or copying third-party source.
"""

from __future__ import annotations

from typing import Any

from ..backbone import build_reference_backbone
from .base import register_backbone
from .mace import MaceBackboneAdapter


def build_zephyr_backbone(
    config: Any, *, device: Any = "cpu"
) -> MaceBackboneAdapter:
    core = build_reference_backbone(config, device=device)
    return MaceBackboneAdapter(
        core,
        kind="zephyr",
        architecture="zephyr-source-parity-symmetric-contraction",
    )


def register() -> None:
    register_backbone(
        "zephyr",
        build_zephyr_backbone,
        description="Exact source-parity symmetric-contraction backbone",
        provenance="Independent ZIVAR registration over ACEsuit/mace 0.3.16 classes",
    )


__all__ = ["build_zephyr_backbone", "register"]
