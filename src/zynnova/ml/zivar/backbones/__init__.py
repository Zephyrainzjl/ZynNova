"""Built-in and extensible ZIVAR local-backbone registry."""

from __future__ import annotations

from .base import (
    BACKBONE_CONTRACT_VERSION,
    BackboneAdapter,
    BackboneCapabilities,
    BackboneManifest,
    BackboneRegistration,
    backbone_registration,
    build_backbone,
    load_backbone_plugins,
    register_backbone,
    registered_backbones,
    validate_backbone_output,
)
from .convolution import register as _register_convolution
from .mace import register as _register_mace
from .zephyr import register as _register_zephyr

_register_mace()
_register_zephyr()
_register_convolution()

# Zodiac is imported last because it owns the native e3nn tensor-product code.
from .zodiac import register as _register_zodiac  # noqa: E402

_register_zodiac()

__all__ = [
    "BACKBONE_CONTRACT_VERSION",
    "BackboneAdapter",
    "BackboneCapabilities",
    "BackboneManifest",
    "BackboneRegistration",
    "backbone_registration",
    "build_backbone",
    "load_backbone_plugins",
    "register_backbone",
    "registered_backbones",
    "validate_backbone_output",
]
