"""Electronic-structure backend discovery and construction."""

from .factory import (
    available_dft_backends,
    calculator_from_config,
    create_dft_calculator,
)

__all__ = [
    "available_dft_backends",
    "calculator_from_config",
    "create_dft_calculator",
]
