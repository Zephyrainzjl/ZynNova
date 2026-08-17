"""Optional dependency gates for ZIVAR.

The public configuration remains importable without the heavy ML stack.  All
runtime modules use these gates so installing ZynNova without the ``zivar``
extra cannot alter the behaviour of any existing model family.
"""

from __future__ import annotations

import importlib
import io
import sys
import warnings
from contextlib import contextmanager, redirect_stdout
from typing import Any


class MissingZIVARDependency(ImportError):
    """Raised when an optional ZIVAR runtime dependency is unavailable."""


@contextmanager
def upstream_warning_guard() -> Any:
    """Silence only reviewed compatibility noise from the pinned upstream stack.

    e3nn 0.4.4 loads its packaged Wigner constants through the pre-``weights_only``
    API and MACE 0.3.16 constructs several scripted helper modules.  Newer torch
    versions warn about those implementation details even though ZIVAR neither
    loads user-controlled pickle data nor requests TorchScript itself.
    """

    captured = io.StringIO()
    with warnings.catch_warnings(), redirect_stdout(captured):
        warnings.filterwarnings(
            "ignore",
            message=r"Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"You are using `torch\.load` with `weights_only=False`.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script` is deprecated\..*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The TorchScript type system doesn't support instance-level annotations.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"To copy construct from a tensor, it is recommended to use.*",
            category=UserWarning,
        )
        try:
            yield
        finally:
            ignored = (
                "cuequivariance or cuequivariance_torch is not available. "
                "Cuequivariance acceleration will be disabled."
            )
            retained = [
                line for line in captured.getvalue().splitlines() if line.strip() != ignored
            ]
            if retained:
                sys.stdout.write("\n".join(retained) + "\n")


def _require(module: str, *, extra: str = "zivar") -> Any:
    try:
        with upstream_warning_guard():
            return importlib.import_module(module)
    except ImportError as exc:
        raise MissingZIVARDependency(
            f"ZIVAR requires {module!r}; install zynnova[{extra}]"
        ) from exc


def require_torch() -> Any:
    return _require("torch")


def require_e3nn() -> Any:
    return _require("e3nn")


def require_mace() -> Any:
    return _require("mace")


def require_ase() -> Any:
    return _require("ase")


__all__ = [
    "MissingZIVARDependency",
    "require_ase",
    "require_e3nn",
    "require_mace",
    "require_torch",
    "upstream_warning_guard",
]
