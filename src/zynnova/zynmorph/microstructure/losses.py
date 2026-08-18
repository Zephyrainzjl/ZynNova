"""Descriptor losses available to ZynMorph reconstruction workflows."""

from __future__ import annotations

from typing import Any

import numpy as np

from .registry import LOSSES, register_loss


def _torch_or_numpy(x: Any, fn_numpy, fn_torch):
    module = type(x).__module__
    if module.startswith("torch"):
        return fn_torch(x)
    return fn_numpy(np.asarray(x))


@register_loss("MSE", aliases=("mean_squared_error",))
def mse_loss(delta: Any) -> Any:
    return _torch_or_numpy(delta, lambda x: np.mean(x * x), lambda x: (x * x).mean())


@register_loss("SSE", aliases=("sum_squared_error",))
def sse_loss(delta: Any) -> Any:
    return _torch_or_numpy(delta, lambda x: np.sum(x * x), lambda x: (x * x).sum())


@register_loss("RMS", aliases=("rmse", "root_mean_square"))
def rms_loss(delta: Any) -> Any:
    def np_fn(x):
        return np.sqrt(np.mean(x * x))

    def torch_fn(x):
        return (x.square().mean() + 1.0e-24).sqrt()

    return _torch_or_numpy(delta, np_fn, torch_fn)


@register_loss("L1", aliases=("mae", "mean_absolute_error"))
def l1_loss(delta: Any) -> Any:
    return _torch_or_numpy(delta, lambda x: np.mean(np.abs(x)), lambda x: x.abs().mean())


@register_loss("L2", aliases=("euclidean",))
def l2_loss(delta: Any) -> Any:
    return _torch_or_numpy(
        delta,
        lambda x: np.sqrt(np.sum(x * x)),
        lambda x: (x.square().sum() + 1.0e-24).sqrt(),
    )


def compute_loss(name: str, actual: Any, target: Any) -> Any:
    """Compute a registered loss from two same-shaped descriptor tensors."""

    return LOSSES.get(name)(actual - target)


__all__ = [
    "compute_loss",
    "l1_loss",
    "l2_loss",
    "mse_loss",
    "rms_loss",
    "sse_loss",
]
