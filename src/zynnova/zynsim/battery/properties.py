"""State-dependent battery material-property laws."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, TypeAlias

import numpy as np

from ..constants import GAS_CONSTANT


class PropertyCallable(Protocol):
    def __call__(self, soc: float, temperature_K: float) -> float: ...


PropertyValue: TypeAlias = float | PropertyCallable


def evaluate_property(
    value: PropertyValue,
    soc: float,
    temperature_K: float,
    *,
    name: str,
    positive: bool = True,
) -> float:
    result = float(value(float(soc), float(temperature_K)) if callable(value) else value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"resolved property {name!r} must be {qualifier}; got {result!r}")
    return result


@dataclass(frozen=True, slots=True)
class ArrheniusLaw:
    """Arrhenius scaling around a reference temperature."""

    reference_value: float
    activation_energy_J_mol: float
    reference_temperature_K: float = 298.15
    soc_multiplier: Callable[[float], float] | None = None

    def __call__(self, soc: float, temperature_K: float) -> float:
        if temperature_K <= 0.0:
            raise ValueError("absolute temperature must be positive")
        multiplier = 1.0 if self.soc_multiplier is None else float(self.soc_multiplier(soc))
        return float(
            self.reference_value
            * multiplier
            * np.exp(
                -self.activation_energy_J_mol
                / GAS_CONSTANT
                * (1.0 / temperature_K - 1.0 / self.reference_temperature_K)
            )
        )


@dataclass(frozen=True, slots=True)
class SOCPropertyTable:
    """Monotone 1-D interpolation with explicit out-of-range behavior."""

    soc: tuple[float, ...]
    values: tuple[float, ...]
    extrapolation: str = "error"

    def __post_init__(self) -> None:
        if len(self.soc) < 2 or len(self.soc) != len(self.values):
            raise ValueError("SOC table needs at least two aligned points")
        if np.any(np.diff(self.soc) <= 0.0):
            raise ValueError("SOC table coordinates must be strictly increasing")
        if self.extrapolation not in {"error", "clamp", "linear"}:
            raise ValueError("extrapolation must be 'error', 'clamp', or 'linear'")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("SOC table contains non-finite values")

    def __call__(self, soc: float, temperature_K: float) -> float:
        del temperature_K
        x = float(soc)
        grid = np.asarray(self.soc)
        values = np.asarray(self.values)
        if self.extrapolation == "error" and not grid[0] <= x <= grid[-1]:
            raise ValueError(f"SOC={x} is outside [{grid[0]}, {grid[-1]}]")
        if self.extrapolation == "clamp":
            x = float(np.clip(x, grid[0], grid[-1]))
        if x < grid[0]:
            slope = (values[1] - values[0]) / (grid[1] - grid[0])
            return float(values[0] + slope * (x - grid[0]))
        if x > grid[-1]:
            slope = (values[-1] - values[-2]) / (grid[-1] - grid[-2])
            return float(values[-1] + slope * (x - grid[-1]))
        return float(np.interp(x, grid, values))


@dataclass(frozen=True, slots=True)
class SOCTemperatureTable:
    """Bilinear interpolation over a rectangular SOC/temperature grid."""

    soc: tuple[float, ...]
    temperature_K: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    clamp: bool = False

    def __post_init__(self) -> None:
        array = np.asarray(self.values, dtype=float)
        if array.shape != (len(self.soc), len(self.temperature_K)):
            raise ValueError("table values must have shape (n_soc, n_temperature)")
        if len(self.soc) < 2 or len(self.temperature_K) < 2:
            raise ValueError("bilinear table needs at least two points per axis")
        if np.any(np.diff(self.soc) <= 0.0) or np.any(np.diff(self.temperature_K) <= 0.0):
            raise ValueError("table axes must be strictly increasing")

    def __call__(self, soc: float, temperature_K: float) -> float:
        x_grid = np.asarray(self.soc, dtype=float)
        t_grid = np.asarray(self.temperature_K, dtype=float)
        x = float(soc)
        t = float(temperature_K)
        if self.clamp:
            x = float(np.clip(x, x_grid[0], x_grid[-1]))
            t = float(np.clip(t, t_grid[0], t_grid[-1]))
        elif not (x_grid[0] <= x <= x_grid[-1] and t_grid[0] <= t <= t_grid[-1]):
            raise ValueError("SOC/temperature query is outside the tabulated domain")
        i = int(np.clip(np.searchsorted(x_grid, x) - 1, 0, len(x_grid) - 2))
        j = int(np.clip(np.searchsorted(t_grid, t) - 1, 0, len(t_grid) - 2))
        wx = (x - x_grid[i]) / (x_grid[i + 1] - x_grid[i])
        wt = (t - t_grid[j]) / (t_grid[j + 1] - t_grid[j])
        values = np.asarray(self.values, dtype=float)
        return float(
            (1.0 - wx) * (1.0 - wt) * values[i, j]
            + wx * (1.0 - wt) * values[i + 1, j]
            + (1.0 - wx) * wt * values[i, j + 1]
            + wx * wt * values[i + 1, j + 1]
        )


__all__ = [
    "ArrheniusLaw",
    "PropertyCallable",
    "PropertyValue",
    "SOCPropertyTable",
    "SOCTemperatureTable",
    "evaluate_property",
]
