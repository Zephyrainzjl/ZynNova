"""Typed, unit-checked material-property exchange across simulation scales."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import numpy as np

from ..exceptions import PropertyResolutionError


PropertyArray = float | np.ndarray


@dataclass(frozen=True, slots=True)
class PropertyRequest:
    name: str
    unit: str
    soc: float
    temperature_K: float
    context: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.unit:
            raise ValueError("property request name and unit cannot be empty")
        if not 0.0 <= self.soc <= 1.0 or self.temperature_K <= 0.0:
            raise ValueError("property request SOC/temperature is invalid")


@dataclass(frozen=True, slots=True)
class MaterialProperty:
    name: str
    value: PropertyArray
    unit: str
    source: str
    soc: float
    temperature_K: float
    standard_uncertainty: PropertyArray | None = None
    valid_soc: tuple[float, float] = (0.0, 1.0)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.value, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"material property {self.name!r} contains non-finite values")
        if not self.source or not self.unit:
            raise ValueError("material property source and unit cannot be empty")
        if not self.valid_soc[0] <= self.soc <= self.valid_soc[1]:
            raise ValueError("material property is outside its declared SOC domain")
        if self.standard_uncertainty is not None:
            uncertainty = np.asarray(self.standard_uncertainty, dtype=np.float64)
            if np.any(uncertainty < 0.0) or not np.all(np.isfinite(uncertainty)):
                raise ValueError("property uncertainty must be finite and non-negative")

    def scalar(self) -> float:
        value = np.asarray(self.value, dtype=np.float64)
        if value.ndim != 0:
            raise PropertyResolutionError(f"property {self.name!r} is not scalar")
        return float(value)

    def relative_uncertainty(self) -> float | None:
        if self.standard_uncertainty is None:
            return None
        value_norm = float(np.linalg.norm(np.asarray(self.value, dtype=float)))
        uncertainty_norm = float(
            np.linalg.norm(np.asarray(self.standard_uncertainty, dtype=float))
        )
        return uncertainty_norm / max(value_norm, 1.0e-30)


class PropertyProvider(Protocol):
    def resolve(self, request: PropertyRequest) -> MaterialProperty | None: ...


_UNIT_SCALE: dict[tuple[str, str], float] = {
    ("GPa", "Pa"): 1.0e9,
    ("MPa", "Pa"): 1.0e6,
    ("Pa", "GPa"): 1.0e-9,
    ("cm2 s-1", "m2 s-1"): 1.0e-4,
    ("cm^2/s", "m2 s-1"): 1.0e-4,
    ("m^2/s", "m2 s-1"): 1.0,
    ("S/m", "S m-1"): 1.0,
    ("J/(m3 K)", "J m-3 K-1"): 1.0,
}


def convert_property(
    property_value: MaterialProperty,
    target_unit: str,
) -> MaterialProperty:
    if property_value.unit == target_unit:
        return property_value
    try:
        scale = _UNIT_SCALE[(property_value.unit, target_unit)]
    except KeyError as exc:
        raise PropertyResolutionError(
            f"cannot convert {property_value.name!r} from {property_value.unit!r} "
            f"to {target_unit!r}"
        ) from exc
    uncertainty = (
        None
        if property_value.standard_uncertainty is None
        else np.asarray(property_value.standard_uncertainty) * abs(scale)
    )
    converted = np.asarray(property_value.value) * scale
    return MaterialProperty(
        name=property_value.name,
        value=float(converted) if converted.ndim == 0 else converted,
        unit=target_unit,
        source=property_value.source,
        soc=property_value.soc,
        temperature_K=property_value.temperature_K,
        standard_uncertainty=uncertainty,
        valid_soc=property_value.valid_soc,
        metadata=property_value.metadata,
    )


@dataclass(slots=True)
class TabulatedPropertyProvider:
    """Linear interpolation over SOC for deterministic reference data."""

    tables: Mapping[str, tuple[np.ndarray, np.ndarray, str]]
    source: str = "tabulated"
    extrapolation: str = "error"

    def __post_init__(self) -> None:
        if self.extrapolation not in {"error", "clamp"}:
            raise ValueError("tabulated extrapolation must be 'error' or 'clamp'")
        normalized: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
        for name, (soc, values, unit) in self.tables.items():
            x = np.asarray(soc, dtype=np.float64)
            y = np.asarray(values, dtype=np.float64)
            if x.ndim != 1 or y.shape[0] != len(x) or len(x) < 2:
                raise ValueError(f"invalid table for {name!r}")
            if np.any(np.diff(x) <= 0.0) or not np.all(np.isfinite(y)):
                raise ValueError(f"table for {name!r} is not monotone/finite")
            normalized[str(name)] = (x, y, str(unit))
        self.tables = normalized

    def resolve(self, request: PropertyRequest) -> MaterialProperty | None:
        entry = self.tables.get(request.name)
        if entry is None:
            return None
        soc_grid, values, unit = entry
        soc = request.soc
        if not soc_grid[0] <= soc <= soc_grid[-1]:
            if self.extrapolation == "error":
                raise PropertyResolutionError(
                    f"{request.name!r} requested at SOC={soc}, outside table domain"
                )
            soc = float(np.clip(soc, soc_grid[0], soc_grid[-1]))
        flat = values.reshape(len(soc_grid), -1)
        interpolated = np.asarray(
            [np.interp(soc, soc_grid, flat[:, column]) for column in range(flat.shape[1])]
        ).reshape(values.shape[1:])
        value: PropertyArray = (
            float(interpolated) if interpolated.ndim == 0 else interpolated
        )
        return convert_property(
            MaterialProperty(
                name=request.name,
                value=value,
                unit=unit,
                source=self.source,
                soc=request.soc,
                temperature_K=request.temperature_K,
                valid_soc=(float(soc_grid[0]), float(soc_grid[-1])),
            ),
            request.unit,
        )


@dataclass(slots=True)
class CompositePropertyProvider:
    """Query providers in priority order and reject silent fabrication."""

    providers: tuple[PropertyProvider, ...]

    def resolve(self, request: PropertyRequest) -> MaterialProperty | None:
        failures: list[str] = []
        for provider in self.providers:
            try:
                result = provider.resolve(request)
            except PropertyResolutionError as exc:
                failures.append(str(exc))
                continue
            if result is not None:
                return convert_property(result, request.unit)
        if failures:
            raise PropertyResolutionError("; ".join(failures))
        return None


@dataclass(slots=True)
class CachingPropertyProvider:
    provider: PropertyProvider
    soc_resolution: float = 1.0e-4
    temperature_resolution_K: float = 0.1
    _cache: dict[tuple[str, str, int, int], MaterialProperty | None] = field(
        default_factory=dict, init=False, repr=False
    )

    def resolve(self, request: PropertyRequest) -> MaterialProperty | None:
        key = (
            request.name,
            request.unit,
            round(request.soc / self.soc_resolution),
            round(request.temperature_K / self.temperature_resolution_K),
        )
        if key not in self._cache:
            self._cache[key] = self.provider.resolve(request)
        return self._cache[key]

    def clear(self) -> None:
        self._cache.clear()


__all__ = [
    "CachingPropertyProvider",
    "CompositePropertyProvider",
    "MaterialProperty",
    "PropertyArray",
    "PropertyProvider",
    "PropertyRequest",
    "TabulatedPropertyProvider",
    "convert_property",
]
