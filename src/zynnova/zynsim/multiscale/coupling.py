"""SOC/temperature-aware micro-to-continuum property coupling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from ..battery.p2d.parameters import P2DParameters
from ..exceptions import PropertyResolutionError
from .properties import MaterialProperty, PropertyProvider, PropertyRequest, convert_property


Validator = Callable[[float], bool]
Transform = Callable[[float], float]


@dataclass(frozen=True, slots=True)
class PropertyBinding:
    property_name: str
    target_path: str
    unit: str
    validator: Validator = lambda value: np.isfinite(value) and value > 0.0
    transform: Transform = lambda value: value
    required: bool = True
    maximum_relative_uncertainty: float | None = None


@dataclass(frozen=True, slots=True)
class CrossScaleRecord:
    index: int
    soc: float
    temperature_K: float
    properties: Mapping[str, MaterialProperty]

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "soc": self.soc,
            "temperature_K": self.temperature_K,
            "properties": {
                path: {
                    "name": value.name,
                    "value": np.asarray(value.value).tolist(),
                    "unit": value.unit,
                    "source": value.source,
                    "standard_uncertainty": (
                        None
                        if value.standard_uncertainty is None
                        else np.asarray(value.standard_uncertainty).tolist()
                    ),
                    "metadata": dict(value.metadata),
                }
                for path, value in self.properties.items()
            },
        }


@dataclass(slots=True)
class MultiscaleCoordinator:
    """Update bound continuum properties at controlled SOC/temperature intervals."""

    provider: PropertyProvider
    bindings: tuple[PropertyBinding, ...]
    soc_update_interval: float = 0.01
    temperature_update_interval_K: float = 1.0
    context: Mapping[str, object] = field(default_factory=dict)
    records: list[CrossScaleRecord] = field(default_factory=list, init=False)
    _last_soc: float | None = field(default=None, init=False, repr=False)
    _last_temperature_K: float | None = field(default=None, init=False, repr=False)
    _last_properties: dict[str, MaterialProperty] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.soc_update_interval < 0.0 or self.temperature_update_interval_K < 0.0:
            raise ValueError("multiscale update intervals cannot be negative")
        if not self.bindings:
            raise ValueError("at least one cross-scale property binding is required")
        targets = [binding.target_path for binding in self.bindings]
        if len(set(targets)) != len(targets):
            raise ValueError("cross-scale target paths must be unique")
        self.context = dict(self.context)

    def update(
        self,
        target: Any,
        soc: float,
        temperature_K: float,
        *,
        force: bool = False,
    ) -> Mapping[str, MaterialProperty]:
        should_resolve = force or self._last_soc is None or (
            abs(soc - self._last_soc) >= self.soc_update_interval
            or abs(temperature_K - self._last_temperature_K)
            >= self.temperature_update_interval_K
        )
        if not should_resolve:
            return dict(self._last_properties)
        resolved: dict[str, MaterialProperty] = {}
        pending: list[tuple[PropertyBinding, MaterialProperty, float]] = []
        for binding in self.bindings:
            request = PropertyRequest(
                name=binding.property_name,
                unit=binding.unit,
                soc=float(soc),
                temperature_K=float(temperature_K),
                context=self.context,
            )
            property_value = self.provider.resolve(request)
            if property_value is None:
                if binding.required:
                    raise PropertyResolutionError(
                        f"no provider resolved required property {binding.property_name!r}"
                    )
                continue
            property_value = convert_property(property_value, binding.unit)
            relative_uncertainty = property_value.relative_uncertainty()
            if (
                binding.maximum_relative_uncertainty is not None
                and relative_uncertainty is not None
                and relative_uncertainty > binding.maximum_relative_uncertainty
            ):
                raise PropertyResolutionError(
                    f"{binding.property_name!r} relative uncertainty "
                    f"{relative_uncertainty:.3g} exceeds "
                    f"{binding.maximum_relative_uncertainty:.3g}"
                )
            scalar = binding.transform(property_value.scalar())
            if not binding.validator(scalar):
                raise PropertyResolutionError(
                    f"{binding.property_name!r} resolved to rejected value {scalar!r}"
                )
            pending.append((binding, property_value, scalar))
            resolved[binding.target_path] = property_value

        # Apply only after every required query and validation has succeeded.
        for binding, _property, scalar in pending:
            _set_nested_attribute(target, binding.target_path, scalar)
        self._last_soc = float(soc)
        self._last_temperature_K = float(temperature_K)
        self._last_properties = resolved
        self.records.append(
            CrossScaleRecord(
                index=len(self.records),
                soc=float(soc),
                temperature_K=float(temperature_K),
                properties=dict(resolved),
            )
        )
        return dict(resolved)

    def p2d_material_update(self) -> Callable[[P2DParameters, float, float], None]:
        def update(parameters: P2DParameters, soc: float, temperature_K: float) -> None:
            self.update(parameters, soc, temperature_K)

        return update


def default_p2d_bindings() -> tuple[PropertyBinding, ...]:
    """Bindings for common atomistic/mesoscopic P2D transport parameters."""

    return (
        PropertyBinding(
            "negative_solid_diffusivity",
            "negative.solid_diffusivity_m2_s",
            "m2 s-1",
        ),
        PropertyBinding(
            "positive_solid_diffusivity",
            "positive.solid_diffusivity_m2_s",
            "m2 s-1",
        ),
        PropertyBinding(
            "negative_electronic_conductivity",
            "negative.electronic_conductivity_S_m",
            "S m-1",
            required=False,
        ),
        PropertyBinding(
            "positive_electronic_conductivity",
            "positive.electronic_conductivity_S_m",
            "S m-1",
            required=False,
        ),
        PropertyBinding(
            "electrolyte_diffusivity",
            "electrolyte.diffusivity_m2_s",
            "m2 s-1",
            required=False,
        ),
        PropertyBinding(
            "electrolyte_conductivity",
            "electrolyte.ionic_conductivity_S_m",
            "S m-1",
            required=False,
        ),
    )


def _set_nested_attribute(target: Any, path: str, value: float) -> None:
    components = path.split(".")
    if not components or any(not component for component in components):
        raise PropertyResolutionError(f"invalid target path {path!r}")
    owner = target
    for component in components[:-1]:
        if not hasattr(owner, component):
            raise PropertyResolutionError(f"unknown property target {path!r}")
        owner = getattr(owner, component)
    leaf = components[-1]
    if not hasattr(owner, leaf):
        raise PropertyResolutionError(f"unknown property target {path!r}")
    setattr(owner, leaf, float(value))


__all__ = [
    "CrossScaleRecord",
    "MultiscaleCoordinator",
    "PropertyBinding",
    "default_p2d_bindings",
]
