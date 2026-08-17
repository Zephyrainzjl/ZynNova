"""On-demand atomistic refresh and uncertainty-aware cross-scale solve loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from ..exceptions import PropertyResolutionError
from .coupling import MultiscaleCoordinator
from .properties import MaterialProperty, PropertyRequest, convert_property


class ParameterEstimateLike(Protocol):
    name: str
    value: float
    unit: str
    soc: float
    temperature_K: float
    electrode_potential_V: float | None
    standard_uncertainty: float
    metadata: Mapping[str, object]


class ParameterSurfaceLike(Protocol):
    estimates: Sequence[ParameterEstimateLike]

    def evaluate(
        self,
        soc: float,
        temperature_K: float,
        electrode_potential_V: float | None = None,
    ) -> ParameterEstimateLike: ...


SurfaceFactory = Callable[[Sequence[ParameterEstimateLike]], ParameterSurfaceLike]


@dataclass(slots=True)
class AtomisticSurfaceProvider:
    surfaces: Mapping[str, ParameterSurfaceLike]
    source: str = "jouleweave-atomistic-surface"
    surface_factory: SurfaceFactory | None = None

    def __post_init__(self) -> None:
        self.surfaces = dict(self.surfaces)

    def resolve(self, request: PropertyRequest) -> MaterialProperty | None:
        surface = self.surfaces.get(request.name)
        if surface is None:
            return None
        potential = request.context.get("electrode_potential_V")
        estimate = surface.evaluate(
            request.soc,
            request.temperature_K,
            None if potential is None else float(potential),
        )
        return convert_property(
            MaterialProperty(
                name=estimate.name,
                value=estimate.value,
                unit=estimate.unit,
                source=self.source,
                soc=estimate.soc,
                temperature_K=estimate.temperature_K,
                standard_uncertainty=estimate.standard_uncertainty,
                metadata={
                    **dict(estimate.metadata),
                    "electrode_potential_V": estimate.electrode_potential_V,
                },
            ),
            request.unit,
        )

    def extend(self, estimates: Sequence[ParameterEstimateLike]) -> None:
        grouped: dict[str, list[ParameterEstimateLike]] = {
            name: list(surface.estimates) for name, surface in self.surfaces.items()
        }
        for estimate in estimates:
            grouped.setdefault(estimate.name, []).append(estimate)
        rebuilt: dict[str, ParameterSurfaceLike] = {}
        for name, values in grouped.items():
            existing = self.surfaces.get(name)
            if existing is not None:
                try:
                    rebuilt[name] = type(existing)(tuple(values))
                    continue
                except (TypeError, ValueError):
                    pass
            if self.surface_factory is None:
                raise PropertyResolutionError(
                    "a surface_factory is required when a refreshed property "
                    f"creates the new surface {name!r}"
                )
            rebuilt[name] = self.surface_factory(tuple(values))
        self.surfaces = rebuilt


RefreshCallback = Callable[[PropertyRequest], Sequence[ParameterEstimateLike]]


@dataclass(slots=True)
class OnDemandAtomisticProvider:
    """Refresh a property surface synchronously when uncertainty is too high."""

    surface_provider: AtomisticSurfaceProvider
    refresh: RefreshCallback
    maximum_relative_uncertainty: float = 0.15
    maximum_refreshes: int = 100
    refresh_count: int = field(default=0, init=False)

    def resolve(self, request: PropertyRequest) -> MaterialProperty | None:
        result = self.surface_provider.resolve(request)
        needs_refresh = result is None
        if result is not None:
            relative = result.relative_uncertainty()
            needs_refresh = relative is not None and relative > self.maximum_relative_uncertainty
        if not needs_refresh:
            return result
        if self.refresh_count >= self.maximum_refreshes:
            raise PropertyResolutionError("atomistic refresh budget has been exhausted")
        estimates = tuple(self.refresh(request))
        self.refresh_count += 1
        if not estimates:
            raise PropertyResolutionError(
                f"atomistic refresh returned no estimates for {request.name!r}"
            )
        self.surface_provider.extend(estimates)
        refreshed = self.surface_provider.resolve(request)
        if refreshed is None:
            raise PropertyResolutionError(
                f"atomistic refresh did not produce {request.name!r}"
            )
        return refreshed


@dataclass(frozen=True, slots=True)
class CrossScaleStepRecord:
    time_s: float
    soc: float
    temperature_K: float
    current_A: float
    property_relative_uncertainty: float
    continuum_state: Any


@dataclass(slots=True)
class CrossScaleSimulationResult:
    records: list[CrossScaleStepRecord]
    final_state: Any


class CrossScaleSimulationLoop:
    """Update SOC/T-dependent atomistic parameters inside a continuum time loop."""

    def __init__(
        self,
        continuum_model: Any,
        parameter_target: Any,
        coordinator: MultiscaleCoordinator,
    ) -> None:
        self.continuum_model = continuum_model
        self.parameter_target = parameter_target
        self.coordinator = coordinator

    def run(
        self,
        initial_state: Any,
        time_s: np.ndarray,
        current_A: np.ndarray,
    ) -> CrossScaleSimulationResult:
        times = np.asarray(time_s, dtype=float)
        currents = np.asarray(current_A, dtype=float)
        if times.ndim != 1 or currents.shape != times.shape or len(times) < 2:
            raise ValueError("cross-scale time/current arrays are invalid")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("cross-scale time grid must be increasing")
        state = initial_state
        records: list[CrossScaleStepRecord] = []
        for left, right, current in zip(times[:-1], times[1:], currents[:-1], strict=True):
            soc = _state_soc(state)
            temperature = _state_temperature(state)
            properties = self.coordinator.update(
                self.parameter_target,
                soc,
                temperature,
            )
            state = self.continuum_model.step(state, float(current), float(right - left))
            relative_uncertainties = [
                value.relative_uncertainty()
                for value in properties.values()
                if value.relative_uncertainty() is not None
            ]
            uncertainty = max(relative_uncertainties, default=0.0)
            records.append(
                CrossScaleStepRecord(
                    time_s=float(right),
                    soc=soc,
                    temperature_K=temperature,
                    current_A=float(current),
                    property_relative_uncertainty=float(uncertainty),
                    continuum_state=state.copy() if hasattr(state, "copy") else state,
                )
            )
        return CrossScaleSimulationResult(records, state)


def _state_soc(state: Any) -> float:
    value = getattr(state, "soc", None)
    if value is not None:
        return float(np.clip(value, 0.0, 1.0))
    metadata = getattr(state, "metadata", {})
    if isinstance(metadata, Mapping) and "soc" in metadata:
        return float(np.clip(metadata["soc"], 0.0, 1.0))
    return 0.5


def _state_temperature(state: Any) -> float:
    value = getattr(state, "temperature_K", 298.15)
    return float(np.mean(np.asarray(value, dtype=float)))


__all__ = [
    "AtomisticSurfaceProvider",
    "CrossScaleSimulationLoop",
    "CrossScaleSimulationResult",
    "CrossScaleStepRecord",
    "OnDemandAtomisticProvider",
    "ParameterEstimateLike",
    "ParameterSurfaceLike",
]
