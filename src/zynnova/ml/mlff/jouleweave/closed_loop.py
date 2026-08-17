"""SOC/potential-resolved low-scale campaign orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .active_learning import ActiveLearningCampaign, ActiveLearningResult
from .parameter_extraction import AtomisticParameterSurface, ParameterEstimate


ParameterExtractor = Callable[[Any, float, float, float | None], Sequence[ParameterEstimate]]
StructureFactory = Callable[[float, float, float | None], Sequence[Any]]


@dataclass(frozen=True, slots=True)
class LowScaleCampaignPoint:
    soc: float
    temperature_K: float
    electrode_potential_V: float | None
    active_learning: ActiveLearningResult
    parameters: tuple[ParameterEstimate, ...]


@dataclass(slots=True)
class LowScaleCampaignResult:
    points: list[LowScaleCampaignPoint]
    surfaces: Mapping[str, AtomisticParameterSurface]


@dataclass(slots=True)
class LowScaleClosedLoop:
    """Execute active learning and parameter extraction at requested states."""

    campaign_factory: Callable[[float, float, float | None], ActiveLearningCampaign]
    structure_factory: StructureFactory
    parameter_extractor: ParameterExtractor
    metadata: Mapping[str, object] = field(default_factory=dict)

    def run(
        self,
        soc_grid: Sequence[float],
        temperature_grid_K: Sequence[float],
        electrode_potential_grid_V: Sequence[float | None] = (None,),
    ) -> LowScaleCampaignResult:
        points: list[LowScaleCampaignPoint] = []
        by_name: dict[str, list[ParameterEstimate]] = {}
        for soc in soc_grid:
            if not 0.0 <= float(soc) <= 1.0:
                raise ValueError("SOC grid must lie in [0,1]")
            for temperature_K in temperature_grid_K:
                if temperature_K <= 0.0:
                    raise ValueError("temperature grid must be positive")
                for potential in electrode_potential_grid_V:
                    seeds = list(
                        self.structure_factory(float(soc), float(temperature_K), potential)
                    )
                    campaign = self.campaign_factory(
                        float(soc), float(temperature_K), potential
                    )
                    active_result = campaign.run(seeds)
                    parameters = tuple(
                        self.parameter_extractor(
                            active_result.models[0],
                            float(soc),
                            float(temperature_K),
                            potential,
                        )
                    )
                    for estimate in parameters:
                        by_name.setdefault(estimate.name, []).append(estimate)
                    points.append(
                        LowScaleCampaignPoint(
                            float(soc),
                            float(temperature_K),
                            potential,
                            active_result,
                            parameters,
                        )
                    )
        surfaces = {
            name: AtomisticParameterSurface(tuple(estimates))
            for name, estimates in by_name.items()
        }
        return LowScaleCampaignResult(points, surfaces)


__all__ = [
    "LowScaleCampaignPoint",
    "LowScaleCampaignResult",
    "LowScaleClosedLoop",
]
