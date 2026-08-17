from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..config import Ensemble, MDConfig, RunConfig, VelocityConfig, VelocityMode
from ..results import SimulationResult
from ..runner import run_md


@dataclass(slots=True)
class TemperatureStage:
    temperature_K: float
    steps: int
    ensemble: Ensemble | str = Ensemble.NVT_LANGEVIN

    def to_md_config(self, template: MDConfig) -> MDConfig:
        return replace(
            template,
            ensemble=Ensemble(self.ensemble),
            temperature_K=self.temperature_K,
            steps=self.steps,
        )


def anneal(
    structure: Any,
    calculator: Any,
    stages: list[TemperatureStage],
    *,
    base_config: RunConfig | None = None,
    output_directory: str | Path = "zynnova-anneal",
) -> list[SimulationResult]:
    if not stages:
        raise ValueError("stages cannot be empty")
    base = base_config or RunConfig()
    root = Path(output_directory)
    current = structure
    results: list[SimulationResult] = []
    for index, stage in enumerate(stages):
        md = stage.to_md_config(base.md)
        output = replace(base.output, directory=root / f"stage-{index:02d}")
        velocities = base.velocities if index == 0 else VelocityConfig(mode=VelocityMode.KEEP)
        result = run_md(
            current,
            calculator,
            RunConfig(md=md, velocities=velocities, output=output, safety=base.safety),
        )
        results.append(result)
        current = result.final_structure
    return results
