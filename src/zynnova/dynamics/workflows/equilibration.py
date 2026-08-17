from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..config import MDConfig, RelaxationConfig, RunConfig, VelocityConfig, VelocityMode
from ..relaxation import RelaxationResult, relax
from ..results import SimulationResult
from ..runner import run_md


@dataclass(slots=True)
class WorkflowResult:
    relaxation: RelaxationResult | None
    stages: list[SimulationResult]

    @property
    def final_structure(self):
        if self.stages:
            return self.stages[-1].final_structure
        if self.relaxation is not None:
            return self.relaxation.final_structure
        return None


def equilibrate(
    structure: Any,
    calculator: Any,
    md_stages: list[MDConfig],
    *,
    relaxation: RelaxationConfig | None = None,
    output_directory: str | Path = "zynnova-equilibration",
    base_config: RunConfig | None = None,
) -> WorkflowResult:
    if not md_stages:
        raise ValueError("md_stages cannot be empty")
    root = Path(output_directory)
    relaxation_result = None
    current = structure
    if relaxation is not None:
        relaxation_result = relax(
            current,
            calculator,
            relaxation,
            output_directory=root / "relaxation",
        )
        current = relaxation_result.final_structure
    base = base_config or RunConfig()
    results: list[SimulationResult] = []
    for index, md in enumerate(md_stages):
        output = replace(base.output, directory=root / f"stage-{index:02d}")
        velocities = (
            base.velocities
            if index == 0
            else VelocityConfig(mode=VelocityMode.KEEP)
        )
        config = RunConfig(md=md, velocities=velocities, output=output, safety=base.safety)
        result = run_md(current, calculator, config)
        results.append(result)
        current = result.final_structure.to_ase()
        if result.final_structure.arrays.get("momenta") is not None:
            current.set_momenta(result.final_structure.arrays["momenta"])
    return WorkflowResult(relaxation=relaxation_result, stages=results)
