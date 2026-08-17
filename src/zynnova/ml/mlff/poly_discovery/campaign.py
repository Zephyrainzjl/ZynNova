from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .active import ActiveDiscoveryLoop
from .config import (
    ActiveLearningConfig,
    MechanismConstraint,
    MechanismDiscoveryConfig,
    MechanismGenerationConfig,
    PolymerPotentialConfig,
)
from .datasets import observations_from_material_samples
from .discovery import MechanismDiscoveryEngine
from .generation import (
    MechanismGuidedGenerationResult,
    generate_mechanism_guided_polymers,
)
from .potential import train_polymer_potential
from .physics_learning import (
    PhysicsLearningConfig,
    PhysicsLearningEngine,
    PhysicsLearningReport,
    VariableSpec,
)
from .reporting import discovery_report_from_dict
from .schema import ActiveCandidate, DiscoveryReport, Observation, SimulationRequest
from .simulation import PolymerMechanismSimulator


@dataclass(slots=True)
class PolymerDiscoveryCampaign:
    """Stateful closed loop connecting data, mechanisms, simulation, and generation."""

    observations: list[Observation] = field(default_factory=list)
    discovery_config: MechanismDiscoveryConfig = field(
        default_factory=MechanismDiscoveryConfig
    )
    active_config: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    generation_config: MechanismGenerationConfig = field(
        default_factory=MechanismGenerationConfig
    )
    report: DiscoveryReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"
    physics_config: PhysicsLearningConfig = field(
        default_factory=PhysicsLearningConfig
    )

    def add_observations(
        self,
        observations: Sequence[Observation],
        *,
        replace_existing: bool = True,
    ) -> None:
        positions = {
            observation.sample_id: index
            for index, observation in enumerate(self.observations)
        }
        for observation in observations:
            position = positions.get(observation.sample_id)
            if position is None:
                positions[observation.sample_id] = len(self.observations)
                self.observations.append(observation)
            elif replace_existing:
                self.observations[position] = observation

    def add_material_samples(
        self,
        samples: Sequence[Any],
        **adapter_kwargs: Any,
    ) -> list[Observation]:
        converted = observations_from_material_samples(samples, **adapter_kwargs)
        self.add_observations(converted)
        return converted

    def discover(
        self,
        target: str,
        *,
        feature_names: Sequence[str] | None = None,
        control_names: Sequence[str] = (),
        include_symbolic_law: bool = True,
        include_advanced_physics: bool | None = None,
        variable_specs: Mapping[str, VariableSpec] | None = None,
    ) -> DiscoveryReport:
        enable_physics = (
            self.physics_config.enabled
            if include_advanced_physics is None
            else bool(include_advanced_physics)
        )
        physics_config = (
            replace(self.physics_config, enabled=True)
            if enable_physics
            else None
        )
        self.report = MechanismDiscoveryEngine(self.discovery_config).discover(
            self.observations,
            target,
            feature_names=feature_names,
            control_names=control_names,
            include_symbolic_law=include_symbolic_law,
            physics_learning_config=physics_config,
            variable_specs=variable_specs,
        )
        return self.report

    def discover_physics(
        self,
        target: str,
        *,
        feature_names: Sequence[str] | None = None,
        variable_specs: Mapping[str, VariableSpec] | None = None,
        config: PhysicsLearningConfig | None = None,
    ) -> PhysicsLearningReport:
        """Run the neural-symbolic layer independently and attach its report."""

        resolved = config or replace(self.physics_config, enabled=True)
        result = PhysicsLearningEngine(resolved).discover(
            self.observations,
            target,
            feature_names=feature_names,
            variable_specs=variable_specs,
        )
        if self.report is not None and self.report.target == target:
            self.report.physics_learning = result
            self.report.schema_version = "1.1"
        return result

    def propose_simulations(
        self,
        candidates: Sequence[ActiveCandidate],
    ) -> tuple[SimulationRequest, ...]:
        if self.report is None:
            raise RuntimeError("discover a mechanism before proposing simulations")
        return ActiveDiscoveryLoop(self.active_config).propose(
            candidates,
            self.observations,
            self.report,
        )

    def generate(
        self,
        generator: Any,
        requested_properties: Mapping[str, float],
        *,
        process_conditions: Mapping[str, float] | None = None,
        predictor: Any | None = None,
        property_constraints: Sequence[Any] = (),
        polyloom_config: Any | None = None,
    ) -> MechanismGuidedGenerationResult:
        if self.report is None:
            raise RuntimeError("discover a mechanism before mechanism-guided generation")
        return generate_mechanism_guided_polymers(
            generator,
            self.report,
            requested_properties,
            self.observations,
            process_conditions=process_conditions,
            predictor=predictor,
            property_constraints=property_constraints,
            config=self.generation_config,
            polyloom_config=polyloom_config,
        )

    @staticmethod
    def train_potential(
        config: PolymerPotentialConfig | None = None,
        *,
        source: Any | None = None,
    ) -> Any:
        return train_polymer_potential(config, source=source)

    @staticmethod
    def simulator(
        potential: Any,
        **kwargs: Any,
    ) -> PolymerMechanismSimulator:
        return PolymerMechanismSimulator(potential, **kwargs)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "observations": [asdict(observation) for observation in self.observations],
            "discovery_config": asdict(self.discovery_config),
            "active_config": asdict(self.active_config),
            "generation_config": asdict(self.generation_config),
            "physics_config": asdict(self.physics_config),
            "report": None if self.report is None else self.report.to_dict(),
            "metadata": self.metadata,
        }
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> PolymerDiscoveryCampaign:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        generation_values = dict(payload.get("generation_config", {}))
        generation_values["constraints"] = tuple(
            constraint
            if isinstance(constraint, MechanismConstraint)
            else MechanismConstraint(**constraint)
            for constraint in generation_values.get("constraints", ())
        )
        report_payload = payload.get("report")
        return cls(
            observations=[
                Observation(**observation)
                for observation in payload.get("observations", ())
            ],
            discovery_config=MechanismDiscoveryConfig(
                **payload.get("discovery_config", {})
            ),
            active_config=ActiveLearningConfig(**payload.get("active_config", {})),
            generation_config=MechanismGenerationConfig(**generation_values),
            physics_config=PhysicsLearningConfig(
                **payload.get("physics_config", {})
            ),
            report=(
                None
                if report_payload is None
                else discovery_report_from_dict(report_payload)
            ),
            metadata=dict(payload.get("metadata", {})),
            schema_version=str(payload.get("schema_version", "1.0")),
        )


__all__ = ["PolymerDiscoveryCampaign"]
