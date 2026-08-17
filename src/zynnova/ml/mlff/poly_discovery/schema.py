from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .physics_learning.schema import PhysicsLearningReport

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class EvidenceLevel(StrEnum):
    """Strength of evidence carried by a mechanism statement."""

    ASSOCIATION = "observational_association"
    MULTI_ENVIRONMENT = "multi_environment_association"
    SIMULATION = "simulation_supported_association"
    TRIANGULATED = "simulation_experiment_triangulation"
    INTERVENTION = "intervention_supported"


@dataclass(slots=True)
class Observation:
    """One polymer, condition, simulation, or intervention result.

    ``features`` should describe structure or composition. ``mediators`` are
    quantities on a proposed mechanism path, such as a torsional barrier or
    dipole-correlation length. Outcomes belong in ``targets``. Keeping these
    namespaces separate prevents a target from accidentally leaking into the
    explanatory feature matrix.
    """

    sample_id: str
    features: dict[str, float] = field(default_factory=dict)
    targets: dict[str, float] = field(default_factory=dict)
    mediators: dict[str, float] = field(default_factory=dict)
    conditions: dict[str, float] = field(default_factory=dict)
    uncertainty: dict[str, float] = field(default_factory=dict)
    environment: str = "unknown"
    fidelity: str = "unknown"
    intervention: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sample_id = str(self.sample_id)
        self.environment = str(self.environment or "unknown")
        self.fidelity = str(self.fidelity or "unknown")
        for name in (
            "features",
            "targets",
            "mediators",
            "conditions",
            "uncertainty",
            "intervention",
        ):
            values = getattr(self, name)
            setattr(self, name, {str(key): float(value) for key, value in values.items()})
        self.provenance = dict(self.provenance)

    @property
    def explanatory_values(self) -> dict[str, float]:
        return {**self.features, **self.mediators, **self.conditions}

    def get(self, name: str, default: float | None = None) -> float | None:
        for mapping in (self.features, self.mediators, self.conditions, self.targets):
            if name in mapping:
                return mapping[name]
        return default

    def was_intervened(self, name: str) -> bool:
        return name in self.intervention


@dataclass(frozen=True, slots=True)
class FeatureEffect:
    term: str
    coefficient: float
    ci_low: float
    ci_high: float
    selection_frequency: float
    sign_consistency: float
    environment_sign_fraction: float
    support: int
    bootstrap_p_value: float
    is_interaction: bool = False
    is_nonlinear: bool = False

    @property
    def direction(self) -> str:
        if self.ci_low > 0:
            return "positive"
        if self.ci_high < 0:
            return "negative"
        return "uncertain"


@dataclass(frozen=True, slots=True)
class MediationResult:
    exposure: str
    mediator: str
    outcome: str
    exposure_to_mediator: float
    mediator_to_outcome: float
    direct_effect: float
    indirect_effect: float
    indirect_ci_low: float
    indirect_ci_high: float
    mediated_fraction: float | None
    sample_count: int
    evidence_level: EvidenceLevel
    caveat: str


@dataclass(frozen=True, slots=True)
class MatchedPairEffect:
    exposure: str
    outcome: str
    threshold: float
    matched_pairs: int
    average_difference: float
    ci_low: float
    ci_high: float
    median_control_distance: float
    evidence_level: EvidenceLevel
    caveat: str


@dataclass(frozen=True, slots=True)
class DiscoveredLaw:
    """Sparse empirical law expressed in standardized variables."""

    target: str
    expression: str
    intercept: float
    terms: tuple[str, ...]
    coefficients: tuple[float, ...]
    coefficient_ci: tuple[tuple[float, float], ...]
    train_r2: float
    validation_r2: float
    bic: float
    sample_count: int
    environments: tuple[str, ...]
    normalized: bool = True
    caveat: str = (
        "This is a compact empirical relation in standardized variables; "
        "it is not a dimensionally exact physical law until independently validated."
    )


@dataclass(frozen=True, slots=True)
class MechanismHypothesis:
    hypothesis_id: str
    statement: str
    target: str
    drivers: tuple[str, ...]
    mediators: tuple[str, ...]
    evidence_level: EvidenceLevel
    confidence: float
    supporting_effects: tuple[str, ...]
    falsification_tests: tuple[str, ...]
    citations: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()


@dataclass(slots=True)
class DiscoveryReport:
    target: str
    sample_count: int
    feature_names: tuple[str, ...]
    effects: tuple[FeatureEffect, ...]
    hypotheses: tuple[MechanismHypothesis, ...]
    mediations: tuple[MediationResult, ...] = ()
    matched_effects: tuple[MatchedPairEffect, ...] = ()
    law: DiscoveredLaw | None = None
    environments: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    schema_version: str = "1.0"
    physics_learning: PhysicsLearningReport | None = None

    def robust_effects(
        self,
        *,
        selection_frequency: float = 0.70,
        sign_consistency: float = 0.80,
    ) -> tuple[FeatureEffect, ...]:
        return tuple(
            effect
            for effect in self.effects
            if effect.selection_frequency >= selection_frequency
            and effect.sign_consistency >= sign_consistency
            and effect.direction != "uncertain"
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class ActiveCandidate:
    candidate_id: str
    features: dict[str, float]
    psmiles: str | None = None
    predictions: dict[str, float] = field(default_factory=dict)
    uncertainty: dict[str, float] = field(default_factory=dict)
    estimated_cost: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.estimated_cost <= 0:
            raise ValueError("estimated_cost must be positive")


@dataclass(frozen=True, slots=True)
class SimulationRequest:
    candidate_id: str
    oracle: str
    observables: tuple[str, ...]
    acquisition_score: float
    rationale: str
    prerequisites: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _jsonable(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


__all__ = [
    "ActiveCandidate",
    "DiscoveredLaw",
    "DiscoveryReport",
    "EvidenceLevel",
    "FeatureEffect",
    "MechanismHypothesis",
    "MatchedPairEffect",
    "MediationResult",
    "Observation",
    "SimulationRequest",
]
