from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class BackendStatus:
    """Availability and execution state of one physics-learning backend."""

    name: str
    available: bool
    executed: bool
    detail: str = ""
    version: str | None = None


@dataclass(frozen=True, slots=True)
class InteractionEdge:
    """A mixed-second-derivative interaction retained by the oracle."""

    left: str
    right: str
    score: float
    signed_score: float


@dataclass(frozen=True, slots=True)
class InteractionDecomposition:
    """PhyE2E-style divide-and-conquer evidence from an oracle Hessian."""

    feature_names: tuple[str, ...]
    score_matrix: tuple[tuple[float, ...], ...]
    signed_matrix: tuple[tuple[float, ...], ...]
    edges: tuple[InteractionEdge, ...]
    components: tuple[tuple[str, ...], ...]
    oracle: str
    oracle_validation_r2: float
    threshold: float
    sample_count: int
    caveat: str = (
        "Mixed second derivatives indicate non-additive predictive interaction. "
        "They do not alone establish a microscopic causal mechanism."
    )


@dataclass(frozen=True, slots=True)
class DimensionlessGroup:
    """One integer-exponent Buckingham-Pi group."""

    expression: str
    variables: tuple[str, ...]
    exponents: tuple[int, ...]
    residual: float


@dataclass(frozen=True, slots=True)
class PhysicsEquation:
    """A validated equation candidate returned by one symbolic backend."""

    equation_id: str
    target: str
    expression: str
    backend: str
    feature_names: tuple[str, ...]
    train_r2: float
    validation_r2: float
    train_rmse: float
    validation_rmse: float
    complexity: int
    unit_consistent: bool | None
    normalized: bool
    stability: float
    environment_consistency: float
    ranking_score: float = float("-inf")
    metadata: Mapping[str, Any] = field(default_factory=dict)
    caveats: tuple[str, ...] = ()


@dataclass(slots=True)
class PhysicsLearningReport:
    """Neural-symbolic physical-law discovery result."""

    target: str
    sample_count: int
    feature_names: tuple[str, ...]
    feature_units: dict[str, str]
    target_unit: str
    equations: tuple[PhysicsEquation, ...]
    best_equation_id: str | None
    interaction_decomposition: InteractionDecomposition | None = None
    dimensionless_groups: tuple[DimensionlessGroup, ...] = ()
    backend_status: tuple[BackendStatus, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    schema_version: str = "1.0"

    @property
    def best_equation(self) -> PhysicsEquation | None:
        if self.best_equation_id is None:
            return None
        return next(
            (
                equation
                for equation in self.equations
                if equation.equation_id == self.best_equation_id
            ),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class DynamicsEquation:
    """One discovered state-evolution equation."""

    state: str
    expression: str
    coefficients: tuple[float, ...]
    terms: tuple[str, ...]
    train_r2: float
    complexity: int


@dataclass(slots=True)
class DynamicsDiscoveryReport:
    """Sparse ODE/PDE surrogate for polymer relaxation or transport."""

    time_name: str
    state_names: tuple[str, ...]
    equations: tuple[DynamicsEquation, ...]
    backend: str
    weak_form: bool
    sample_count: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def physics_learning_report_from_dict(
    payload: Mapping[str, Any],
) -> PhysicsLearningReport:
    interaction_payload = payload.get("interaction_decomposition")
    interaction = None
    if interaction_payload is not None:
        interaction = InteractionDecomposition(
            feature_names=tuple(interaction_payload["feature_names"]),
            score_matrix=tuple(
                tuple(float(value) for value in row)
                for row in interaction_payload["score_matrix"]
            ),
            signed_matrix=tuple(
                tuple(float(value) for value in row)
                for row in interaction_payload["signed_matrix"]
            ),
            edges=tuple(
                InteractionEdge(
                    left=str(item["left"]),
                    right=str(item["right"]),
                    score=float(item["score"]),
                    signed_score=float(item["signed_score"]),
                )
                for item in interaction_payload.get("edges", ())
            ),
            components=tuple(
                tuple(component)
                for component in interaction_payload.get("components", ())
            ),
            oracle=str(interaction_payload["oracle"]),
            oracle_validation_r2=float(
                interaction_payload["oracle_validation_r2"]
            ),
            threshold=float(interaction_payload["threshold"]),
            sample_count=int(interaction_payload["sample_count"]),
            caveat=str(
                interaction_payload.get(
                    "caveat",
                    InteractionDecomposition.__dataclass_fields__["caveat"].default,
                )
            ),
        )
    return PhysicsLearningReport(
        target=str(payload["target"]),
        sample_count=int(payload["sample_count"]),
        feature_names=tuple(payload["feature_names"]),
        feature_units={
            str(name): str(unit)
            for name, unit in payload.get("feature_units", {}).items()
        },
        target_unit=str(payload.get("target_unit", "1")),
        equations=tuple(
            PhysicsEquation(
                equation_id=str(item["equation_id"]),
                target=str(item["target"]),
                expression=str(item["expression"]),
                backend=str(item["backend"]),
                feature_names=tuple(item.get("feature_names", ())),
                train_r2=float(item["train_r2"]),
                validation_r2=float(item["validation_r2"]),
                train_rmse=float(item["train_rmse"]),
                validation_rmse=float(item["validation_rmse"]),
                complexity=int(item["complexity"]),
                unit_consistent=item.get("unit_consistent"),
                normalized=bool(item.get("normalized", False)),
                stability=float(item.get("stability", 0.0)),
                environment_consistency=float(
                    item.get("environment_consistency", 0.0)
                ),
                ranking_score=float(item.get("ranking_score", float("-inf"))),
                metadata=dict(item.get("metadata", {})),
                caveats=tuple(item.get("caveats", ())),
            )
            for item in payload.get("equations", ())
        ),
        best_equation_id=(
            None
            if payload.get("best_equation_id") is None
            else str(payload["best_equation_id"])
        ),
        interaction_decomposition=interaction,
        dimensionless_groups=tuple(
            DimensionlessGroup(
                expression=str(item["expression"]),
                variables=tuple(item["variables"]),
                exponents=tuple(int(value) for value in item["exponents"]),
                residual=float(item["residual"]),
            )
            for item in payload.get("dimensionless_groups", ())
        ),
        backend_status=tuple(
            BackendStatus(
                name=str(item["name"]),
                available=bool(item["available"]),
                executed=bool(item["executed"]),
                detail=str(item.get("detail", "")),
                version=(
                    None if item.get("version") is None else str(item["version"])
                ),
            )
            for item in payload.get("backend_status", ())
        ),
        diagnostics=dict(payload.get("diagnostics", {})),
        warnings=tuple(payload.get("warnings", ())),
        schema_version=str(payload.get("schema_version", "1.0")),
    )


def dynamics_discovery_report_from_dict(
    payload: Mapping[str, Any],
) -> DynamicsDiscoveryReport:
    return DynamicsDiscoveryReport(
        time_name=str(payload["time_name"]),
        state_names=tuple(payload["state_names"]),
        equations=tuple(
            DynamicsEquation(
                state=str(item["state"]),
                expression=str(item["expression"]),
                coefficients=tuple(
                    float(value) for value in item.get("coefficients", ())
                ),
                terms=tuple(item.get("terms", ())),
                train_r2=float(item["train_r2"]),
                complexity=int(item["complexity"]),
            )
            for item in payload.get("equations", ())
        ),
        backend=str(payload["backend"]),
        weak_form=bool(payload.get("weak_form", False)),
        sample_count=int(payload["sample_count"]),
        diagnostics=dict(payload.get("diagnostics", {})),
        warnings=tuple(payload.get("warnings", ())),
        schema_version=str(payload.get("schema_version", "1.0")),
    )


def _jsonable(value: Any) -> Any:
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
    "BackendStatus",
    "DimensionlessGroup",
    "DynamicsDiscoveryReport",
    "DynamicsEquation",
    "InteractionDecomposition",
    "InteractionEdge",
    "PhysicsEquation",
    "PhysicsLearningReport",
    "dynamics_discovery_report_from_dict",
    "physics_learning_report_from_dict",
]
