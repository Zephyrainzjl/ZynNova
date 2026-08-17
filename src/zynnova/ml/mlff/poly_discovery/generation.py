from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .config import MechanismConstraint, MechanismGenerationConfig
from .features import FunctionalGroupFeaturizer
from .schema import ActiveCandidate, DiscoveryReport, Observation


@dataclass(slots=True)
class MechanismGuidedCandidate:
    """A PolyLoom proposal reranked inside a discovered mechanism domain."""

    psmiles: str
    score: float
    generator_score: float
    mechanism_score: float
    applicability_distance: float
    feature_coverage: float
    novelty_score: float
    uncertainty_penalty: float
    predicted_properties: dict[str, float]
    property_uncertainty: dict[str, float]
    features: dict[str, float]
    mechanism_contributions: dict[str, float]
    warnings: tuple[str, ...] = ()
    source_candidate: Any = field(default=None, repr=False, compare=False)

    def to_active_candidate(
        self,
        *,
        candidate_id: str | None = None,
        estimated_cost: float = 1.0,
    ) -> ActiveCandidate:
        return ActiveCandidate(
            candidate_id=candidate_id or self.psmiles,
            psmiles=self.psmiles,
            features=dict(self.features),
            predictions=dict(self.predicted_properties),
            uncertainty=dict(self.property_uncertainty),
            estimated_cost=estimated_cost,
            metadata={
                "mechanism_guided_score": self.score,
                "applicability_distance": self.applicability_distance,
                "mechanism_contributions": dict(self.mechanism_contributions),
            },
        )


@dataclass(slots=True)
class MechanismGuidedGenerationResult:
    candidates: tuple[MechanismGuidedCandidate, ...]
    target: str
    requested_properties: dict[str, float]
    attempted: int
    chemically_valid: int
    rejection_counts: dict[str, int]
    raw_result: Any = field(default=None, repr=False, compare=False)


def generate_mechanism_guided_polymers(
    generator: Any,
    report: DiscoveryReport,
    requested_properties: Mapping[str, float],
    observations: Sequence[Observation],
    *,
    process_conditions: Mapping[str, float] | None = None,
    predictor: Any | None = None,
    property_constraints: Sequence[Any] = (),
    config: MechanismGenerationConfig | None = None,
    polyloom_config: Any | None = None,
    featurizer: FunctionalGroupFeaturizer | None = None,
) -> MechanismGuidedGenerationResult:
    """Generate with PolyLoom, then rerank by stable discovered mechanisms.

    PolyLoom remains responsible for chemical syntax, endpoint validity, property
    conditioning, and independent PolyPrism screening. This layer adds only
    evidence-backed mechanism terms, applicability-domain checks, and uncertainty.
    """

    from ...generation.PolyLoom import PolyLoomSamplingConfig, generate_poly_loom

    resolved = config or MechanismGenerationConfig()
    resolved.__post_init__()
    base = polyloom_config or PolyLoomSamplingConfig()
    raw_count = resolved.num_candidates * resolved.oversample_factor
    sampling = replace(
        base,
        num_candidates=raw_count,
        oversample_factor=1,
        seed=resolved.seed,
    )
    raw = generate_poly_loom(
        generator,
        requested_properties,
        process_conditions=process_conditions,
        config=sampling,
        predictor=predictor,
        constraints=property_constraints,
    )
    ranked, mechanism_rejections = rank_mechanism_candidates(
        raw.candidates,
        report,
        observations,
        config=resolved,
        process_conditions=process_conditions,
        featurizer=featurizer,
    )
    rejection_counts = dict(raw.rejection_counts)
    for reason, count in mechanism_rejections.items():
        rejection_counts[reason] = rejection_counts.get(reason, 0) + count
    return MechanismGuidedGenerationResult(
        candidates=ranked[: resolved.num_candidates],
        target=report.target,
        requested_properties={
            str(name): float(value) for name, value in requested_properties.items()
        },
        attempted=int(raw.attempted),
        chemically_valid=int(raw.chemically_valid),
        rejection_counts=rejection_counts,
        raw_result=raw,
    )


def rank_mechanism_candidates(
    candidates: Sequence[Any],
    report: DiscoveryReport,
    observations: Sequence[Observation],
    *,
    config: MechanismGenerationConfig | None = None,
    process_conditions: Mapping[str, float] | None = None,
    featurizer: FunctionalGroupFeaturizer | None = None,
) -> tuple[tuple[MechanismGuidedCandidate, ...], dict[str, int]]:
    """Rerank PolyLoom-like candidate objects without invoking a generator."""

    resolved = config or MechanismGenerationConfig()
    resolved.__post_init__()
    featurizer = featurizer or FunctionalGroupFeaturizer()
    conditions = {
        str(name): float(value) for name, value in (process_conditions or {}).items()
    }
    center, scale = _feature_scaling(report)
    training = _training_matrix(observations, report.feature_names, center, scale)
    known_smiles = tuple(
        str(observation.provenance["psmiles"])
        for observation in observations
        if observation.provenance.get("psmiles")
    )
    robust_effects = report.robust_effects()
    staged: list[dict[str, Any]] = []
    rejections: dict[str, int] = {}

    for candidate in candidates:
        psmiles = str(getattr(candidate, "psmiles", "")).strip()
        if not psmiles:
            _reject(rejections, "candidate has no pSMILES")
            continue
        vector = featurizer.transform(psmiles)
        physics = _numeric_mapping(getattr(candidate, "physics_descriptors", {}))
        explicit_features = _numeric_mapping(getattr(candidate, "features", {}))
        predicted = _numeric_mapping(
            getattr(
                candidate,
                "predicted_properties",
                getattr(candidate, "predictions", {}),
            )
        )
        uncertainty = _numeric_mapping(
            getattr(
                candidate,
                "property_uncertainty",
                getattr(candidate, "uncertainty", {}),
            )
        )
        features = {
            **vector.values,
            **physics,
            **explicit_features,
            **conditions,
        }
        constraint_penalty, reason = _constraint_penalty(
            features,
            predicted,
            resolved.constraints,
        )
        if reason is not None:
            _reject(rejections, reason)
            continue

        row, present = _candidate_row(features, report.feature_names, center)
        row_z = (row - center) / scale
        coverage = float(np.mean(present)) if present.size else 1.0
        distance = (
            float(np.min(np.linalg.norm(training - row_z, axis=1)))
            if len(training)
            else 0.0
        )
        if (
            resolved.maximum_applicability_distance is not None
            and distance > resolved.maximum_applicability_distance
        ):
            _reject(rejections, "outside mechanism applicability domain")
            continue

        z_lookup = dict(zip(report.feature_names, row_z, strict=True))
        term_center = report.diagnostics.get("candidate_term_center", {})
        term_scale = report.diagnostics.get("candidate_term_scale", {})
        contributions = {
            effect.term: float(
                effect.coefficient
                * _standardized_term_value(
                    effect.term,
                    z_lookup,
                    term_center,
                    term_scale,
                )
            )
            for effect in robust_effects
        }
        mechanism_raw = float(sum(contributions.values()))
        relative_uncertainty = sum(
            uncertainty.get(name, 0.0) / max(abs(value), 1.0e-8)
            for name, value in predicted.items()
        )
        novelty = _chemical_novelty(psmiles, known_smiles)
        if not known_smiles:
            novelty = min(distance / 3.0, 1.0)
        warnings = list(vector.warnings)
        if coverage < 0.8:
            warnings.append(
                f"Only {coverage:.0%} of discovery features were available; "
                "missing values were imputed at the training center."
            )
        staged.append(
            {
                "candidate": candidate,
                "psmiles": psmiles,
                "generator_raw": float(getattr(candidate, "score", 0.0)),
                "mechanism_raw": mechanism_raw,
                "distance": distance,
                "coverage": coverage,
                "novelty": novelty,
                "uncertainty_raw": relative_uncertainty,
                "constraint_penalty": constraint_penalty,
                "predicted": predicted,
                "uncertainty": uncertainty,
                "features": features,
                "contributions": contributions,
                "warnings": tuple(warnings),
            }
        )

    if not staged:
        return (), rejections
    generator_score = _unit_scale([item["generator_raw"] for item in staged])
    mechanism_score = _unit_scale([item["mechanism_raw"] for item in staged])
    applicability = _unit_scale([item["distance"] for item in staged])
    uncertainty_penalty = _unit_scale(
        [item["uncertainty_raw"] for item in staged]
    )
    novelty = np.asarray([item["novelty"] for item in staged], dtype=float)
    result = []
    for index, item in enumerate(staged):
        coverage_penalty = 1.0 - item["coverage"]
        score = (
            generator_score[index]
            + resolved.mechanism_weight * mechanism_score[index]
            - resolved.applicability_weight
            * (applicability[index] + coverage_penalty)
            - resolved.uncertainty_weight * uncertainty_penalty[index]
            + resolved.novelty_weight * novelty[index]
            - item["constraint_penalty"]
        )
        result.append(
            MechanismGuidedCandidate(
                psmiles=item["psmiles"],
                score=float(score),
                generator_score=float(generator_score[index]),
                mechanism_score=float(mechanism_score[index]),
                applicability_distance=float(item["distance"]),
                feature_coverage=float(item["coverage"]),
                novelty_score=float(novelty[index]),
                uncertainty_penalty=float(uncertainty_penalty[index]),
                predicted_properties=item["predicted"],
                property_uncertainty=item["uncertainty"],
                features=item["features"],
                mechanism_contributions=item["contributions"],
                warnings=item["warnings"],
                source_candidate=item["candidate"],
            )
        )
    result.sort(key=lambda item: (item.score, item.mechanism_score), reverse=True)
    return tuple(result), rejections


def _feature_scaling(report: DiscoveryReport) -> tuple[np.ndarray, np.ndarray]:
    feature_center = report.diagnostics.get("feature_center", {})
    feature_scale = report.diagnostics.get("feature_scale", {})
    center = np.asarray(
        [float(feature_center.get(name, 0.0)) for name in report.feature_names],
        dtype=float,
    )
    scale = np.asarray(
        [float(feature_scale.get(name, 1.0)) for name in report.feature_names],
        dtype=float,
    )
    scale[~np.isfinite(scale) | (np.abs(scale) < 1.0e-12)] = 1.0
    return center, scale


def _training_matrix(
    observations: Sequence[Observation],
    names: Sequence[str],
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    if not observations:
        return np.empty((0, len(names)), dtype=float)
    rows = []
    for observation in observations:
        values = observation.explanatory_values
        rows.append(
            [
                float(values.get(name, center[index]))
                for index, name in enumerate(names)
            ]
        )
    matrix = np.asarray(rows, dtype=float)
    matrix = np.where(np.isfinite(matrix), matrix, center[None, :])
    return (matrix - center) / scale


def _candidate_row(
    features: Mapping[str, float],
    names: Sequence[str],
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    present = []
    for index, name in enumerate(names):
        value = features.get(name)
        is_present = value is not None and np.isfinite(float(value))
        values.append(float(value) if is_present else float(center[index]))
        present.append(is_present)
    return np.asarray(values, dtype=float), np.asarray(present, dtype=bool)


def _constraint_penalty(
    features: Mapping[str, float],
    predictions: Mapping[str, float],
    constraints: Sequence[MechanismConstraint],
) -> tuple[float, str | None]:
    penalty = 0.0
    for constraint in constraints:
        value = features.get(constraint.feature, predictions.get(constraint.feature))
        if value is None or not np.isfinite(float(value)):
            if constraint.required:
                return penalty, f"missing mechanism constraint: {constraint.feature}"
            penalty += constraint.weight
            continue
        violation = 0.0
        if constraint.lower is not None and value < constraint.lower:
            violation = (constraint.lower - value) / max(abs(constraint.lower), 1.0)
        if constraint.upper is not None and value > constraint.upper:
            violation = max(
                violation,
                (value - constraint.upper) / max(abs(constraint.upper), 1.0),
            )
        if violation > 0 and constraint.required:
            return penalty, f"mechanism constraint failed: {constraint.feature}"
        penalty += constraint.weight * violation
    return float(penalty), None


def _term_value(term: str, values: Mapping[str, float]) -> float:
    result = 1.0
    for component in (part.strip() for part in term.split(" × ")):
        squared = component.endswith("^2")
        name = component.removesuffix("^2")
        value = float(values.get(name, 0.0))
        result *= value**2 if squared else value
    return float(result)


def _standardized_term_value(
    term: str,
    values: Mapping[str, float],
    centers: Mapping[str, Any],
    scales: Mapping[str, Any],
) -> float:
    raw = _term_value(term, values)
    center = float(centers.get(term, 0.0))
    scale = max(abs(float(scales.get(term, 1.0))), 1.0e-12)
    return float((raw - center) / scale)


def _chemical_novelty(psmiles: str, known_smiles: Sequence[str]) -> float:
    query = _tokens(psmiles)
    if not known_smiles:
        return 1.0
    similarities = []
    for known in known_smiles:
        reference = _tokens(known)
        union = query | reference
        similarities.append(len(query & reference) / len(union) if union else 1.0)
    return float(1.0 - max(similarities))


def _tokens(psmiles: str) -> set[str]:
    return set(re.findall(r"Cl|Br|\[[^\]]+\]|[A-Z][a-z]?|[cnosp]|[=#()]", psmiles))


def _numeric_mapping(values: Any) -> dict[str, float]:
    if not isinstance(values, Mapping):
        return {}
    result = {}
    for name, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            result[str(name)] = number
    return result


def _unit_scale(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array)
    clean = np.where(finite, array, np.nanmedian(array[finite]))
    lower = float(np.min(clean))
    upper = float(np.max(clean))
    if upper - lower < 1.0e-12:
        return np.zeros_like(clean)
    return (clean - lower) / (upper - lower)


def _reject(counts: dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


__all__ = [
    "MechanismGuidedCandidate",
    "MechanismGuidedGenerationResult",
    "generate_mechanism_guided_polymers",
    "rank_mechanism_candidates",
]
