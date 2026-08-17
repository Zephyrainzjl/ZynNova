from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import ActiveLearningConfig
from .schema import (
    ActiveCandidate,
    DiscoveryReport,
    Observation,
    SimulationRequest,
)


@dataclass(frozen=True, slots=True)
class AcquisitionBreakdown:
    candidate_id: str
    total: float
    uncertainty: float
    novelty: float
    information: float
    diversity: float
    cost_penalty: float


class ActiveDiscoveryLoop:
    """Select calculations that are informative for mechanisms, not only prediction."""

    def __init__(self, config: ActiveLearningConfig | None = None) -> None:
        self.config = config or ActiveLearningConfig()
        self.config.__post_init__()

    def propose(
        self,
        candidates: Sequence[ActiveCandidate],
        observations: Sequence[Observation],
        report: DiscoveryReport,
    ) -> tuple[SimulationRequest, ...]:
        requests, _breakdown = self.propose_with_scores(candidates, observations, report)
        return requests

    def propose_with_scores(
        self,
        candidates: Sequence[ActiveCandidate],
        observations: Sequence[Observation],
        report: DiscoveryReport,
    ) -> tuple[tuple[SimulationRequest, ...], tuple[AcquisitionBreakdown, ...]]:
        if not candidates:
            return (), ()
        names = report.feature_names
        center = np.asarray(
            [report.diagnostics.get("feature_center", {}).get(name, 0.0) for name in names],
            dtype=float,
        )
        scale = np.asarray(
            [report.diagnostics.get("feature_scale", {}).get(name, 1.0) for name in names],
            dtype=float,
        )
        scale[scale < 1.0e-12] = 1.0
        candidate_matrix = np.asarray(
            [
                [
                    candidate.features.get(name, center[index])
                    for index, name in enumerate(names)
                ]
                for candidate in candidates
            ],
            dtype=float,
        )
        candidate_z = (candidate_matrix - center) / scale
        training_matrix = np.asarray(
            [
                [
                    observation.explanatory_values.get(name, center[index])
                    for index, name in enumerate(names)
                ]
                for observation in observations
            ],
            dtype=float,
        )
        training_z = (training_matrix - center) / scale if len(observations) else np.empty(
            (0, len(names))
        )
        novelty = np.asarray(
            [
                (
                    float(np.min(np.linalg.norm(training_z - row, axis=1)))
                    if len(training_z)
                    else 1.0
                )
                for row in candidate_z
            ],
            dtype=float,
        )
        uncertainty = np.asarray(
            [candidate.uncertainty.get(report.target, 0.0) for candidate in candidates],
            dtype=float,
        )
        information = np.asarray(
            [
                self._information_score(row, names, report)
                for row in candidate_z
            ],
            dtype=float,
        )
        cost = np.log1p(
            np.asarray([candidate.estimated_cost for candidate in candidates], dtype=float)
        )
        uncertainty = _unit_scale(uncertainty)
        novelty = _unit_scale(novelty)
        information = _unit_scale(information)
        cost = _unit_scale(cost)
        base = (
            self.config.uncertainty_weight * uncertainty
            + self.config.novelty_weight * novelty
            + self.config.information_weight * information
            - self.config.cost_weight * cost
        )

        remaining = list(range(len(candidates)))
        selected: list[int] = []
        breakdown: list[AcquisitionBreakdown] = []
        while remaining and len(selected) < min(self.config.batch_size, len(candidates)):
            scored = []
            for index in remaining:
                diversity_distance = (
                    1.0
                    if not selected
                    else float(
                        min(
                            np.linalg.norm(candidate_z[index] - candidate_z[chosen])
                            for chosen in selected
                        )
                    )
                )
                diversity = (
                    1.0
                    if not selected
                    else min(diversity_distance / 3.0, 1.0)
                )
                total = base[index] + self.config.diversity_weight * diversity
                scored.append(
                    (float(total), index, diversity, diversity_distance)
                )
            total, best, diversity, diversity_distance = max(
                scored,
                key=lambda item: (item[0], -item[1]),
            )
            if selected and diversity_distance < self.config.minimum_distance:
                remaining.remove(best)
                continue
            selected.append(best)
            remaining.remove(best)
            breakdown.append(
                AcquisitionBreakdown(
                    candidate_id=candidates[best].candidate_id,
                    total=total,
                    uncertainty=float(uncertainty[best]),
                    novelty=float(novelty[best]),
                    information=float(information[best]),
                    diversity=float(diversity),
                    cost_penalty=float(cost[best]),
                )
            )

        requests = tuple(
            _simulation_request(
                candidates[index],
                report,
                acquisition_score=breakdown[position].total,
            )
            for position, index in enumerate(selected)
        )
        return requests, tuple(breakdown)

    @staticmethod
    def _information_score(
        feature_z: np.ndarray,
        feature_names: Sequence[str],
        report: DiscoveryReport,
    ) -> float:
        lookup = dict(zip(feature_names, feature_z, strict=True))
        score = 0.0
        term_center = report.diagnostics.get("candidate_term_center", {})
        term_scale = report.diagnostics.get("candidate_term_scale", {})
        for effect in report.effects[:30]:
            term_value = _term_value(effect.term, lookup)
            center = float(term_center.get(effect.term, 0.0))
            scale = max(abs(float(term_scale.get(effect.term, 1.0))), 1.0e-12)
            term_value = (term_value - center) / scale
            interval_width = max(effect.ci_high - effect.ci_low, 0.0)
            unresolved = 1.0 - min(effect.selection_frequency, effect.sign_consistency)
            score += abs(term_value) * (interval_width + unresolved)
        physics = report.physics_learning
        interaction = (
            None if physics is None else physics.interaction_decomposition
        )
        if interaction is not None and interaction.edges:
            maximum = max(edge.score for edge in interaction.edges)
            maximum = max(float(maximum), 1.0e-12)
            for edge in interaction.edges[:20]:
                left = lookup.get(edge.left, 0.0)
                right = lookup.get(edge.right, 0.0)
                score += (
                    abs(float(left) * float(right))
                    * float(edge.score)
                    / maximum
                )
        return float(score)


def propose_counterfactual_pairs(
    candidates: Sequence[ActiveCandidate],
    *,
    exposure: str,
    control_names: Sequence[str],
    pair_count: int = 5,
) -> tuple[tuple[str, str, float], ...]:
    """Find pairs with a large exposure contrast and similar measured controls."""

    eligible = [
        candidate
        for candidate in candidates
        if exposure in candidate.features
        and all(name in candidate.features for name in control_names)
    ]
    if len(eligible) < 2 or pair_count < 1:
        return ()
    controls = np.asarray(
        [[candidate.features[name] for name in control_names] for candidate in eligible],
        dtype=float,
    )
    center = np.mean(controls, axis=0)
    scale = np.std(controls, axis=0)
    scale[scale < 1.0e-12] = 1.0
    controls = (controls - center) / scale
    exposures = np.asarray([candidate.features[exposure] for candidate in eligible])
    exposure_scale = max(float(np.std(exposures)), 1.0e-12)
    pairs = []
    for left in range(len(eligible)):
        for right in range(left + 1, len(eligible)):
            contrast = abs(float(exposures[left] - exposures[right])) / exposure_scale
            distance = float(np.linalg.norm(controls[left] - controls[right]))
            score = contrast / (1.0 + distance)
            pairs.append((score, eligible[left].candidate_id, eligible[right].candidate_id))
    pairs.sort(reverse=True)
    selected = []
    used: set[str] = set()
    for score, left, right in pairs:
        if left in used or right in used:
            continue
        selected.append((left, right, float(score)))
        used.update((left, right))
        if len(selected) == pair_count:
            break
    return tuple(selected)


def _term_value(term: str, values: dict[str, float]) -> float:
    components = [part.strip() for part in term.split(" × ")]
    result = 1.0
    for component in components:
        squared = component.endswith("^2")
        name = component.removesuffix("^2")
        value = float(values.get(name, 0.0))
        result *= value**2 if squared else value
    return result


def _unit_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    clean = np.where(finite, values, np.nanmedian(values[finite]))
    lower = float(np.min(clean))
    upper = float(np.max(clean))
    if upper - lower < 1.0e-12:
        return np.zeros_like(clean)
    return (clean - lower) / (upper - lower)


def _simulation_request(
    candidate: ActiveCandidate,
    report: DiscoveryReport,
    *,
    acquisition_score: float,
) -> SimulationRequest:
    target = report.target.lower()
    hypothesis_mediators = tuple(
        dict.fromkeys(
            mediator
            for hypothesis in report.hypotheses[:5]
            for mediator in hypothesis.mediators
        )
    )
    if any(
        token in target
        for token in ("bandgap", "band_gap", "homo", "lumo", "electron_affinity")
    ):
        oracle = "dft_electronic"
        observables = ("homo_eV", "lumo_eV", "bandgap_eV", "electron_affinity_eV")
        prerequisites = ("3D periodic or oligomer structure", "fixed electronic method")
    elif any(token in target for token in ("barrier", "activation", "torsion")):
        oracle = "jouleweave_neb"
        observables = ("forward_barrier_eV", "reverse_barrier_eV", "reaction_energy_eV")
        prerequisites = ("validated potential applicability", "mapped endpoint atom order")
    elif any(token in target for token in ("dielectric", "loss", "relaxation")):
        oracle = "dipole_md"
        observables = (
            "dielectric_constant",
            "dipole_autocorrelation",
            "segmental_relaxation_time_s",
        )
        prerequisites = ("equilibrated periodic cell", "charge/dipole-valid potential")
    elif any(token in target for token in ("d33", "piezo")):
        oracle = "neb_plus_piezoelectric_validation"
        observables = (
            "torsional_barrier_eV",
            "helix_trans_energy_delta_eV",
            "piezoelectric_d33_pC_N",
        )
        prerequisites = ("crosslinked atomistic endpoints", "phase-matched control")
    else:
        oracle = "multiscale_energy_storage_validation"
        observables = (
            "barrier_distribution_eV",
            "dipole_fluctuation",
            "P_E_loop",
            report.target,
        )
        prerequisites = (
            "validated potential",
            "condition-matched P-E or phase-field protocol",
        )
    if hypothesis_mediators:
        observables = tuple(dict.fromkeys((*observables, *hypothesis_mediators)))
    return SimulationRequest(
        candidate_id=candidate.candidate_id,
        oracle=oracle,
        observables=observables,
        acquisition_score=float(acquisition_score),
        rationale=(
            "High expected information for the current mechanism coefficients, "
            "combined with model uncertainty, novelty, diversity, and evaluation cost."
        ),
        prerequisites=prerequisites,
        metadata={"psmiles": candidate.psmiles, **candidate.metadata},
    )


__all__ = [
    "AcquisitionBreakdown",
    "ActiveDiscoveryLoop",
    "propose_counterfactual_pairs",
]
