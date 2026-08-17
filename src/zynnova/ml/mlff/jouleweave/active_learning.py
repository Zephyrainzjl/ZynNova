"""Uncertainty- and diversity-driven active learning for JouleWeave."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .dft_oracle import DFTOracle, DFTReferenceResult
from .uncertainty import JouleWeaveCommittee


class StructureSampler(Protocol):
    def sample(
        self,
        seeds: Sequence[Any],
        *,
        model: Any,
        cycle: int,
    ) -> Iterable[Any]: ...


Descriptor = Callable[[Any], np.ndarray]
RetrainCallback = Callable[[Sequence[tuple[Any, DFTReferenceResult]], int], Sequence[Any]]
ConditionResolver = Callable[[Any], Mapping[str, float]]


@dataclass(slots=True)
class ActiveLearningConfig:
    cycles: int = 8
    candidates_per_cycle: int = 256
    queries_per_cycle: int = 32
    minimum_uncertainty: float = 0.02
    force_uncertainty_weight: float = 1.0
    energy_uncertainty_weight: float = 0.25
    diversity_weight: float = 0.35
    maximum_failed_dft_fraction: float = 0.25
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.cycles < 1 or self.candidates_per_cycle < 1 or self.queries_per_cycle < 1:
            raise ValueError("active-learning counts must be positive")
        if self.queries_per_cycle > self.candidates_per_cycle:
            raise ValueError("queries_per_cycle cannot exceed candidates_per_cycle")
        if self.minimum_uncertainty < 0.0:
            raise ValueError("minimum_uncertainty cannot be negative")
        if min(
            self.force_uncertainty_weight,
            self.energy_uncertainty_weight,
            self.diversity_weight,
        ) < 0.0:
            raise ValueError("active-learning weights cannot be negative")
        if not 0.0 <= self.maximum_failed_dft_fraction < 1.0:
            raise ValueError("maximum_failed_dft_fraction must lie in [0,1)")


@dataclass(frozen=True, slots=True)
class QueryCandidate:
    structure: Any
    energy_uncertainty: float
    force_uncertainty: float
    acquisition_score: float
    descriptor: np.ndarray
    conditions: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ActiveLearningCycleRecord:
    cycle: int
    sampled: int
    above_threshold: int
    queried: int
    converged_dft: int
    failed_dft: int
    maximum_score: float
    mean_score: float


@dataclass(slots=True)
class ActiveLearningResult:
    models: tuple[Any, ...]
    labeled_data: list[tuple[Any, DFTReferenceResult]]
    records: list[ActiveLearningCycleRecord]


class ActiveLearningCampaign:
    """Closed-loop MD/configuration sampling, DFT labeling, and retraining."""

    def __init__(
        self,
        models: Sequence[Any],
        sampler: StructureSampler,
        oracle: DFTOracle,
        retrain: RetrainCallback,
        descriptor: Descriptor,
        *,
        condition_resolver: ConditionResolver | None = None,
        config: ActiveLearningConfig | None = None,
    ) -> None:
        self.models = tuple(models)
        self.sampler = sampler
        self.oracle = oracle
        self.retrain = retrain
        self.descriptor = descriptor
        self.condition_resolver = condition_resolver or (lambda _: {})
        self.config = config or ActiveLearningConfig()
        if len(self.models) < 2:
            raise ValueError("active learning requires at least two committee models")

    def run(self, seeds: Sequence[Any]) -> ActiveLearningResult:
        if not seeds:
            raise ValueError("active-learning seeds cannot be empty")
        rng = np.random.default_rng(self.config.random_seed)
        labeled: list[tuple[Any, DFTReferenceResult]] = []
        records: list[ActiveLearningCycleRecord] = []
        current_seeds = list(seeds)
        for cycle in range(self.config.cycles):
            committee = JouleWeaveCommittee(self.models)
            sampled = list(
                self.sampler.sample(current_seeds, model=self.models[0], cycle=cycle)
            )[: self.config.candidates_per_cycle]
            candidates = [
                self._candidate(structure, committee) for structure in sampled
            ]
            eligible = [
                candidate
                for candidate in candidates
                if max(candidate.energy_uncertainty, candidate.force_uncertainty)
                >= self.config.minimum_uncertainty
            ]
            selected = _diverse_select(
                eligible,
                self.config.queries_per_cycle,
                diversity_weight=self.config.diversity_weight,
                rng=rng,
            )
            cycle_labels: list[tuple[Any, DFTReferenceResult]] = []
            failures = 0
            for candidate in selected:
                try:
                    result = self.oracle.evaluate(
                        candidate.structure,
                        electrode_potential_V=candidate.conditions.get(
                            "electrode_potential_V"
                        ),
                        electron_count=candidate.conditions.get("electron_count"),
                    )
                except Exception:
                    failures += 1
                    continue
                if not result.converged:
                    failures += 1
                    continue
                cycle_labels.append((candidate.structure, result))
            if selected and failures / len(selected) > self.config.maximum_failed_dft_fraction:
                raise RuntimeError(
                    "DFT failure fraction exceeded the active-learning safety limit"
                )
            labeled.extend(cycle_labels)
            if cycle_labels:
                updated = tuple(self.retrain(cycle_labels, cycle))
                if len(updated) < 2:
                    raise ValueError("retrain callback must return at least two models")
                self.models = updated
                current_seeds = [structure for structure, _ in cycle_labels]
            scores = np.asarray([candidate.acquisition_score for candidate in eligible])
            records.append(
                ActiveLearningCycleRecord(
                    cycle=cycle,
                    sampled=len(sampled),
                    above_threshold=len(eligible),
                    queried=len(selected),
                    converged_dft=len(cycle_labels),
                    failed_dft=failures,
                    maximum_score=float(np.max(scores)) if scores.size else 0.0,
                    mean_score=float(np.mean(scores)) if scores.size else 0.0,
                )
            )
            if not eligible:
                break
        return ActiveLearningResult(self.models, labeled, records)

    def _candidate(
        self,
        structure: Any,
        committee: JouleWeaveCommittee,
    ) -> QueryCandidate:
        inputs = _structure_inputs(structure)
        prediction = committee.predict(inputs, compute_forces=True)
        energy = float(
            np.max(prediction.energy_standard_uncertainty_eV.detach().cpu().numpy())
        )
        assert prediction.maximum_atomic_force_uncertainty_eV_A is not None
        force = float(
            np.max(
                prediction.maximum_atomic_force_uncertainty_eV_A.detach().cpu().numpy()
            )
        )
        descriptor = np.asarray(self.descriptor(structure), dtype=float).reshape(-1)
        if descriptor.size == 0 or not np.isfinite(descriptor).all():
            raise ValueError("structure descriptor must be finite and non-empty")
        score = (
            self.config.energy_uncertainty_weight * energy
            + self.config.force_uncertainty_weight * force
        )
        return QueryCandidate(
            structure=structure,
            energy_uncertainty=energy,
            force_uncertainty=force,
            acquisition_score=float(score),
            descriptor=descriptor,
            conditions=dict(self.condition_resolver(structure)),
        )


def _structure_inputs(structure: Any) -> Mapping[str, Any]:
    if isinstance(structure, Mapping):
        return structure
    if hasattr(structure, "model_inputs"):
        return structure.model_inputs()
    raise TypeError(
        "active-learning structures must be model-input mappings or expose model_inputs()"
    )


def _diverse_select(
    candidates: Sequence[QueryCandidate],
    count: int,
    *,
    diversity_weight: float,
    rng: np.random.Generator,
) -> list[QueryCandidate]:
    if not candidates or count <= 0:
        return []
    count = min(count, len(candidates))
    descriptors = np.stack([candidate.descriptor for candidate in candidates])
    mean = np.mean(descriptors, axis=0)
    scale = np.std(descriptors, axis=0)
    scale[scale <= np.finfo(float).eps] = 1.0
    descriptors = (descriptors - mean) / scale
    scores = np.asarray([candidate.acquisition_score for candidate in candidates])
    score_scale = np.ptp(scores)
    normalized_scores = (
        np.ones_like(scores) if score_scale <= np.finfo(float).eps else (scores - np.min(scores)) / score_scale
    )
    selected = [int(np.argmax(normalized_scores + rng.normal(0.0, 1.0e-12, len(scores))))]
    remaining = set(range(len(candidates))) - set(selected)
    while remaining and len(selected) < count:
        indices = np.asarray(sorted(remaining), dtype=int)
        distances = np.min(
            np.linalg.norm(
                descriptors[indices, None, :] - descriptors[np.asarray(selected), :][None, :, :],
                axis=-1,
            ),
            axis=1,
        )
        distance_scale = np.max(distances)
        normalized_distance = distances / max(distance_scale, np.finfo(float).eps)
        combined = normalized_scores[indices] + diversity_weight * normalized_distance
        chosen = int(indices[int(np.argmax(combined))])
        selected.append(chosen)
        remaining.remove(chosen)
    return [candidates[index] for index in selected]


__all__ = [
    "ActiveLearningCampaign",
    "ActiveLearningConfig",
    "ActiveLearningCycleRecord",
    "ActiveLearningResult",
    "QueryCandidate",
    "StructureSampler",
]
