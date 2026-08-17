"""Exact-composition correlated generation and descriptor-preserving refinement."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .metrics import analyze_microstructure, descriptor_loss
from .schema import DEFAULT_PHASE_NAMES, MicrostructureCondition
from .volume import MicrostructureVolume


@dataclass(frozen=True, slots=True)
class GenerationResult:
    volume: MicrostructureVolume
    backend: str
    exact_counts: Mapping[int, int]
    achieved_counts: Mapping[int, int]
    refinement_loss: float | None
    metadata: Mapping[str, object]


class SpectralConditionalGenerator:
    """Multi-field Gaussian spectral generator with exact discrete phase quotas.

    It is deliberately a strong, fully local baseline rather than a claim to
    reproduce a learned diffusion model. Independent anisotropic Gaussian fields
    provide morphology; bias balancing and quota repair enforce exact mass.
    """

    name = "spectral-exact"

    def generate(
        self,
        condition: MicrostructureCondition,
        *,
        refinement_steps: int = 0,
        temperature: float = 0.15,
    ) -> GenerationResult:
        rng = np.random.default_rng(condition.seed)
        phases = condition.phases
        fields = np.stack(
            [
                _correlated_field(
                    condition.shape,
                    condition.correlation_lengths_voxels.get(phase, (3.0, 3.0, 3.0)),
                    rng,
                )
                for phase in phases
            ],
            axis=0,
        )
        fields = _apply_interface_affinity(fields, phases, condition.interface_affinity)
        exact_counts = condition.exact_phase_counts()
        labels = _quota_assign(fields, phases, exact_counts, temperature=temperature)
        volume = MicrostructureVolume(
            labels=labels,
            voxel_size_m=condition.voxel_size_m,
            phase_names=DEFAULT_PHASE_NAMES,
            metadata={
                "generator": self.name,
                "seed": condition.seed,
                "correlation_lengths_voxels": condition.correlation_lengths_voxels,
            },
        )
        final_loss: float | None = None
        if refinement_steps > 0 and condition.descriptor_targets:
            volume, final_loss = refine_by_label_swaps(
                volume,
                condition.descriptor_targets,
                steps=refinement_steps,
                seed=condition.seed + 1,
            )
        achieved = {
            int(phase): int(np.count_nonzero(volume.labels == phase)) for phase in phases
        }
        return GenerationResult(
            volume=volume,
            backend=self.name,
            exact_counts=exact_counts,
            achieved_counts=achieved,
            refinement_loss=final_loss,
            metadata={"temperature": temperature, "refinement_steps": refinement_steps},
        )


def _correlated_field(
    shape: tuple[int, int, int],
    correlation_lengths: tuple[float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    white = rng.standard_normal(shape)
    frequencies = [np.fft.fftfreq(size) for size in shape]
    kz, ky, kx = np.meshgrid(*frequencies, indexing="ij")
    scaled_square = (
        (2.0 * np.pi * correlation_lengths[0] * kz) ** 2
        + (2.0 * np.pi * correlation_lengths[1] * ky) ** 2
        + (2.0 * np.pi * correlation_lengths[2] * kx) ** 2
    )
    kernel = np.exp(-0.5 * scaled_square)
    field = np.fft.ifftn(np.fft.fftn(white) * kernel).real
    field -= field.mean()
    standard_deviation = field.std()
    if standard_deviation > 0.0:
        field /= standard_deviation
    return field


def _apply_interface_affinity(
    fields: np.ndarray,
    phases: tuple[int, ...],
    affinity: Mapping[tuple[int, int], float],
) -> np.ndarray:
    if not affinity:
        return fields
    result = fields.copy()
    phase_index = {phase: index for index, phase in enumerate(phases)}
    for pair, strength in affinity.items():
        if pair[0] not in phase_index or pair[1] not in phase_index:
            continue
        left = phase_index[pair[0]]
        right = phase_index[pair[1]]
        mixed = 0.5 * (fields[left] + fields[right])
        value = float(np.clip(strength, -1.0, 1.0))
        if value >= 0.0:
            result[left] = (1.0 - value) * result[left] + value * mixed
            result[right] = (1.0 - value) * result[right] + value * mixed
        else:
            result[left] -= abs(value) * fields[right]
            result[right] -= abs(value) * fields[left]
    return result


def _quota_assign(
    fields: np.ndarray,
    phases: tuple[int, ...],
    target_counts: Mapping[int, int],
    *,
    temperature: float,
) -> np.ndarray:
    flat = fields.reshape(len(phases), -1) / max(float(temperature), 1.0e-8)
    target = np.asarray([target_counts[phase] for phase in phases], dtype=np.int64)
    biases = np.zeros(len(phases), dtype=np.float64)
    assignment = np.argmax(flat, axis=0)
    for iteration in range(240):
        assignment = np.argmax(flat + biases[:, None], axis=0)
        counts = np.bincount(assignment, minlength=len(phases))
        error = target - counts
        if np.all(error == 0):
            break
        learning_rate = 1.0 / np.sqrt(iteration + 1.0)
        biases += learning_rate * error / max(len(assignment), 1)
    assignment = _repair_quotas(flat + biases[:, None], assignment, target)
    phase_values = np.asarray(phases, dtype=np.int32)
    return phase_values[assignment].reshape(fields.shape[1:])


def _repair_quotas(
    scores: np.ndarray,
    assignment: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    result = assignment.copy()
    counts = np.bincount(result, minlength=len(target)).astype(np.int64)
    max_moves = len(result) * max(len(target), 1)
    moves = 0
    while not np.array_equal(counts, target):
        over = np.flatnonzero(counts > target)
        under = np.flatnonzero(counts < target)
        if not len(over) or not len(under):
            break
        best_move: tuple[float, int, int, int] | None = None
        for source in over:
            indices = np.flatnonzero(result == source)
            spare = int(counts[source] - target[source])
            if spare <= 0 or not len(indices):
                continue
            source_score = scores[source, indices]
            for destination in under:
                need = int(target[destination] - counts[destination])
                if need <= 0:
                    continue
                penalties = source_score - scores[destination, indices]
                local = int(np.argmin(penalties))
                candidate = (float(penalties[local]), int(indices[local]), int(source), int(destination))
                if best_move is None or candidate[0] < best_move[0]:
                    best_move = candidate
        if best_move is None:
            raise RuntimeError("could not satisfy exact phase quotas")
        _, index, source, destination = best_move
        result[index] = destination
        counts[source] -= 1
        counts[destination] += 1
        moves += 1
        if moves > max_moves:
            raise RuntimeError("phase quota repair failed to converge")
    return result


def refine_by_label_swaps(
    volume: MicrostructureVolume,
    targets: Mapping[str, float],
    *,
    steps: int,
    seed: int = 0,
) -> tuple[MicrostructureVolume, float]:
    """Annealed cross-phase swaps that preserve every phase count exactly."""

    labels = volume.labels.copy()
    rng = np.random.default_rng(seed)
    current = descriptor_loss(analyze_microstructure(volume), targets)
    flat = labels.ravel()
    for step in range(steps):
        first = int(rng.integers(0, len(flat)))
        candidates = np.flatnonzero(flat != flat[first])
        if not len(candidates):
            break
        second = int(candidates[int(rng.integers(0, len(candidates)))])
        flat[first], flat[second] = flat[second], flat[first]
        candidate_volume = MicrostructureVolume(
            labels=labels,
            voxel_size_m=volume.voxel_size_m,
            origin_m=volume.origin_m,
            phase_names=volume.phase_names,
            metadata=volume.metadata,
        )
        candidate = descriptor_loss(analyze_microstructure(candidate_volume), targets)
        temperature = max(1.0e-6, 1.0 - step / max(steps, 1))
        accept = candidate <= current or rng.random() < np.exp((current - candidate) / temperature)
        if accept:
            current = candidate
        else:
            flat[first], flat[second] = flat[second], flat[first]
    return (
        MicrostructureVolume(
            labels=labels,
            voxel_size_m=volume.voxel_size_m,
            origin_m=volume.origin_m,
            phase_names=volume.phase_names,
            metadata={**volume.metadata, "descriptor_refined": True},
        ),
        current,
    )


def enforce_generation_constraints(
    result: GenerationResult,
    condition: MicrostructureCondition,
    *,
    preserve_exact_fractions: bool = True,
) -> GenerationResult:
    """Apply hard composition and face-to-face connectivity constraints.

    Connectivity is enforced by label swaps, so every phase count remains unchanged.
    Lines are selected where the requested phase already occupies the most voxels and
    never overwrite a previously protected path belonging to another phase.
    """

    labels = np.asarray(result.volume.labels).copy()
    present = {int(item) for item in np.unique(labels)}
    unexpected = sorted(present - set(condition.phases))
    if unexpected:
        raise ValueError(f"backend produced phase ids absent from the condition: {unexpected}")
    before_counts = {
        phase: int(np.count_nonzero(labels == phase)) for phase in condition.phases
    }
    expected = dict(condition.exact_phase_counts())
    if preserve_exact_fractions and before_counts != expected:
        raise ValueError(
            f"backend did not preserve exact phase counts: expected {expected}, got {before_counts}"
        )
    protected: dict[int, set[tuple[int, int, int]]] = {
        phase: set() for phase in condition.phases
    }
    reports: list[Mapping[str, object]] = []
    for phase in sorted(condition.percolation_axes):
        for axis in condition.percolation_axes[phase]:
            path = _best_percolation_line(labels, phase, axis, protected)
            incoming = [index for index in path if int(labels[index]) != phase]
            protected_all = set().union(*protected.values()) if protected else set()
            forbidden = protected_all | set(path)
            donor_coordinates = [
                tuple(int(item) for item in coordinate)
                for coordinate in np.argwhere(labels == phase)
                if tuple(int(item) for item in coordinate) not in forbidden
            ]
            if len(donor_coordinates) < len(incoming):
                raise ValueError(
                    f"phase {phase} cannot satisfy percolation on axis {axis} while "
                    "preserving phase counts and previously protected paths"
                )
            for destination, donor in zip(incoming, donor_coordinates, strict=False):
                displaced = int(labels[destination])
                labels[destination] = phase
                labels[donor] = displaced
            protected[phase].update(path)
            reports.append(
                {
                    "phase": phase,
                    "axis": axis,
                    "path_voxels": len(path),
                    "label_swaps": len(incoming),
                }
            )
    volume = MicrostructureVolume(
        labels=labels,
        voxel_size_m=result.volume.voxel_size_m,
        origin_m=result.volume.origin_m,
        phase_names=result.volume.phase_names,
        metadata={
            **result.volume.metadata,
            "hard_percolation_enforced": bool(reports),
            "percolation_repairs": reports,
        },
    )
    achieved = {
        phase: int(np.count_nonzero(labels == phase)) for phase in condition.phases
    }
    if achieved != before_counts:
        raise RuntimeError("percolation repair changed phase counts")
    metrics = analyze_microstructure(volume)
    failures = [
        (phase, axis)
        for phase, axes in condition.percolation_axes.items()
        for axis in axes
        if not metrics.phases[phase].percolates[axis]
    ]
    if failures:
        raise RuntimeError(f"failed to enforce percolation constraints: {failures}")
    return replace(
        result,
        volume=volume,
        exact_counts=expected,
        achieved_counts=achieved,
        metadata={**result.metadata, "percolation_repairs": reports},
    )


def _best_percolation_line(
    labels: np.ndarray,
    phase: int,
    axis: int,
    protected: Mapping[int, set[tuple[int, int, int]]],
) -> tuple[tuple[int, int, int], ...]:
    other_protected = set().union(
        *(points for key, points in protected.items() if key != phase)
    ) if protected else set()
    blocked = np.zeros(labels.shape, dtype=bool)
    for index in other_protected:
        blocked[index] = True
    target = labels == phase
    transverse_axes = tuple(item for item in range(3) if item != axis)
    count_map = target.sum(axis=axis)
    blocked_map = blocked.any(axis=axis)
    score = np.where(blocked_map, -1, count_map)
    if np.all(score < 0):
        raise ValueError(
            f"no non-conflicting line is available for phase {phase} on axis {axis}"
        )
    transverse = np.unravel_index(int(np.argmax(score)), score.shape)
    path: list[tuple[int, int, int]] = []
    for coordinate in range(labels.shape[axis]):
        index = [0, 0, 0]
        index[axis] = coordinate
        index[transverse_axes[0]] = int(transverse[0])
        index[transverse_axes[1]] = int(transverse[1])
        path.append(tuple(index))
    return tuple(path)


__all__ = [
    "GenerationResult",
    "SpectralConditionalGenerator",
    "enforce_generation_constraints",
    "refine_by_label_swaps",
]
