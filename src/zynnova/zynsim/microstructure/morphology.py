"""ZynMorph spectral-topological morphology representation and fast editing.

The fixed-size descriptor combines phase fraction, specific interface density,
axis-resolved chord statistics, radial power-spectrum moments, Minkowski-style
connectivity measures, and a structure-tensor anisotropy signature.  It is
intended as a compact, reproducible conditioning vector rather than a learned
black-box embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class PhaseMorphology:
    label: int
    fraction: float
    specific_interface: float
    chord_mean_xyz: tuple[float, float, float]
    chord_std_xyz: tuple[float, float, float]
    chord_p90_xyz: tuple[float, float, float]
    spectral_centroid: float
    spectral_bandwidth: float
    low_frequency_fraction: float
    component_density: float
    percolates_xyz: tuple[bool, bool, bool]
    anisotropy_eigenvalues: tuple[float, float, float]
    anisotropy_principal_axis: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ZynMorphDescriptor:
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    phases: tuple[PhaseMorphology, ...]
    vector: np.ndarray
    fingerprint: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", np.ascontiguousarray(self.vector, dtype=np.float64))
        object.__setattr__(self, "metadata", dict(self.metadata))


def characterize_morphology(
    labels: np.ndarray,
    *,
    spacing: float | Sequence[float] = 1.0,
    phase_labels: Sequence[int] | None = None,
    spectral_bins: int = 12,
    maximum_fft_voxels: int = 16_777_216,
    maximum_analysis_voxels: int = 16_777_216,
) -> ZynMorphDescriptor:
    """Compute the deterministic ZynMorph descriptor for a 2-D/3-D label field.

    For very large arrays, geometric/spectral terms are evaluated on a
    deterministic strided representative volume while phase fractions are
    measured on the complete field.  This bounds temporary memory without
    losing the exact domain proportions that condition downstream generation.
    """

    raw = np.asarray(labels)
    field = raw if raw.dtype == np.int32 and raw.flags.c_contiguous else np.ascontiguousarray(raw, dtype=np.int32)
    if field.ndim not in (2, 3) or min(field.shape) < 2:
        raise ValueError("labels must be a non-trivial 2-D or 3-D array")
    if maximum_analysis_voxels < 1:
        raise ValueError("maximum_analysis_voxels must be positive")
    spacing_tuple = _spacing(spacing, field.ndim)
    selected = tuple(sorted(map(int, np.unique(field) if phase_labels is None else phase_labels)))
    analysis = field
    analysis_spacing = spacing_tuple
    strides = (1,) * field.ndim
    if field.size > maximum_analysis_voxels:
        scale = (field.size / maximum_analysis_voxels) ** (1.0 / field.ndim)
        strides = tuple(max(1, int(np.ceil(scale))) for _ in range(field.ndim))
        analysis = field[tuple(slice(None, None, stride) for stride in strides)]
        analysis_spacing = tuple(spacing_tuple[i] * strides[i] for i in range(field.ndim))
    if not selected:
        raise ValueError("no phases selected")
    phases: list[PhaseMorphology] = []
    vector_parts: list[np.ndarray] = []
    for label in selected:
        mask = analysis == label
        fraction = float(np.count_nonzero(field == label) / field.size)
        interface = _specific_interface(mask, analysis_spacing)
        chord_mean, chord_std, chord_p90 = _chord_statistics(mask, analysis_spacing)
        centroid, bandwidth, low_fraction, spectrum = _spectrum_statistics(
            mask, analysis_spacing, spectral_bins, maximum_fft_voxels
        )
        component_density, percolates = _connectivity(mask)
        eigenvalues, axis = _anisotropy(mask, spacing_tuple)
        phase = PhaseMorphology(
            label=label,
            fraction=fraction,
            specific_interface=interface,
            chord_mean_xyz=_pad3(chord_mean),
            chord_std_xyz=_pad3(chord_std),
            chord_p90_xyz=_pad3(chord_p90),
            spectral_centroid=centroid,
            spectral_bandwidth=bandwidth,
            low_frequency_fraction=low_fraction,
            component_density=component_density,
            percolates_xyz=tuple(bool(v) for v in _pad3(percolates, False)),
            anisotropy_eigenvalues=_pad3(eigenvalues),
            anisotropy_principal_axis=_pad3(axis),
        )
        phases.append(phase)
        vector_parts.append(
            np.asarray(
                [fraction, interface, *phase.chord_mean_xyz, *phase.chord_std_xyz,
                 *phase.chord_p90_xyz, centroid, bandwidth, low_fraction,
                 component_density, *map(float, phase.percolates_xyz),
                 *phase.anisotropy_eigenvalues, *phase.anisotropy_principal_axis,
                 *spectrum],
                dtype=np.float64,
            )
        )
    vector = np.concatenate(vector_parts)
    digest = sha256(vector.tobytes() + np.asarray(field.shape, dtype=np.int64).tobytes()).hexdigest()
    return ZynMorphDescriptor(
        shape=tuple(map(int, field.shape)),
        spacing=spacing_tuple,
        phases=tuple(phases),
        vector=vector,
        fingerprint=f"zynmorph-v1:{digest[:24]}",
        metadata={
            "spectral_bins": int(spectral_bins),
            "phase_order": selected,
            "analysis_shape": tuple(map(int, analysis.shape)),
            "analysis_strides": tuple(map(int, strides)),
            "exact_phase_fractions": True,
        },
    )


def retarget_phase_fractions(
    labels: np.ndarray,
    target_fractions: Mapping[int, float],
    *,
    spacing: float | Sequence[float] = 1.0,
    fixed_mask: np.ndarray | None = None,
    preserve_topology_weight: float = 1.0,
    iterations: int = 64,
    method: Literal["auto", "distance", "streaming"] = "auto",
    maximum_distance_voxels: int = 16_777_216,
    chunk_size_x: int = 32,
    in_place: bool = False,
    random_seed: int = 0,
) -> np.ndarray:
    """Retarget phase fractions to exact integer voxel quotas.

    ``method="distance"`` uses global signed-distance competition and best
    preserves smooth interfaces for moderate arrays.  ``method="streaming"``
    uses bounded-memory interface growth/erosion with deterministic local
    affinity scores.  The latter never allocates a phase-by-volume score stack
    and is therefore appropriate for memory maps and billion-voxel data.

    The streaming score favours voxels adjacent to the receiving phase while
    penalizing removal from the interior of the donor phase.  Every transfer is
    made between a surplus and a deficit phase, so the final integer quotas are
    exact.  ``fixed_mask`` voxels are never changed.  Set ``in_place=True`` to
    edit an existing writable array or memory map without allocating a second
    label volume.
    """

    raw = np.asarray(labels)
    if raw.ndim not in (2, 3) or raw.size == 0:
        raise ValueError("labels must be a non-empty 2-D or 3-D array")
    if not np.issubdtype(raw.dtype, np.integer):
        if not np.allclose(raw, np.rint(raw)):
            raise ValueError("labels must contain integer values")
    field = raw if raw.dtype == np.int32 and raw.flags.c_contiguous else np.ascontiguousarray(raw, dtype=np.int32)
    if in_place and (not field.flags.writeable or field is not raw):
        raise ValueError("in_place=True requires a writable C-contiguous int32 input")
    phases = tuple(sorted(map(int, np.unique(field))))
    requested = {int(k): float(v) for k, v in target_fractions.items()}
    if not requested:
        return field if in_place else field.copy()
    if set(requested) - set(phases):
        raise ValueError("target_fractions contains phases absent from labels")
    if any((not np.isfinite(value)) or value < 0.0 for value in requested.values()):
        raise ValueError("target fractions must be finite and non-negative")
    if method not in {"auto", "distance", "streaming"}:
        raise ValueError("method must be 'auto', 'distance', or 'streaming'")
    if maximum_distance_voxels < 1 or chunk_size_x < 1:
        raise ValueError("memory/chunk limits must be positive")

    current_counts = np.asarray([np.count_nonzero(field == phase) for phase in phases], dtype=np.int64)
    current = current_counts / field.size
    target = _complete_fraction_target(phases, current, requested)
    quotas = _largest_remainder_counts(target, field.size)

    locked = np.zeros(field.shape, dtype=bool) if fixed_mask is None else np.asarray(fixed_mask, dtype=bool)
    if locked.shape != field.shape:
        raise ValueError("fixed_mask shape mismatch")
    locked_counts = np.asarray([np.count_nonzero(locked & (field == phase)) for phase in phases], dtype=np.int64)
    if np.any(locked_counts > quotas):
        bad = [phases[i] for i in np.flatnonzero(locked_counts > quotas)]
        raise ValueError(
            "fixed_mask makes the requested phase quotas infeasible for phases "
            f"{bad}; locked voxels already exceed their targets"
        )
    if np.array_equal(current_counts, quotas):
        return field if in_place else field.copy()

    resolved = method
    if resolved == "auto":
        score_voxels = field.size * max(len(phases), 1)
        resolved = "streaming" if isinstance(labels, np.memmap) or score_voxels > maximum_distance_voxels else "distance"
    if resolved == "streaming":
        result = field if in_place else field.copy()
        _retarget_streaming_exact(
            result,
            phases,
            quotas,
            locked,
            chunk_size_x=int(chunk_size_x),
            random_seed=int(random_seed),
        )
        return result

    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("distance-based fraction retargeting requires scipy") from exc
    sampling = _spacing(spacing, field.ndim)
    scale = max(float(preserve_topology_weight), 1e-12)
    phase_array = np.asarray(phases, dtype=np.int32)
    score_stack = np.empty((len(phases),) + field.shape, dtype=np.float32)
    for index, phase in enumerate(phases):
        inside = field == phase
        signed = distance_transform_edt(inside, sampling=sampling) - distance_transform_edt(
            ~inside, sampling=sampling
        )
        score_stack[index] = np.asarray(signed * scale, dtype=np.float32)

    free = ~locked
    free_count = int(np.count_nonzero(free))
    if free_count == 0:
        if np.array_equal(locked_counts, quotas):
            return field if in_place else field.copy()
        raise ValueError("fixed_mask locks every voxel but target fractions differ")

    biases = np.zeros(len(phases), dtype=np.float64)
    characteristic = max(float(np.mean(sampling)), 1e-12)
    gain = 2.5 * characteristic
    result = field if in_place else field.copy()
    broadcast = (len(phases),) + (1,) * field.ndim
    for _ in range(max(1, int(iterations))):
        chosen = np.argmax(score_stack + biases.reshape(broadcast), axis=0)
        result[free] = phase_array[chosen[free]]
        result[locked] = field[locked]
        counts = np.asarray([np.count_nonzero(result == phase) for phase in phases], dtype=np.int64)
        mismatch = quotas - counts
        if np.all(mismatch == 0):
            return result
        biases += gain * mismatch / max(free_count, 1)
        biases -= float(np.mean(biases))
        gain *= 0.985

    _repair_integer_quotas(result, field, free, phases, quotas, score_stack)
    final_counts = np.asarray([np.count_nonzero(result == phase) for phase in phases], dtype=np.int64)
    if not np.array_equal(final_counts, quotas):  # pragma: no cover - invariant guard
        raise RuntimeError(f"internal quota repair failed: {final_counts.tolist()} != {quotas.tolist()}")
    return result


def _retarget_streaming_exact(
    result: np.ndarray,
    phases: tuple[int, ...],
    quotas: np.ndarray,
    locked: np.ndarray,
    *,
    chunk_size_x: int,
    random_seed: int,
) -> None:
    """Exact bounded-memory morphology editing by interface-ranked transfers."""

    counts = np.asarray([np.count_nonzero(result == phase) for phase in phases], dtype=np.int64)
    transfer_index = 0
    while not np.array_equal(counts, quotas):
        surplus = counts - quotas
        deficit = quotas - counts
        target_index = int(np.argmax(deficit))
        source_index = int(np.argmax(surplus))
        need = int(deficit[target_index])
        available = int(surplus[source_index])
        if need <= 0 or available <= 0:  # pragma: no cover - conservation guard
            raise RuntimeError("phase quota transfer became inconsistent")
        amount = min(need, available)
        source = phases[source_index]
        target = phases[target_index]
        moved = _stream_transfer(
            result,
            locked,
            source=source,
            target=target,
            amount=amount,
            chunk_size_x=chunk_size_x,
            rotation=(random_seed + 104729 * transfer_index),
        )
        if moved != amount:  # pragma: no cover - invariant guard
            raise RuntimeError(f"streaming phase transfer moved {moved}, expected {amount}")
        counts[source_index] -= moved
        counts[target_index] += moved
        transfer_index += 1


def _stream_transfer(
    result: np.ndarray,
    locked: np.ndarray,
    *,
    source: int,
    target: int,
    amount: int,
    chunk_size_x: int,
    rotation: int,
) -> int:
    """Move exactly ``amount`` source voxels using a two-pass local score."""

    # score = 8 * receiving-neighbours - donor-neighbours, range [-6, 48].
    score_min, score_max = -6, 48
    histogram = np.zeros(score_max - score_min + 1, dtype=np.int64)
    chunks = list(_rotated_slabs(result.shape[0], chunk_size_x, rotation))
    for start, stop in chunks:
        score, candidates = _transfer_score_slab(result, locked, source, target, start, stop)
        if np.any(candidates):
            histogram += np.bincount(
                (score[candidates] - score_min).astype(np.int64),
                minlength=len(histogram),
            )
    if int(histogram.sum()) < amount:
        raise ValueError("fixed_mask leaves too few donor voxels for requested fractions")
    cumulative = 0
    cutoff = score_min
    needed_at_cutoff = 0
    for raw_score in range(score_max, score_min - 1, -1):
        count = int(histogram[raw_score - score_min])
        if cumulative + count >= amount:
            cutoff = raw_score
            needed_at_cutoff = amount - cumulative
            break
        cumulative += count

    moved = 0
    remaining_equal = needed_at_cutoff
    for start, stop in chunks:
        score, candidates = _transfer_score_slab(result, locked, source, target, start, stop)
        remaining_total = amount - moved
        high_flat = np.flatnonzero(candidates.ravel() & (score.ravel() > cutoff))
        take_high = min(remaining_total, len(high_flat))
        if take_high:
            slab_flat = result[start:stop].reshape(-1)
            slab_flat[high_flat[:take_high]] = target
            moved += take_high
            remaining_total -= take_high
        if remaining_total > 0 and remaining_equal > 0:
            equal_flat = np.flatnonzero(candidates.ravel() & (score.ravel() == cutoff))
            take = min(remaining_total, remaining_equal, len(equal_flat))
            if take:
                slab_flat = result[start:stop].reshape(-1)
                slab_flat[equal_flat[:take]] = target
                moved += take
                remaining_equal -= take
        if moved >= amount:
            break
    return moved


def _transfer_score_slab(
    values: np.ndarray,
    locked: np.ndarray,
    source: int,
    target: int,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    slab = values[start:stop]
    candidates = (slab == source) & ~locked[start:stop]
    target_neighbors = np.zeros(slab.shape, dtype=np.int8)
    source_neighbors = np.zeros(slab.shape, dtype=np.int8)

    def add_pair(destination: tuple, neighbor: np.ndarray) -> None:
        target_neighbors[destination] += neighbor == target
        source_neighbors[destination] += neighbor == source

    if values.ndim == 3:
        if start > 0:
            add_pair((0, slice(None), slice(None)), values[start - 1])
        if stop < values.shape[0]:
            add_pair((-1, slice(None), slice(None)), values[stop])
        if len(slab) > 1:
            add_pair((slice(1, None), slice(None), slice(None)), slab[:-1])
            add_pair((slice(None, -1), slice(None), slice(None)), slab[1:])
        add_pair((slice(None), slice(1, None), slice(None)), slab[:, :-1, :])
        add_pair((slice(None), slice(None, -1), slice(None)), slab[:, 1:, :])
        add_pair((slice(None), slice(None), slice(1, None)), slab[:, :, :-1])
        add_pair((slice(None), slice(None), slice(None, -1)), slab[:, :, 1:])
    else:
        if start > 0:
            add_pair((0, slice(None)), values[start - 1])
        if stop < values.shape[0]:
            add_pair((-1, slice(None)), values[stop])
        if len(slab) > 1:
            add_pair((slice(1, None), slice(None)), slab[:-1])
            add_pair((slice(None, -1), slice(None)), slab[1:])
        add_pair((slice(None), slice(1, None)), slab[:, :-1])
        add_pair((slice(None), slice(None, -1)), slab[:, 1:])
    score = 8 * target_neighbors.astype(np.int16) - source_neighbors.astype(np.int16)
    return score, candidates


def _rotated_slabs(length: int, chunk: int, rotation: int):
    slabs = [(start, min(length, start + chunk)) for start in range(0, length, chunk)]
    if not slabs:
        return
    offset = int(rotation) % len(slabs)
    yield from slabs[offset:]
    yield from slabs[:offset]

def _complete_fraction_target(
    phases: tuple[int, ...],
    current: np.ndarray,
    requested: Mapping[int, float],
) -> np.ndarray:
    index = {phase: i for i, phase in enumerate(phases)}
    target = np.zeros(len(phases), dtype=float)
    if len(requested) == len(phases):
        total = float(sum(requested.values()))
        if total <= 0.0:
            raise ValueError("target fractions must sum to a positive value")
        for phase, value in requested.items():
            target[index[phase]] = value / total
        return target

    specified_sum = float(sum(requested.values()))
    if specified_sum > 1.0 + 1e-12:
        raise ValueError("partial target fractions cannot sum to more than one")
    for phase, value in requested.items():
        target[index[phase]] = value
    unspecified = [i for i, phase in enumerate(phases) if phase not in requested]
    residual = max(0.0, 1.0 - specified_sum)
    current_residual = float(np.sum(current[unspecified]))
    if unspecified:
        if current_residual > 0.0:
            target[unspecified] = residual * current[unspecified] / current_residual
        else:
            target[unspecified] = residual / len(unspecified)
    target /= max(float(np.sum(target)), np.finfo(float).tiny)
    return target


def _largest_remainder_counts(fractions: np.ndarray, total: int) -> np.ndarray:
    raw = np.asarray(fractions, dtype=float) * int(total)
    counts = np.floor(raw).astype(np.int64)
    remaining = int(total - np.sum(counts))
    if remaining:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remaining]] += 1
    return counts


def _repair_integer_quotas(
    result: np.ndarray,
    original: np.ndarray,
    free: np.ndarray,
    phases: tuple[int, ...],
    quotas: np.ndarray,
    scores: np.ndarray,
) -> None:
    phase_index = {phase: i for i, phase in enumerate(phases)}
    counts = np.asarray([np.count_nonzero(result == phase) for phase in phases], dtype=np.int64)
    for target_index, target_phase in enumerate(phases):
        need = int(quotas[target_index] - counts[target_index])
        while need > 0:
            surplus = counts - quotas
            sources = np.flatnonzero(surplus > 0)
            if sources.size == 0:
                raise RuntimeError("quota repair ran out of donor phases")
            candidate_indices: list[np.ndarray] = []
            candidate_costs: list[np.ndarray] = []
            candidate_sources: list[np.ndarray] = []
            flat_result = result.ravel()
            flat_free = free.ravel()
            flat_scores = scores.reshape(len(phases), -1)
            for source_index in sources:
                source_phase = phases[int(source_index)]
                eligible = np.flatnonzero(flat_free & (flat_result == source_phase))
                take = min(int(surplus[source_index]), need, len(eligible))
                if take <= 0:
                    continue
                cost = flat_scores[source_index, eligible] - flat_scores[target_index, eligible]
                if take < len(eligible):
                    local = np.argpartition(cost, take - 1)[:take]
                    eligible = eligible[local]
                    cost = cost[local]
                candidate_indices.append(eligible)
                candidate_costs.append(cost)
                candidate_sources.append(np.full(len(eligible), source_index, dtype=np.int32))
            if not candidate_indices:
                raise RuntimeError("quota repair found no eligible donor voxels")
            indices = np.concatenate(candidate_indices)
            costs = np.concatenate(candidate_costs)
            source_ids = np.concatenate(candidate_sources)
            take = min(need, len(indices))
            if take < len(indices):
                chosen = np.argpartition(costs, take - 1)[:take]
                indices = indices[chosen]
                source_ids = source_ids[chosen]
            flat_result[indices] = target_phase
            moved_by_source = np.bincount(source_ids, minlength=len(phases)).astype(np.int64)
            counts -= moved_by_source
            counts[target_index] += len(indices)
            need = int(quotas[target_index] - counts[target_index])

def _spacing(value: float | Sequence[float], ndim: int) -> tuple[float, ...]:
    if np.isscalar(value):
        result = (float(value),) * ndim
    else:
        result = tuple(map(float, value))
    if len(result) != ndim or min(result) <= 0.0:
        raise ValueError("spacing must contain one positive value per dimension")
    return result


def _specific_interface(mask: np.ndarray, spacing: tuple[float, ...]) -> float:
    area = 0.0
    for axis in range(mask.ndim):
        transitions = np.count_nonzero(np.diff(mask.astype(np.int8), axis=axis))
        face_measure = float(np.prod([spacing[i] for i in range(mask.ndim) if i != axis]))
        area += transitions * face_measure
    volume = mask.size * float(np.prod(spacing))
    return float(area / max(volume, np.finfo(float).tiny))


def _chord_statistics(mask: np.ndarray, spacing: tuple[float, ...]) -> tuple[list[float], list[float], list[float]]:
    means: list[float] = []
    stds: list[float] = []
    p90s: list[float] = []
    for axis in range(mask.ndim):
        moved = np.moveaxis(mask, axis, -1).reshape(-1, mask.shape[axis])
        lengths: list[int] = []
        for row in moved:
            padded = np.concatenate(([False], row, [False])).astype(np.int8)
            edges = np.diff(padded)
            starts = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1)
            lengths.extend((ends - starts).tolist())
        physical = np.asarray(lengths, dtype=float) * spacing[axis]
        means.append(float(np.mean(physical)) if physical.size else 0.0)
        stds.append(float(np.std(physical)) if physical.size else 0.0)
        p90s.append(float(np.quantile(physical, 0.9)) if physical.size else 0.0)
    return means, stds, p90s


def _spectrum_statistics(mask: np.ndarray, spacing: tuple[float, ...], bins: int, maximum: int) -> tuple[float, float, float, np.ndarray]:
    values = mask.astype(float) - float(np.mean(mask))
    if values.size > maximum:
        factors = [max(1, int(np.ceil(size / max(8, (maximum ** (1 / values.ndim)))))) for size in values.shape]
        slices = tuple(slice(None, None, factor) for factor in factors)
        values = values[slices]
        spacing = tuple(spacing[i] * factors[i] for i in range(values.ndim))
    power = np.abs(np.fft.fftn(values)) ** 2
    frequencies = np.meshgrid(*[np.fft.fftfreq(values.shape[i], d=spacing[i]) for i in range(values.ndim)], indexing="ij")
    radius = np.sqrt(sum(component * component for component in frequencies))
    flat_r = radius.ravel()
    flat_p = power.ravel()
    nonzero = flat_r > 0
    flat_r = flat_r[nonzero]
    flat_p = flat_p[nonzero]
    total = float(np.sum(flat_p))
    if total <= 0.0 or flat_r.size == 0:
        return 0.0, 0.0, 0.0, np.zeros(bins, dtype=float)
    centroid = float(np.sum(flat_r * flat_p) / total)
    bandwidth = float(np.sqrt(np.sum((flat_r - centroid) ** 2 * flat_p) / total))
    cutoff = float(np.quantile(flat_r, 0.25))
    low = float(np.sum(flat_p[flat_r <= cutoff]) / total)
    edges = np.linspace(0.0, float(np.max(flat_r)) + np.finfo(float).eps, bins + 1)
    radial = np.zeros(bins, dtype=float)
    indices = np.clip(np.searchsorted(edges, flat_r, side="right") - 1, 0, bins - 1)
    np.add.at(radial, indices, flat_p)
    radial /= max(float(np.sum(radial)), np.finfo(float).tiny)
    return centroid, bandwidth, low, radial


def _connectivity(mask: np.ndarray) -> tuple[float, tuple[bool, ...]]:
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover
        return 0.0, tuple(False for _ in range(mask.ndim))
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    labels, count = ndimage.label(mask, structure=structure)
    density = float(count / max(mask.size, 1))
    percolates = []
    for axis in range(mask.ndim):
        first = np.unique(np.take(labels, 0, axis=axis))
        last = np.unique(np.take(labels, -1, axis=axis))
        shared = np.intersect1d(first[first > 0], last[last > 0])
        percolates.append(bool(shared.size))
    return density, tuple(percolates)


def _anisotropy(mask: np.ndarray, spacing: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    gradients = np.gradient(mask.astype(float), *spacing, edge_order=1)
    matrix = np.zeros((mask.ndim, mask.ndim), dtype=float)
    for i in range(mask.ndim):
        for j in range(mask.ndim):
            matrix[i, j] = float(np.mean(gradients[i] * gradients[j]))
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    values = values[order]
    axis = vectors[:, order[0]] if np.any(values > 0.0) else np.eye(mask.ndim)[:, 0]
    scale = max(float(np.sum(values)), np.finfo(float).tiny)
    return values / scale, axis


def _pad3(values: Sequence[object], fill: object = 0.0) -> tuple:
    result = list(values[:3])
    result.extend([fill] * (3 - len(result)))
    return tuple(result)


__all__ = [
    "PhaseMorphology",
    "ZynMorphDescriptor",
    "characterize_morphology",
    "retarget_phase_fractions",
]
