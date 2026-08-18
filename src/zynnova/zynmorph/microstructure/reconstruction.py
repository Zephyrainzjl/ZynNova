"""Descriptor-space reconstruction using gradients or Yeong-Torquato swaps."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Mapping

import numpy as np

from .characterization import Characterization, characterize_microstructure
from .descriptors import (
    compute_descriptor_numpy_spatial,
    compute_descriptor_torch_spatial,
    descriptor_definition,
    phase_probabilities,
)
from .losses import compute_loss
from .optimizers import make_torch_optimizer, optimizer_definition
from .settings import CharacterizationSettings, ReconstructionSettings


@dataclass(frozen=True, slots=True)
class ConvergenceRecord:
    iteration: int
    loss: float
    descriptor_losses: Mapping[str, float]
    temperature: float | None = None
    accepted: bool | None = None


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    labels: np.ndarray
    probabilities: np.ndarray | None
    phase_ids: tuple[int, ...]
    convergence: tuple[ConvergenceRecord, ...]
    settings: ReconstructionSettings
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def final_loss(self) -> float:
        return float(self.convergence[-1].loss) if self.convergence else math.nan


def _device(settings: ReconstructionSettings) -> str:
    if settings.device != "auto":
        return settings.device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _volume_fractions(target: Characterization) -> np.ndarray:
    """Return full phase fractions even for single-phase characterizations.

    MCR-style ``use_multiphase=False`` stores descriptors for phase 1 only,
    while reconstruction still needs the complete binary phase fractions for
    initialization and exact phase-swap annealing.  ZynNova therefore treats
    the characterization metadata as authoritative when it contains the full
    phase vector and only falls back to the VolumeFractions descriptor.
    """

    metadata_values = target.metadata.get("volume_fractions")
    if metadata_values is not None:
        result = np.asarray(metadata_values, dtype=np.float64).reshape(-1)
        if result.size == len(target.phase_ids):
            result = np.clip(result, 1.0e-12, None)
            return result / result.sum()

    if "VolumeFractions" in target.descriptors:
        values = np.asarray(
            target.descriptors["VolumeFractions"].values, dtype=np.float64
        ).reshape(-1)
        # Single-phase binary characterization describes phase 1 only.
        if len(target.phase_ids) == 2 and values.size == 1:
            second = float(np.clip(values[0], 0.0, 1.0))
            return np.asarray([1.0 - second, second], dtype=np.float64)
        # Multigrid may concatenate several levels; the first P entries are
        # the full-resolution phase fractions.
        if values.size >= len(target.phase_ids):
            result = values[: len(target.phase_ids)]
            result = np.clip(result, 1.0e-12, None)
            return result / result.sum()

    return np.full(len(target.phase_ids), 1.0 / len(target.phase_ids))


def _descriptor_prob_torch(prob: Any, target: Characterization, settings: ReconstructionSettings):
    """Select the descriptor channels used by single-/multi-phase modes."""

    if settings.use_multiphase:
        return prob
    if len(target.phase_ids) != 2 or int(prob.shape[0]) != 2:
        raise ValueError("use_multiphase=False is supported only for binary reconstruction")
    # Match MCRpy's single-phase convention: characterize phase 1 only.
    return prob[1:2]


def _descriptor_prob_numpy(prob: np.ndarray, target: Characterization, settings: ReconstructionSettings) -> np.ndarray:
    if settings.use_multiphase:
        return prob
    if len(target.phase_ids) != 2 or prob.shape[0] != 2:
        raise ValueError("use_multiphase=False is supported only for binary reconstruction")
    return prob[1:2]


def _initial_logits(
    shape: tuple[int, ...],
    target: Characterization,
    settings: ReconstructionSettings,
    initial_microstructure: Any | None,
    *,
    dtype: Any,
    device: str,
):
    import torch

    phases = target.phase_ids
    rng = np.random.default_rng(settings.seed)
    if initial_microstructure is not None:
        prob, init_phases = phase_probabilities(initial_microstructure, phase_ids=phases)
        if tuple(prob.shape[1:]) != tuple(shape):
            raise ValueError("initial microstructure shape differs from desired shape")
        if init_phases != phases:
            raise ValueError("initial microstructure phase IDs differ from target")
        prob = np.clip(prob, 1.0e-6, 1.0)
        logits = np.log(prob)
    else:
        vf = _volume_fractions(target)
        base = np.log(np.clip(vf, 1.0e-8, None)).reshape((-1,) + (1,) * len(shape))
        noise_scale = 1.0 if settings.initialization == "random" else 0.25
        logits = base + noise_scale * rng.normal(size=(len(phases), *shape))
    return torch.nn.Parameter(torch.as_tensor(logits, dtype=dtype, device=device))


def _targets_torch(target: Characterization, settings: ReconstructionSettings, dtype: Any, device: str):
    import torch

    targets = {}
    for name in settings.descriptor_types:
        canonical = descriptor_definition(name).name
        if canonical not in target.descriptors:
            raise ValueError(f"target characterization lacks descriptor {canonical}")
        targets[canonical] = torch.as_tensor(
            np.asarray(target.descriptors[canonical].values).reshape(-1),
            dtype=dtype,
            device=device,
        )
    return targets


def _torch_loss(logits, target, settings, target_tensors):
    import torch

    prob = torch.softmax(logits, dim=0)
    descriptor_prob = _descriptor_prob_torch(prob, target, settings)
    total = prob.new_zeros(())
    component: dict[str, Any] = {}
    for requested, weight in zip(settings.descriptor_types, settings.weights, strict=True):
        canonical = descriptor_definition(requested).name
        definition = descriptor_definition(canonical)
        if not definition.differentiable:
            raise ValueError(
                f"descriptor {canonical} is non-differentiable; use SimulatedAnnealing"
            )
        actual = compute_descriptor_torch_spatial(
            canonical,
            descriptor_prob,
            slice_mode=settings.slice_mode,
            isotropic=settings.isotropic,
            sample_seed=settings.seed,
            limit_to=settings.limit_to,
            periodic=settings.periodic,
            use_multigrid=settings.use_multigrid_descriptors,
            multigrid_levels=settings.multigrid_levels,
            kwargs=settings.descriptor_kwargs.get(requested, settings.descriptor_kwargs.get(canonical, {})),
        ).reshape(-1)
        desired = target_tensors[canonical]
        if actual.numel() != desired.numel():
            raise ValueError(
                f"descriptor shape mismatch for {canonical}: current={actual.numel()} target={desired.numel()}; "
                "standalone reconstruction must use a shape compatible with the characterization"
            )
        value = compute_loss(settings.loss_type, actual, desired)
        component[canonical] = value
        total = total + float(weight) * value
    # softmax enforces phase sum exactly; penalties are retained only for API
    # compatibility and intentionally default to zero.
    return total, component, prob


def _record(iteration: int, loss: Any, components: Mapping[str, Any]) -> ConvergenceRecord:
    return ConvergenceRecord(
        iteration=int(iteration),
        loss=float(loss.detach().cpu() if hasattr(loss, "detach") else loss),
        descriptor_losses={
            name: float(value.detach().cpu() if hasattr(value, "detach") else value)
            for name, value in components.items()
        },
    )


def _gradient_reconstruct(
    target: Characterization,
    shape: tuple[int, ...],
    settings: ReconstructionSettings,
    initial_microstructure: Any | None,
) -> ReconstructionResult:
    import torch

    device = _device(settings)
    dtype = torch.float64 if settings.dtype == "float64" else torch.float32
    torch.manual_seed(settings.seed)
    logits = _initial_logits(
        shape,
        target,
        settings,
        initial_microstructure,
        dtype=dtype,
        device=device,
    )
    target_tensors = _targets_torch(target, settings, dtype, device)
    definition = optimizer_definition(settings.optimizer_type)
    convergence: list[ConvergenceRecord] = []

    if definition.family == "torch":
        optimizer = make_torch_optimizer(settings.optimizer_type, [logits], settings)
        previous = math.inf
        for iteration in range(settings.max_iter):
            optimizer.zero_grad(set_to_none=True)
            loss, components, prob = _torch_loss(logits, target, settings, target_tensors)
            if not torch.isfinite(loss):
                raise FloatingPointError("microstructure reconstruction produced non-finite loss")
            loss.backward()
            optimizer.step()
            current = float(loss.detach().cpu())
            if iteration % max(1, settings.convergence_data_steps) == 0 or iteration == settings.max_iter - 1:
                convergence.append(_record(iteration, loss, components))
            if abs(previous - current) <= settings.tolerance * max(1.0, abs(previous)):
                if iteration > 2:
                    break
            previous = current
    elif definition.family == "scipy":
        from scipy.optimize import minimize

        shape_logits = tuple(logits.shape)
        x0 = logits.detach().cpu().numpy().reshape(-1)
        call_count = 0

        def objective(x):
            nonlocal call_count, logits
            local = torch.tensor(x.reshape(shape_logits), dtype=dtype, device=device, requires_grad=True)
            loss, components, _ = _torch_loss(local, target, settings, target_tensors)
            loss.backward()
            gradient = local.grad.detach().cpu().numpy().reshape(-1).astype(np.float64)
            if call_count % max(1, settings.convergence_data_steps) == 0:
                convergence.append(_record(call_count, loss, components))
            call_count += 1
            return float(loss.detach().cpu()), gradient

        result = minimize(
            objective,
            x0,
            method=definition.implementation,
            jac=True,
            options={"maxiter": settings.max_iter, "ftol": settings.tolerance},
        )
        logits = torch.nn.Parameter(torch.as_tensor(result.x.reshape(shape_logits), dtype=dtype, device=device))
        loss, components, prob = _torch_loss(logits, target, settings, target_tensors)
        convergence.append(_record(call_count, loss, components))
    else:
        raise ValueError(f"gradient path does not support optimizer {definition.name}")

    with torch.no_grad():
        prob = torch.softmax(logits, dim=0).detach().cpu().numpy()
    labels_index = np.argmax(prob, axis=0)
    phases = np.asarray(target.phase_ids, dtype=np.int64)
    labels = phases[labels_index]
    return ReconstructionResult(
        labels=labels,
        probabilities=prob,
        phase_ids=target.phase_ids,
        convergence=tuple(convergence),
        settings=settings,
        metadata={"engine": "torch-autograd", "device": device},
    )


def _descriptor_objective_numpy(labels: np.ndarray, target: Characterization, settings: ReconstructionSettings):
    prob, _ = phase_probabilities(labels, phase_ids=target.phase_ids)
    descriptor_prob = _descriptor_prob_numpy(prob, target, settings)
    total = 0.0
    components: dict[str, float] = {}
    for requested, weight in zip(settings.descriptor_types, settings.weights, strict=True):
        canonical = descriptor_definition(requested).name
        result = compute_descriptor_numpy_spatial(
            canonical,
            descriptor_prob,
            slice_mode=settings.slice_mode,
            isotropic=settings.isotropic,
            rng=np.random.default_rng(settings.seed),
            limit_to=settings.limit_to,
            periodic=settings.periodic,
            use_multigrid=settings.use_multigrid_descriptors,
            multigrid_levels=settings.multigrid_levels,
            kwargs=settings.descriptor_kwargs.get(requested, settings.descriptor_kwargs.get(canonical, {})),
        )
        desired = np.asarray(target.descriptors[canonical].values).reshape(-1)
        actual = result.flat
        if actual.size != desired.size:
            raise ValueError(f"descriptor size mismatch for {canonical}")
        value = float(compute_loss(settings.loss_type, actual, desired))
        components[canonical] = value
        total += float(weight) * value
    return total, components


def _exact_volume_fraction_labels(shape, phases, fractions, rng):
    total = int(np.prod(shape))
    raw = np.asarray(fractions, dtype=np.float64) * total
    counts = np.floor(raw).astype(int)
    remaining = total - int(counts.sum())
    if remaining:
        order = np.argsort(-(raw - counts))
        counts[order[:remaining]] += 1
    flat = np.concatenate([
        np.full(count, phase, dtype=np.int64)
        for phase, count in zip(phases, counts, strict=True)
    ])
    rng.shuffle(flat)
    return flat.reshape(shape)


def _annealing_reconstruct(
    target: Characterization,
    shape: tuple[int, ...],
    settings: ReconstructionSettings,
    initial_microstructure: Any | None,
) -> ReconstructionResult:
    rng = np.random.default_rng(settings.seed)
    if initial_microstructure is None:
        labels = _exact_volume_fraction_labels(shape, target.phase_ids, _volume_fractions(target), rng)
    else:
        prob, _ = phase_probabilities(initial_microstructure, phase_ids=target.phase_ids)
        if tuple(prob.shape[1:]) != tuple(shape):
            raise ValueError("initial microstructure shape differs from desired shape")
        labels = np.asarray(target.phase_ids)[np.argmax(prob, axis=0)]

    current, components = _descriptor_objective_numpy(labels, target, settings)
    convergence = [ConvergenceRecord(0, current, components, settings.initial_temperature, True)]
    flat = labels.reshape(-1)
    final_temperature = (
        settings.final_temperature
        if settings.final_temperature is not None
        else max(settings.initial_temperature * 1.0e-4, 1.0e-12)
    )
    temperature = settings.initial_temperature
    for iteration in range(1, settings.max_iter + 1):
        # Swap two different phases, exactly preserving every volume fraction.
        first = int(rng.integers(0, flat.size))
        second = int(rng.integers(0, flat.size))
        attempts = 0
        while flat[first] == flat[second] and attempts < 32:
            second = int(rng.integers(0, flat.size))
            attempts += 1
        if flat[first] == flat[second]:
            continue
        flat[first], flat[second] = flat[second], flat[first]
        proposed, proposed_components = _descriptor_objective_numpy(labels, target, settings)
        delta = proposed - current
        accepted = delta <= 0.0 or rng.random() < math.exp(-delta / max(temperature, 1.0e-15))
        if accepted:
            current = proposed
            components = proposed_components
        else:
            flat[first], flat[second] = flat[second], flat[first]
        temperature = max(final_temperature, temperature * settings.cooldown_factor)
        if iteration % max(1, settings.convergence_data_steps) == 0 or iteration == settings.max_iter:
            convergence.append(ConvergenceRecord(iteration, current, components, temperature, accepted))
        if current <= settings.tolerance:
            break
    return ReconstructionResult(
        labels=labels.copy(),
        probabilities=None,
        phase_ids=target.phase_ids,
        convergence=tuple(convergence),
        settings=settings,
        metadata={"engine": "yeong-torquato-phase-swap"},
    )



def _coarsen_characterization(
    target: Characterization,
    level: int,
    *,
    spatial_shape: tuple[int, ...] | None = None,
) -> Characterization:
    """Select descriptor levels for one coarse-to-fine reconstruction stage.

    Directionally merged 2-D characterizations store three descriptor rows
    (x/y/z).  Their multigrid level offsets must therefore be applied to each
    row independently instead of flattening the directional axis away.
    """

    if level < 0:
        raise ValueError("level must be non-negative")
    if level == 0 and spatial_shape is None:
        return target

    descriptors = {}
    for name, result in target.descriptors.items():
        level_shapes = tuple(
            tuple(map(int, shape)) for shape in result.metadata.get("level_shapes", ())
        )
        if not level_shapes:
            if level != 0:
                raise ValueError(f"descriptor {name} does not contain multigrid levels")
            descriptors[name] = result
            continue
        if len(level_shapes) <= level:
            raise ValueError(
                f"descriptor {name} has only {len(level_shapes)} multigrid levels, "
                f"cannot select level {level}"
            )
        sizes = [int(np.prod(shape)) for shape in level_shapes]
        offset = sum(sizes[:level])
        values = np.asarray(result.values, dtype=np.float64)
        directional = bool(result.metadata.get("directional_merge"))
        if directional:
            if values.ndim < 2 or values.shape[0] != 3:
                raise ValueError(
                    f"directional descriptor {name} must have leading x/y/z axis"
                )
            rows = values.reshape(3, -1)[:, offset:]
            selected = rows
        else:
            selected = values.reshape(-1)[offset:]
        descriptors[name] = type(result)(
            name=result.name,
            values=selected,
            differentiable=result.differentiable,
            metadata={
                **result.metadata,
                "level_shapes": level_shapes[level:],
                "coarsened_from_level": level,
            },
        )

    if spatial_shape is None:
        coarse_shape = tuple(
            max(2, int(size // (2**level))) for size in target.spatial_shape
        )
    else:
        coarse_shape = tuple(map(int, spatial_shape))
    return Characterization(
        descriptors=descriptors,
        settings=replace(target.settings, multigrid_levels=None),
        phase_ids=target.phase_ids,
        spatial_shape=coarse_shape,
        metadata={**target.metadata, "coarsened_from_level": level},
    )


def _upsample_labels(labels: np.ndarray, desired_shape: tuple[int, ...]) -> np.ndarray:
    result = labels
    for axis in range(labels.ndim):
        repeat = max(1, int(math.ceil(desired_shape[axis] / result.shape[axis])))
        result = np.repeat(result, repeat, axis=axis)
    slices = tuple(slice(0, size) for size in desired_shape)
    return result[slices]


def _multigrid_reconstruct(
    target: Characterization,
    shape: tuple[int, ...],
    settings: ReconstructionSettings,
    initial_microstructure: Any | None,
) -> ReconstructionResult:
    directional_target = bool(target.metadata.get("directional_merge"))
    if not directional_target and tuple(shape) != tuple(target.spatial_shape):
        raise ValueError(
            "multigrid reconstruction requires desired_shape to equal the characterization "
            "shape unless the target was created by merge_directional()"
        )
    if directional_target and len(shape) != 3:
        raise ValueError("directionally merged characterization requires a 3-D desired_shape")
    level_counts = [
        len(result.metadata.get("level_shapes", ()))
        for result in target.descriptors.values()
    ]
    usable_levels = min(level_counts) if level_counts else 1
    if usable_levels <= 1:
        return _gradient_reconstruct(
            target,
            shape,
            replace(settings, use_multigrid_reconstruction=False),
            initial_microstructure,
        )
    previous = initial_microstructure
    combined_records: list[ConvergenceRecord] = []
    last_result: ReconstructionResult | None = None
    for level in reversed(range(usable_levels)):
        coarse_shape = tuple(max(2, int(size // (2**level))) for size in shape)
        coarse_target = _coarsen_characterization(
            target, level, spatial_shape=coarse_shape
        )
        if previous is not None:
            if hasattr(previous, "labels"):
                previous = previous.labels
            previous_array = np.asarray(previous)
            if previous_array.shape != coarse_shape:
                # When walking coarse -> fine, nearest-neighbor expansion is the
                # correct phase-preserving initialization.
                previous = _upsample_labels(previous_array, coarse_shape)
        level_settings = replace(
            settings,
            use_multigrid_reconstruction=False,
            # The coarsened target contains only this level and all finer
            # descriptor levels.  Compute exactly the same number of levels
            # from the current reconstruction grid.
            multigrid_levels=max(1, usable_levels - level),
            max_iter=max(3, int(math.ceil(settings.max_iter / usable_levels))),
        )
        last_result = _gradient_reconstruct(
            coarse_target,
            coarse_shape,
            level_settings,
            previous,
        )
        combined_records.extend(
            ConvergenceRecord(
                iteration=len(combined_records) + record.iteration,
                loss=record.loss,
                descriptor_losses=record.descriptor_losses,
                temperature=record.temperature,
                accepted=record.accepted,
            )
            for record in last_result.convergence
        )
        previous = last_result.labels
    assert last_result is not None
    return ReconstructionResult(
        labels=last_result.labels,
        probabilities=last_result.probabilities,
        phase_ids=last_result.phase_ids,
        convergence=tuple(combined_records),
        settings=settings,
        metadata={**last_result.metadata, "multigrid_reconstruction_levels": usable_levels},
    )

def reconstruct_microstructure(
    characterization: Characterization,
    desired_shape: tuple[int, ...] | None = None,
    *,
    settings: ReconstructionSettings | None = None,
    initial_microstructure: Any | None = None,
) -> ReconstructionResult:
    settings = ReconstructionSettings(
        descriptor_types=characterization.descriptor_types,
        phase_ids=characterization.phase_ids,
    ) if settings is None else settings
    shape = characterization.spatial_shape if desired_shape is None else tuple(map(int, desired_shape))
    if len(shape) not in {2, 3} or any(value < 2 for value in shape):
        raise ValueError("desired_shape must be a 2-D or 3-D shape with entries >= 2")
    definition = optimizer_definition(settings.optimizer_type)
    if settings.use_multigrid_reconstruction and definition.family in {"torch", "scipy"}:
        if not settings.use_multigrid_descriptors:
            raise ValueError("multigrid reconstruction requires multigrid descriptors")
        return _multigrid_reconstruct(
            characterization, shape, settings, initial_microstructure
        )
    if definition.family == "annealing":
        return _annealing_reconstruct(characterization, shape, settings, initial_microstructure)
    if definition.family == "postprocess":
        base_settings = replace(settings, optimizer_type="LBFGSB")
        result = _gradient_reconstruct(characterization, shape, base_settings, initial_microstructure)
        if settings.postprocess_annealing_steps <= 0:
            return result
        anneal = replace(
            settings,
            optimizer_type="SimulatedAnnealing",
            max_iter=settings.postprocess_annealing_steps,
        )
        return _annealing_reconstruct(characterization, shape, anneal, result.labels)
    return _gradient_reconstruct(characterization, shape, settings, initial_microstructure)


def match_microstructure(
    microstructure: Any,
    *,
    desired_shape: tuple[int, ...] | None = None,
    characterization_settings: CharacterizationSettings | None = None,
    reconstruction_settings: ReconstructionSettings | None = None,
) -> tuple[Characterization, ReconstructionResult]:
    characterization = characterize_microstructure(microstructure, characterization_settings)
    if reconstruction_settings is None:
        reconstruction_settings = ReconstructionSettings(
            descriptor_types=characterization.descriptor_types,
            phase_ids=characterization.phase_ids,
            use_multigrid_descriptors=characterization.settings.use_multigrid_descriptors,
            multigrid_levels=characterization.settings.multigrid_levels,
            limit_to=characterization.settings.limit_to,
            use_multiphase=characterization.settings.use_multiphase,
            slice_mode=characterization.settings.slice_mode,
            isotropic=characterization.settings.isotropic,
        )
    return characterization, reconstruct_microstructure(
        characterization,
        desired_shape,
        settings=reconstruction_settings,
    )


reconstruct = reconstruct_microstructure
match = match_microstructure


__all__ = [
    "ConvergenceRecord",
    "ReconstructionResult",
    "match",
    "match_microstructure",
    "reconstruct",
    "reconstruct_microstructure",
]
