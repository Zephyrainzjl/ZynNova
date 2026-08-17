from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ....structure import StructureData
from ...common import load_checkpoint, resolve_device
from .config import (
    QM9GeneratorModelConfig,
    QM9GeneratorSamplingConfig,
    QM9_PROPERTY_UNITS,
)
from .data import center_coordinates
from .model import QM9ConditionalGenerator
from .normalizer import QM9PropertyNormalizer
from .validation import GeometryReport, analyze_generated_structure


_ELEMENT_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
_Z_TO_ELEMENT = {value: key for key, value in _ELEMENT_TO_Z.items()}


@dataclass(slots=True)
class GeneratedMolecule:
    structure: StructureData
    requested_properties: dict[str, float]
    predicted_properties: dict[str, float]
    property_errors: dict[str, float]
    normalized_condition_error: float
    geometry: GeometryReport
    score: float

    def metadata(self) -> dict[str, Any]:
        return {
            "requested_properties": dict(self.requested_properties),
            "predicted_properties": dict(self.predicted_properties),
            "property_errors": dict(self.property_errors),
            "normalized_condition_error": self.normalized_condition_error,
            "geometry": self.geometry.to_dict(),
            "score": self.score,
        }


@dataclass(slots=True)
class QM9GenerationResult:
    composition: dict[str, int]
    property_units: dict[str, str]
    candidates: list[GeneratedMolecule]

    @property
    def best(self) -> GeneratedMolecule:
        if not self.candidates:
            raise ValueError("generation result contains no candidates")
        return self.candidates[0]

    @property
    def structures(self) -> list[StructureData]:
        return [candidate.structure for candidate in self.candidates]


def composition_to_atomic_numbers(
    composition: str | Mapping[str, int] | Iterable[int],
) -> list[int]:
    if isinstance(composition, str):
        matches = list(re.finditer(r"([A-Z][a-z]?)([0-9]*)", composition))
        if not matches or "".join(match.group(0) for match in matches) != composition:
            raise ValueError(f"invalid molecular formula: {composition!r}")
        parsed: dict[str, int] = {}
        for match in matches:
            symbol = match.group(1)
            count = int(match.group(2) or "1")
            parsed[symbol] = parsed.get(symbol, 0) + count
        composition = parsed
    if isinstance(composition, Mapping):
        unknown = sorted(set(composition) - set(_ELEMENT_TO_Z))
        if unknown:
            raise ValueError(
                f"QM9 supports H, C, N, O and F; unsupported elements={unknown}"
            )
        numbers: list[int] = []
        # Heavy atoms first gives deterministic padding and matches common QM9 order.
        for symbol in ("C", "N", "O", "F", "H"):
            count = int(composition.get(symbol, 0))
            if count < 0:
                raise ValueError(f"negative element count: {symbol}={count}")
            numbers.extend([_ELEMENT_TO_Z[symbol]] * count)
    else:
        numbers = [int(value) for value in composition]
    if not numbers:
        raise ValueError("composition cannot be empty")
    invalid = sorted(set(numbers) - set(_Z_TO_ELEMENT))
    if invalid:
        raise ValueError(
            f"QM9 supports atomic numbers {sorted(_Z_TO_ELEMENT)}; got {invalid}"
        )
    return numbers


def _composition_mapping(numbers: list[int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for number in numbers:
        symbol = _Z_TO_ELEMENT[number]
        result[symbol] = result.get(symbol, 0) + 1
    return result


def _condition_velocity(
    model,
    z,
    positions,
    time,
    mask,
    properties,
    property_mask,
    guidance_scale: float,
):
    import torch

    with torch.no_grad():
        conditional = model(
            z,
            positions,
            time,
            mask,
            properties,
            property_mask,
        )
        if guidance_scale == 1.0 or not bool(property_mask.any().item()):
            return conditional
        unconditional_mask = torch.zeros_like(property_mask)
        unconditional = model(
            z,
            positions,
            time,
            mask,
            torch.zeros_like(properties),
            unconditional_mask,
        )
    return unconditional + guidance_scale * (conditional - unconditional)


def _property_gradient_velocity(
    model,
    z,
    positions,
    mask,
    target,
    property_mask,
    *,
    scale: float,
    clip: float | None,
):
    import torch

    if scale == 0.0 or not bool(property_mask.any().item()):
        return torch.zeros_like(positions)
    working = positions.detach().requires_grad_(True)
    prediction = model.predict_properties(z, working, mask)
    squared = (prediction - target).square() * property_mask.to(prediction.dtype)
    loss = squared.sum(dim=-1) / property_mask.sum(dim=-1).clamp_min(1)
    gradient = torch.autograd.grad(loss.sum(), working)[0]
    gradient = center_coordinates(gradient, mask)
    if clip is not None:
        norm = torch.linalg.vector_norm(gradient.reshape(gradient.shape[0], -1), dim=-1)
        factor = (clip / norm.clamp_min(clip)).clamp_max(1.0)
        gradient = gradient * factor[:, None, None]
    return -scale * gradient.detach()


def _collision_velocity(positions, mask, *, minimum_distance: float, scale: float):
    import torch

    if scale == 0.0:
        return torch.zeros_like(positions)
    displacement = positions[:, :, None, :] - positions[:, None, :, :]
    distance = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(1.0e-6)
    pair_mask = mask[:, :, None] & mask[:, None, :]
    eye = torch.eye(
        positions.shape[1],
        device=positions.device,
        dtype=torch.bool,
    )[None, :, :]
    pair_mask = pair_mask & (~eye)
    overlap = (minimum_distance - distance).clamp_min(0.0) * pair_mask
    direction = displacement / distance[..., None]
    velocity = (direction * overlap[..., None]).sum(dim=2)
    return center_coordinates(scale * velocity, mask)


def _guided_velocity(
    model,
    z,
    positions,
    time,
    mask,
    properties,
    property_mask,
    config: QM9GeneratorSamplingConfig,
):
    base = _condition_velocity(
        model,
        z,
        positions,
        time,
        mask,
        properties,
        property_mask,
        config.guidance_scale,
    )
    property_velocity = _property_gradient_velocity(
        model,
        z,
        positions,
        mask,
        properties,
        property_mask,
        scale=config.property_guidance_scale,
        clip=config.property_gradient_clip,
    )
    collision_velocity = _collision_velocity(
        positions,
        mask,
        minimum_distance=config.minimum_distance_A,
        scale=config.collision_guidance_scale,
    )
    return center_coordinates(base + property_velocity + collision_velocity, mask)


def generate_qm9_candidates(
    model: QM9ConditionalGenerator,
    *,
    composition: str | Mapping[str, int] | Iterable[int],
    properties: Mapping[str, float],
    sampling: QM9GeneratorSamplingConfig | None = None,
    normalizer: QM9PropertyNormalizer | None = None,
) -> QM9GenerationResult:
    import torch

    sampling = sampling or QM9GeneratorSamplingConfig()
    sampling.__post_init__()
    resolved = resolve_device(sampling.device)
    normalizer = normalizer or model.property_normalizer
    if normalizer is None:
        raise ValueError(
            "property normalizer is required; load a trained checkpoint or pass normalizer="
        )
    if tuple(normalizer.names) != tuple(model.config.property_names):
        raise ValueError("normalizer property order does not match the model")
    atomic_numbers = composition_to_atomic_numbers(composition)
    if len(atomic_numbers) > model.config.max_atoms:
        raise ValueError(
            f"composition has {len(atomic_numbers)} atoms, "
            f"max_atoms={model.config.max_atoms}"
        )
    encoded, encoded_mask = normalizer.encode_mapping(properties)
    batch_size = sampling.num_candidates
    max_atoms = model.config.max_atoms
    dtype = next(model.parameters()).dtype
    z = torch.zeros((batch_size, max_atoms), device=resolved, dtype=torch.long)
    mask = torch.zeros((batch_size, max_atoms), device=resolved, dtype=torch.bool)
    z[:, : len(atomic_numbers)] = torch.as_tensor(
        atomic_numbers,
        device=resolved,
        dtype=torch.long,
    )[None, :]
    mask[:, : len(atomic_numbers)] = True
    property_tensor = torch.as_tensor(encoded, device=resolved, dtype=dtype)[None, :]
    property_tensor = property_tensor.expand(batch_size, -1).clone()
    property_mask = torch.as_tensor(encoded_mask, device=resolved, dtype=torch.bool)[None, :]
    property_mask = property_mask.expand(batch_size, -1).clone()

    generator = None
    if sampling.seed is not None:
        generator = torch.Generator(device=resolved).manual_seed(sampling.seed)
    positions = torch.randn(
        (batch_size, max_atoms, 3),
        device=resolved,
        dtype=dtype,
        generator=generator,
    ) * sampling.noise_scale_A
    positions = center_coordinates(positions, mask)
    model = model.to(resolved).eval()
    dt = 1.0 / sampling.steps

    for step in range(sampling.steps):
        time = torch.full(
            (batch_size,),
            step / sampling.steps,
            device=resolved,
            dtype=dtype,
        )
        velocity = _guided_velocity(
            model,
            z,
            positions,
            time,
            mask,
            property_tensor,
            property_mask,
            sampling,
        )
        if sampling.solver == "euler":
            positions = positions + dt * velocity
        else:
            predictor = center_coordinates(positions + dt * velocity, mask)
            next_time = torch.full(
                (batch_size,),
                min((step + 1) / sampling.steps, 1.0),
                device=resolved,
                dtype=dtype,
            )
            corrected = _guided_velocity(
                model,
                z,
                predictor,
                next_time,
                mask,
                property_tensor,
                property_mask,
                sampling,
            )
            positions = positions + 0.5 * dt * (velocity + corrected)
        positions = center_coordinates(positions.detach(), mask)

    with torch.no_grad():
        predicted_normalized = model.predict_properties(z, positions, mask)
    candidates: list[GeneratedMolecule] = []
    requested = {name: float(value) for name, value in properties.items()}
    for candidate_index in range(batch_size):
        candidate_positions = positions[
            candidate_index,
            : len(atomic_numbers),
        ].cpu().numpy()
        predicted = normalizer.decode_mapping(predicted_normalized[candidate_index].cpu().numpy())
        errors = {
            name: abs(predicted[name] - requested[name])
            for name in requested
        }
        requested_indices = [normalizer.names.index(name) for name in requested]
        if requested_indices:
            normalized_error = float(
                np.mean(
                    np.abs(
                        predicted_normalized[candidate_index, requested_indices]
                        .cpu()
                        .numpy()
                        - encoded[requested_indices]
                    )
                )
            )
        else:
            normalized_error = 0.0
        structure = StructureData(
            atomic_numbers=np.asarray(atomic_numbers, dtype=np.int64),
            positions=np.asarray(candidate_positions, dtype=np.float64),
            pbc=np.zeros(3, dtype=bool),
            cell=np.zeros((3, 3), dtype=np.float64),
            info={
                "generator": "zynnova.ml.generation.qm9_generator",
                "requested_properties": requested,
                "predicted_properties": predicted,
            },
        )
        geometry, bonds = analyze_generated_structure(
            structure,
            minimum_distance_A=sampling.minimum_distance_A,
            bond_scale=sampling.bond_scale,
        )
        if len(bonds):
            structure.bonds = bonds
            structure.bond_orders = np.ones(len(bonds), dtype=np.float64)
        geometry_penalty = (
            5.0 * geometry.collision_count
            + (0.0 if geometry.connected else 2.0 * max(geometry.component_count - 1, 1))
            + (0.0 if geometry.approximate_valence_valid else 2.0)
        )
        score = normalized_error + geometry_penalty
        candidates.append(
            GeneratedMolecule(
                structure=structure,
                requested_properties=requested,
                predicted_properties=predicted,
                property_errors=errors,
                normalized_condition_error=normalized_error,
                geometry=geometry,
                score=float(score),
            )
        )
    candidates.sort(key=lambda candidate: candidate.score)
    return QM9GenerationResult(
        composition=_composition_mapping(atomic_numbers),
        property_units={
            name: QM9_PROPERTY_UNITS[name]
            for name in model.config.property_names
        },
        candidates=candidates,
    )


def generate_qm9_molecule(
    model: QM9ConditionalGenerator,
    *,
    composition: str | Mapping[str, int] | Iterable[int],
    properties: Mapping[str, float],
    sampling: QM9GeneratorSamplingConfig | None = None,
    normalizer: QM9PropertyNormalizer | None = None,
) -> StructureData:
    return generate_qm9_candidates(
        model,
        composition=composition,
        properties=properties,
        sampling=sampling,
        normalizer=normalizer,
    ).best.structure


def save_qm9_generation(
    result: QM9GenerationResult,
    directory: str | Path,
    *,
    prefix: str = "qm9-generator",
) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    summary = {
        "composition": result.composition,
        "property_units": result.property_units,
        "candidates": [],
    }
    for index, candidate in enumerate(result.candidates):
        stem = f"{prefix}-{index:05d}"
        xyz_path = directory / f"{stem}.xyz"
        try:
            from ase.io import write

            write(xyz_path, candidate.structure.to_ase())
            structure_path = xyz_path
        except ImportError:
            structure_path = xyz_path.with_suffix(".npz")
            np.savez_compressed(
                structure_path,
                atomic_numbers=candidate.structure.atomic_numbers,
                positions=candidate.structure.positions,
                bonds=candidate.structure.bonds,
            )
        metadata_path = directory / f"{stem}.json"
        metadata_path.write_text(
            json.dumps(candidate.metadata(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        paths.extend((structure_path, metadata_path))
        summary["candidates"].append(
            {
                "rank": index,
                "structure": structure_path.name,
                "metadata": metadata_path.name,
                **candidate.metadata(),
            }
        )
    summary_path = directory / f"{prefix}-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths.append(summary_path)
    return paths


def load_qm9_generator(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> QM9ConditionalGenerator:
    payload = load_checkpoint(checkpoint, map_location=device)
    model = QM9ConditionalGenerator(
        QM9GeneratorModelConfig(**payload["model_config"])
    )
    model.load_state_dict(payload["model_state"])
    model.property_normalizer = QM9PropertyNormalizer.from_state_dict(
        payload["property_normalizer"]
    )
    model.to(resolve_device(device)).eval()
    return model


__all__ = [
    "GeneratedMolecule",
    "QM9GenerationResult",
    "composition_to_atomic_numbers",
    "generate_qm9_candidates",
    "generate_qm9_molecule",
    "load_qm9_generator",
    "save_qm9_generation",
]
