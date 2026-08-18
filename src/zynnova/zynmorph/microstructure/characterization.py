"""Characterization, descriptor algebra, serialization, and visualization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .descriptors import DescriptorResult, compute_descriptor_numpy_spatial, phase_probabilities
from .settings import CharacterizationSettings


@dataclass(frozen=True, slots=True)
class Characterization:
    descriptors: Mapping[str, DescriptorResult]
    settings: CharacterizationSettings
    phase_ids: tuple[int, ...]
    spatial_shape: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> DescriptorResult:
        return self.descriptors[key]

    @property
    def descriptor_types(self) -> tuple[str, ...]:
        return tuple(self.descriptors)



def characterize_microstructure(
    microstructure: Any,
    settings: CharacterizationSettings | None = None,
) -> Characterization:
    settings = CharacterizationSettings() if settings is None else settings
    prob, phases = phase_probabilities(microstructure, phase_ids=settings.phase_ids)
    descriptor_prob = prob
    single_phase_index = None
    if not settings.use_multiphase:
        if prob.shape[0] != 2:
            raise ValueError("use_multiphase=False is supported only for binary microstructures")
        single_phase_index = 1
        descriptor_prob = prob[single_phase_index : single_phase_index + 1]
    descriptors: dict[str, DescriptorResult] = {}
    for name in settings.descriptor_types:
        result = compute_descriptor_numpy_spatial(
            name,
            descriptor_prob,
            slice_mode=settings.slice_mode,
            isotropic=settings.isotropic,
            rng=np.random.default_rng(0),
            limit_to=settings.limit_to,
            periodic=settings.periodic,
            use_multigrid=settings.use_multigrid_descriptors,
            multigrid_levels=settings.multigrid_levels,
            kwargs=settings.descriptor_kwargs.get(name, {}),
        )
        descriptors[result.name] = result
    return Characterization(
        descriptors=descriptors,
        settings=replace(settings, phase_ids=phases),
        phase_ids=phases,
        spatial_shape=tuple(map(int, prob.shape[1:])),
        metadata={
            "schema": "zynnova.microstructure-characterization.v1",
            "volume_fractions": prob.mean(axis=tuple(range(1, prob.ndim))).tolist(),
            "single_phase_index": single_phase_index,
            "single_phase_id": None if single_phase_index is None else int(phases[single_phase_index]),
        },
    )


def merge_characterizations(
    characterizations: Sequence[Characterization],
    *,
    weights: Sequence[float] | None = None,
) -> Characterization:
    if not characterizations:
        raise ValueError("at least one characterization is required")
    first = characterizations[0]
    for item in characterizations[1:]:
        if item.phase_ids != first.phase_ids:
            raise ValueError("cannot merge different phase sets")
        if item.descriptor_types != first.descriptor_types:
            raise ValueError("cannot merge different descriptor sets")
    if weights is None:
        normalized = np.full(len(characterizations), 1.0 / len(characterizations))
    else:
        normalized = np.asarray(weights, dtype=np.float64)
        if normalized.shape != (len(characterizations),) or np.any(normalized < 0):
            raise ValueError("weights must be non-negative and match input count")
        if normalized.sum() <= 0:
            raise ValueError("weights must have positive sum")
        normalized /= normalized.sum()
    descriptors: dict[str, DescriptorResult] = {}
    for name in first.descriptor_types:
        values = sum(
            weight * np.asarray(item.descriptors[name].values, dtype=np.float64)
            for weight, item in zip(normalized, characterizations, strict=True)
        )
        descriptors[name] = DescriptorResult(
            name=name,
            values=np.asarray(values),
            differentiable=first.descriptors[name].differentiable,
            metadata={**first.descriptors[name].metadata, "merged": True},
        )
    return Characterization(
        descriptors=descriptors,
        settings=first.settings,
        phase_ids=first.phase_ids,
        spatial_shape=first.spatial_shape,
        metadata={"schema": "zynnova.microstructure-characterization.v1", "merged": len(characterizations)},
    )



def merge_directional_characterizations(
    characterizations: Sequence[Characterization],
) -> Characterization:
    """Merge 2-D characterizations into anisotropic x/y/z 3-D descriptors.

    One input is replicated to all three orientations.  Two inputs follow the
    MCR convention ``(first, first, second)``.  Three inputs map directly to
    x/y/z.  Unlike :func:`merge_characterizations`, this operation does not
    average descriptor values; it preserves directional information.
    """

    if len(characterizations) not in {1, 2, 3}:
        raise ValueError("directional merge requires one, two, or three characterizations")
    if len(characterizations) == 1:
        directional = (characterizations[0],) * 3
    elif len(characterizations) == 2:
        directional = (characterizations[0], characterizations[0], characterizations[1])
    else:
        directional = tuple(characterizations)

    first = directional[0]
    for item in directional[1:]:
        if item.phase_ids != first.phase_ids:
            raise ValueError("cannot directionally merge different phase sets")
        if item.descriptor_types != first.descriptor_types:
            raise ValueError("cannot directionally merge different descriptor sets")

    descriptors: dict[str, DescriptorResult] = {}
    for name in first.descriptor_types:
        values = [np.asarray(item.descriptors[name].values, dtype=np.float64) for item in directional]
        if any(value.shape != values[0].shape for value in values[1:]):
            raise ValueError(f"descriptor {name} shapes differ across directions")
        descriptors[name] = DescriptorResult(
            name=name,
            values=np.stack(values, axis=0),
            differentiable=first.descriptors[name].differentiable,
            metadata={
                **first.descriptors[name].metadata,
                "directional": True,
                "directional_merge": True,
                "direction_count": 3,
            },
        )

    # Directional characterizations are targets for a 3-D reconstruction.  A
    # specific 3-D shape is intentionally not invented here; callers provide it
    # to reconstruct().  Store the 2-D source shape for provenance.
    return Characterization(
        descriptors=descriptors,
        settings=replace(first.settings, slice_mode="average", isotropic=False),
        phase_ids=first.phase_ids,
        spatial_shape=first.spatial_shape,
        metadata={
            "schema": "zynnova.microstructure-characterization.v1",
            "directional_merge": True,
            "source_spatial_shape": first.spatial_shape,
            "volume_fractions": first.metadata.get("volume_fractions"),
        },
    )

def interpolate_characterizations(
    start: Characterization,
    stop: Characterization,
    count: int,
    *,
    include_endpoints: bool = True,
) -> tuple[Characterization, ...]:
    if count < 1:
        raise ValueError("count must be >= 1")
    if start.phase_ids != stop.phase_ids or start.descriptor_types != stop.descriptor_types:
        raise ValueError("characterizations are incompatible")
    if count == 1:
        alphas = np.asarray([0.0 if include_endpoints else 0.5])
    else:
        alphas = np.linspace(0.0, 1.0, count + (0 if include_endpoints else 2))
        if not include_endpoints:
            alphas = alphas[1:-1]
    outputs = []
    for alpha in alphas:
        outputs.append(
            merge_characterizations((start, stop), weights=(1.0 - float(alpha), float(alpha)))
        )
    return tuple(outputs)


def save_characterization(path: str | Path, characterization: Characterization) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".npz":
        target = target.with_suffix(".npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"descriptor__{name}": np.asarray(result.values)
        for name, result in characterization.descriptors.items()
    }
    settings_dict = asdict(characterization.settings)
    metadata = {
        "settings": settings_dict,
        "phase_ids": characterization.phase_ids,
        "spatial_shape": characterization.spatial_shape,
        "metadata": dict(characterization.metadata),
        "descriptor_metadata": {
            name: dict(result.metadata) for name, result in characterization.descriptors.items()
        },
        "descriptor_differentiable": {
            name: bool(result.differentiable) for name, result in characterization.descriptors.items()
        },
    }
    np.savez_compressed(
        target,
        **arrays,
        __metadata_json__=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return target


def load_characterization(path: str | Path) -> Characterization:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(str(data["__metadata_json__"]))
        settings_payload = dict(metadata["settings"])
        settings_payload["descriptor_types"] = tuple(settings_payload["descriptor_types"])
        if settings_payload.get("descriptor_weights") is not None:
            settings_payload["descriptor_weights"] = tuple(settings_payload["descriptor_weights"])
        if settings_payload.get("phase_ids") is not None:
            settings_payload["phase_ids"] = tuple(settings_payload["phase_ids"])
        settings = CharacterizationSettings(**settings_payload)
        descriptors = {}
        for key in data.files:
            if not key.startswith("descriptor__"):
                continue
            name = key[len("descriptor__"):]
            descriptors[name] = DescriptorResult(
                name=name,
                values=np.asarray(data[key]),
                differentiable=bool(metadata["descriptor_differentiable"][name]),
                metadata=metadata["descriptor_metadata"].get(name, {}),
            )
    return Characterization(
        descriptors=descriptors,
        settings=settings,
        phase_ids=tuple(map(int, metadata["phase_ids"])),
        spatial_shape=tuple(map(int, metadata["spatial_shape"])),
        metadata=metadata.get("metadata", {}),
    )


def view_characterization(
    characterization: Characterization,
    *,
    descriptor: str | None = None,
):
    import matplotlib.pyplot as plt

    names = [descriptor] if descriptor is not None else list(characterization.descriptor_types)
    fig, axes = plt.subplots(len(names), 1, figsize=(8, max(3, 3 * len(names))))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names, strict=True):
        values = np.asarray(characterization.descriptors[name].values)
        if values.ndim == 1:
            ax.plot(values)
        elif values.ndim == 2:
            image = ax.imshow(values, aspect="auto")
            fig.colorbar(image, ax=ax)
        else:
            flat = values.reshape(values.shape[0], -1) if values.shape[0] < 64 else values.reshape(1, -1)
            image = ax.imshow(flat, aspect="auto")
            fig.colorbar(image, ax=ax)
        ax.set_title(name)
    fig.tight_layout()
    return fig


# Short names intentionally mirror the conceptual workflow, but stay inside
# ZynNova's namespace.
characterize = characterize_microstructure
merge = merge_characterizations
merge_directional = merge_directional_characterizations
interpolate = interpolate_characterizations


__all__ = [
    "Characterization",
    "characterize",
    "characterize_microstructure",
    "interpolate",
    "interpolate_characterizations",
    "load_characterization",
    "merge",
    "merge_characterizations",
    "merge_directional",
    "merge_directional_characterizations",
    "save_characterization",
    "view_characterization",
]
