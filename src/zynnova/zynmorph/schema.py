"""Physical phase schema and generation conditions for ZynMorph."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Mapping

import numpy as np

from ..core.exceptions import ConfigurationError
from ..geometry.voxel import normalize_spacing


class BatteryPhase(IntEnum):
    """Canonical multi-material labels, compatible with battery FEM workflows."""

    SEPARATOR_ELECTROLYTE = 0
    POSITIVE_ACTIVE = 1
    POSITIVE_ELECTROLYTE = 2
    NEGATIVE_ACTIVE = 3
    NEGATIVE_ELECTROLYTE = 4
    POSITIVE_CBD = 5
    NEGATIVE_CBD = 6
    POSITIVE_CEI = 7
    NEGATIVE_SEI = 8
    CRACK = 9
    NEGATIVE_CURRENT_COLLECTOR = 10
    POSITIVE_CURRENT_COLLECTOR = 11


DEFAULT_PHASE_NAMES: Mapping[int, str] = {
    int(phase): phase.name.lower() for phase in BatteryPhase
}


@dataclass(frozen=True, slots=True)
class MicrostructureCondition:
    """Serializable condition vector for generation, reconstruction, and inverse design."""

    shape: tuple[int, int, int]
    phase_fractions: Mapping[int, float]
    voxel_size_m: float | tuple[float, float, float] = 1.0e-7
    correlation_lengths_voxels: Mapping[int, float | tuple[float, float, float]] = field(
        default_factory=dict
    )
    interface_affinity: Mapping[tuple[int, int], float] = field(default_factory=dict)
    percolation_axes: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    manufacturing: Mapping[str, float] = field(default_factory=dict)
    descriptor_targets: Mapping[str, float] = field(default_factory=dict)
    periodic: tuple[bool, bool, bool] = (False, False, False)
    seed: int = 42

    def __post_init__(self) -> None:
        shape = tuple(int(item) for item in self.shape)
        if len(shape) != 3 or min(shape) < 2:
            raise ConfigurationError("shape must contain three integers of at least two")
        fractions = {int(key): float(value) for key, value in self.phase_fractions.items()}
        if not fractions:
            raise ConfigurationError("phase_fractions cannot be empty")
        if any(value < 0.0 or not np.isfinite(value) for value in fractions.values()):
            raise ConfigurationError("phase fractions must be finite and non-negative")
        total = sum(fractions.values())
        if total <= 0.0:
            raise ConfigurationError("phase fractions must have a positive sum")
        fractions = {key: value / total for key, value in fractions.items() if value > 0.0}
        valid = {int(phase) for phase in BatteryPhase}
        unknown = sorted(set(fractions) - valid)
        if unknown:
            raise ConfigurationError(f"unknown battery phase ids: {unknown}")
        spacing = normalize_spacing(self.voxel_size_m)
        periodic = tuple(bool(item) for item in self.periodic)
        if len(periodic) != 3:
            raise ConfigurationError("periodic must contain exactly three booleans")
        lengths: dict[int, tuple[float, float, float]] = {}
        for phase, value in self.correlation_lengths_voxels.items():
            if np.isscalar(value):
                resolved = (float(value),) * 3
            else:
                resolved = tuple(float(item) for item in value)
            if len(resolved) != 3 or min(resolved) <= 0.0:
                raise ConfigurationError("correlation lengths must be three positive values")
            lengths[int(phase)] = resolved
        affinity: dict[tuple[int, int], float] = {}
        for pair, value in self.interface_affinity.items():
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ConfigurationError("interface affinity keys must contain two distinct phases")
            affinity[tuple(sorted((int(pair[0]), int(pair[1]))))] = float(value)
        axes: dict[int, tuple[int, ...]] = {}
        exact_counts = _exact_counts(shape, fractions)
        for phase, values in self.percolation_axes.items():
            phase_id = int(phase)
            resolved_axes = tuple(sorted({int(axis) for axis in values}))
            if any(axis not in {0, 1, 2} for axis in resolved_axes):
                raise ConfigurationError("percolation axes must be selected from 0, 1, and 2")
            if phase_id not in fractions:
                raise ConfigurationError(
                    f"percolation phase {phase_id} is absent from phase_fractions"
                )
            minimum_path_voxels = sum(shape[axis] for axis in resolved_axes)
            if resolved_axes:
                minimum_path_voxels -= len(resolved_axes) - 1
            if exact_counts[phase_id] < minimum_path_voxels:
                raise ConfigurationError(
                    f"phase {phase_id} has {exact_counts[phase_id]} voxels but at least "
                    f"{minimum_path_voxels} are needed for requested percolation axes"
                )
            axes[phase_id] = resolved_axes
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "voxel_size_m", spacing)
        object.__setattr__(self, "phase_fractions", fractions)
        object.__setattr__(self, "periodic", periodic)
        object.__setattr__(self, "correlation_lengths_voxels", lengths)
        object.__setattr__(self, "interface_affinity", affinity)
        object.__setattr__(self, "percolation_axes", axes)
        object.__setattr__(
            self,
            "manufacturing",
            {str(key): float(value) for key, value in self.manufacturing.items()},
        )
        object.__setattr__(
            self,
            "descriptor_targets",
            {str(key): float(value) for key, value in self.descriptor_targets.items()},
        )

    @property
    def phases(self) -> tuple[int, ...]:
        return tuple(sorted(self.phase_fractions))

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape))

    def exact_phase_counts(self) -> Mapping[int, int]:
        """Largest-remainder apportionment with an exact total voxel count."""

        return _exact_counts(self.shape, self.phase_fractions)


def _exact_counts(
    shape: tuple[int, int, int],
    fractions: Mapping[int, float],
) -> Mapping[int, int]:
    phases = tuple(sorted(fractions))
    n_voxels = int(np.prod(shape))
    raw = np.asarray([fractions[phase] * n_voxels for phase in phases])
    counts = np.floor(raw).astype(np.int64)
    remaining = n_voxels - int(counts.sum())
    if remaining:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remaining]] += 1
    return {phase: int(count) for phase, count in zip(phases, counts, strict=True)}


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    backend: str = "auto"
    refinement_steps: int = 0
    temperature: float = 0.15
    preserve_exact_fractions: bool = True
    output_directory: str = "zynnova_runs/zynmorph"
    export_volume_formats: tuple[str, ...] = ("npz",)
    export_mesh_formats: tuple[str, ...] = ("vtk", "msh", "inp")
    mesh_backend: str = "tetgen"
    tetgen_options: Mapping[str, object] = field(default_factory=dict)
    maximum_tetrahedra: int = 12_000_000

    def __post_init__(self) -> None:
        if self.refinement_steps < 0:
            raise ConfigurationError("refinement_steps cannot be negative")
        if self.temperature <= 0.0:
            raise ConfigurationError("temperature must be positive")
        if self.maximum_tetrahedra < 6:
            raise ConfigurationError("maximum_tetrahedra must be at least six")
        mesh_backend = str(self.mesh_backend).strip().lower().replace("_", "-")
        mesh_backend = {
            "voxel": "structured",
            "six-tet": "structured",
            "adaptive": "tetgen",
            "tetgen-1.6": "tetgen",
        }.get(mesh_backend, mesh_backend)
        if mesh_backend not in {"structured", "tetgen"}:
            raise ConfigurationError("mesh_backend must be 'structured' or 'tetgen'")
        object.__setattr__(self, "mesh_backend", mesh_backend)
        object.__setattr__(self, "tetgen_options", dict(self.tetgen_options))
        volume_formats = tuple(str(item).strip().lower().lstrip(".") for item in self.export_volume_formats)
        unsupported_volume = sorted(set(volume_formats) - {"npz", "npy", "raw", "tif", "tiff"})
        if unsupported_volume:
            raise ConfigurationError(f"unsupported volume formats: {unsupported_volume}")
        if not volume_formats:
            raise ConfigurationError("at least one volume export format is required")
        object.__setattr__(self, "export_volume_formats", volume_formats)
        mesh_formats = tuple(
            "mphtxt" if str(item).strip().lower().lstrip(".") == "comsol"
            else str(item).strip().lower().lstrip(".")
            for item in self.export_mesh_formats
        )
        unsupported_mesh = sorted(
            set(mesh_formats) - {"vtk", "msh", "inp", "npz", "mphtxt"}
        )
        if unsupported_mesh:
            raise ConfigurationError(f"unsupported FEM mesh formats: {unsupported_mesh}")
        if not mesh_formats:
            raise ConfigurationError("at least one FEM mesh export format is required")
        object.__setattr__(
            self,
            "export_mesh_formats",
            tuple(dict.fromkeys(mesh_formats)),
        )


__all__ = [
    "BatteryPhase",
    "DEFAULT_PHASE_NAMES",
    "GenerationConfig",
    "MicrostructureCondition",
]
