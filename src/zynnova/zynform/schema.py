"""Configuration schema for image-conditioned high-fidelity 3D assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping

from ..core import ConfigurationError


class PhysicalScaleBasis(str, Enum):
    UNSPECIFIED = "unspecified"
    USER_DECLARED = "user-declared"
    KNOWN_DIMENSION = "known-dimension"
    CALIBRATION_TARGET = "calibration-target"
    SENSOR_METADATA = "sensor-metadata"


class FEMMethod(str, Enum):
    AUTO = "auto"
    TETGEN = "tetgen"
    GMSH = "gmsh"
    VOXEL = "voxel"


@dataclass(frozen=True, slots=True)
class ObjectRequest:
    image: Path
    prompt: str | None = None
    backend: str = "auto"
    device: str = "auto"
    model_id: str | None = None
    foreground_mask: Path | None = None
    physical_extent_m: float | None = None
    physical_scale_basis: PhysicalScaleBasis = PhysicalScaleBasis.UNSPECIFIED
    physical_scale_evidence: Path | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        image = Path(self.image)
        if not image.is_file():
            raise FileNotFoundError(image)
        mask = None if self.foreground_mask is None else Path(self.foreground_mask)
        if mask is not None and not mask.is_file():
            raise FileNotFoundError(mask)
        if self.physical_extent_m is not None and self.physical_extent_m <= 0.0:
            raise ConfigurationError("physical_extent_m must be positive")
        evidence = (
            None
            if self.physical_scale_evidence is None
            else Path(self.physical_scale_evidence)
        )
        if evidence is not None and not evidence.is_file():
            raise FileNotFoundError(evidence)
        basis = self.physical_scale_basis
        if self.physical_extent_m is not None and basis is PhysicalScaleBasis.UNSPECIFIED:
            basis = PhysicalScaleBasis.USER_DECLARED
        if (
            self.physical_extent_m is not None
            and basis in {
                PhysicalScaleBasis.KNOWN_DIMENSION,
                PhysicalScaleBasis.CALIBRATION_TARGET,
                PhysicalScaleBasis.SENSOR_METADATA,
            }
            and evidence is None
        ):
            raise ConfigurationError(
                f"physical scale basis {basis.value!r} requires physical_scale_evidence"
            )
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "foreground_mask", mask)
        object.__setattr__(self, "physical_scale_basis", basis)
        object.__setattr__(self, "physical_scale_evidence", evidence)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class FEMConfig:
    method: FEMMethod = FEMMethod.AUTO
    target_edge_length: float = 0.02
    voxel_pitch: float = 0.01
    minimum_dihedral_degrees: float = 8.0
    minimum_radius_edge_ratio: float = 1.4
    maximum_cells: int = 2_000_000
    optimization_level: int = 2
    minimum_mean_ratio: float = 0.0
    prefer_native_tetgen: bool = True
    require_watertight: bool = True
    region_id: int = 1
    region_name: str = "OBJECT"

    def __post_init__(self) -> None:
        if self.target_edge_length <= 0.0 or self.voxel_pitch <= 0.0:
            raise ConfigurationError("FEM edge length and voxel pitch must be positive")
        if not 0.0 <= self.minimum_dihedral_degrees < 90.0:
            raise ConfigurationError("minimum_dihedral_degrees must lie in [0,90)")
        if self.minimum_radius_edge_ratio <= 0.0:
            raise ConfigurationError("minimum_radius_edge_ratio must be positive")
        if self.maximum_cells < 1:
            raise ConfigurationError("maximum_cells must be positive")
        if not 0 <= self.optimization_level <= 10:
            raise ConfigurationError("optimization_level must lie in [0,10]")
        if not 0.0 <= self.minimum_mean_ratio <= 1.0:
            raise ConfigurationError("minimum_mean_ratio must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class ObjectConfig:
    output_directory: str = "zynnova_runs/zynform"
    export_formats: tuple[str, ...] = (
        "glb",
        "obj",
        "ply",
        "stl",
    )
    normalize_extent: float | None = 1.0
    clean_mesh: bool = True
    weld_tolerance: float = 1.0e-7
    generate_fem: bool = True
    fem: FEMConfig = field(default_factory=FEMConfig)
    fem_export_formats: tuple[str, ...] = ("msh", "vtk", "inp", "npz", "mphtxt")
    export_fem_repair_surface: bool = True
    repair_fem_surface: bool = True
    backend_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.normalize_extent is not None and self.normalize_extent <= 0.0:
            raise ConfigurationError("normalize_extent must be positive or None")
        if self.weld_tolerance < 0.0:
            raise ConfigurationError("weld_tolerance cannot be negative")
        surface_formats = tuple(
            dict.fromkeys(str(item).strip().lower().lstrip(".") for item in self.export_formats)
        )
        supported_surface = {
            "obj", "ply", "stl", "npz", "glb", "gltf", "fbx", "usd", "usda",
            "usdc", "dae", "abc",
        }
        unsupported_surface = sorted(set(surface_formats) - supported_surface)
        if unsupported_surface:
            raise ConfigurationError(f"unsupported object surface formats: {unsupported_surface}")
        if not surface_formats:
            raise ConfigurationError("at least one object surface export format is required")
        volume_formats = tuple(
            dict.fromkeys(str(item).strip().lower().lstrip(".") for item in self.fem_export_formats)
        )
        supported_volume = {"vtk", "msh", "inp", "npz", "mphtxt"}
        unsupported_volume = sorted(set(volume_formats) - supported_volume)
        if unsupported_volume:
            raise ConfigurationError(f"unsupported FEM export formats: {unsupported_volume}")
        if self.generate_fem and not volume_formats:
            raise ConfigurationError("FEM generation requires at least one FEM export format")
        object.__setattr__(self, "export_formats", surface_formats)
        object.__setattr__(self, "fem_export_formats", volume_formats)
        object.__setattr__(self, "backend_options", dict(self.backend_options))


__all__ = [
    "FEMConfig",
    "FEMMethod",
    "ObjectConfig",
    "ObjectRequest",
    "PhysicalScaleBasis",
]
