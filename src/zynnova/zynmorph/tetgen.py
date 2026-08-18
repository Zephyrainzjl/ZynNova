"""Adaptive, region-partitioned TetGen meshing for ZynMorph volumes.

This module deliberately has no structured-voxel fallback.  Selecting TetGen
means: regularize invalid voxel junctions, extract one conforming multi-phase
PLC, smooth only topologically safe surface vertices, then call the vendored
TetGen 1.6 C++ library through the compiled pybind11 extension.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import importlib.machinery
import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
import site
import sys
from typing import Any

import numpy as np

from ..core.exceptions import GeometryError
from ..geometry import (
    TetraQuality,
    TriangleMesh,
    VolumeMesh,
    extract_material_surfaces,
    tetra_quality,
    tetrahedron_signed_volumes,
)
from .surface import (
    JunctionRegularizationReport,
    MultiphasePLC,
    SurfacePLCAudit,
    audit_multiphase_plc,
    count_nonmanifold_voxel_edges,
    extract_multiphase_plc,
    lock_plc_interfaces,
    regularize_nonmanifold_junctions,
    smooth_multiphase_plc,
)
from .volume import MicrostructureVolume


@dataclass(frozen=True, slots=True)
class LocalRefinementZone:
    """Spherical TetGen ``-u`` refinement zone in physical XYZ coordinates."""

    center_m_xyz: tuple[float, float, float]
    radius_m: float
    maximum_tetra_volume_m3: float
    name: str = "local_refinement"

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center_m_xyz)
        if len(center) != 3 or not np.all(np.isfinite(center)):
            raise ValueError("center_m_xyz must contain three finite values")
        if not np.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive and finite")
        if (
            not np.isfinite(self.maximum_tetra_volume_m3)
            or self.maximum_tetra_volume_m3 <= 0.0
        ):
            raise ValueError("maximum_tetra_volume_m3 must be positive and finite")
        object.__setattr__(self, "center_m_xyz", center)
        object.__setattr__(self, "radius_m", float(self.radius_m))
        object.__setattr__(
            self, "maximum_tetra_volume_m3", float(self.maximum_tetra_volume_m3)
        )
        object.__setattr__(self, "name", str(self.name))


@dataclass(frozen=True, slots=True)
class TetGenMeshingConfig:
    """Surface, quality, and spatial-size controls passed to TetGen."""

    radius_edge_ratio: float = 1.45
    minimum_dihedral_degrees: float = 8.0
    optimization_level: int = 2
    maximum_steiner_points: int = -1
    global_maximum_tetra_volume_m3: float | None = None
    phase_maximum_tetra_volume_m3: Mapping[int, float] = field(default_factory=dict)
    facet_maximum_area_m2: Mapping[object, float] = field(default_factory=dict)
    local_refinement_zones: tuple[LocalRefinementZone, ...] = ()
    smoothing_iterations: int = 8
    smoothing_relaxation: float = 0.34
    smoothing_taubin_mu: float = -0.36
    maximum_surface_displacement_voxels: float = 0.42
    checkerboard_diagonals: bool = True
    normalize_coordinates: bool = True
    preserve_outer_boundary: bool = True
    preserve_multiphase_junctions: bool = True
    fixed_interface_pairs: tuple[tuple[int, int], ...] = ()
    regularize_junctions: bool = True
    junction_maximum_changed_fraction: float = 0.005
    junction_adaptive_budget: bool = True
    junction_hard_maximum_changed_fraction: float = 0.05
    junction_budget_growth_factor: float = 2.0
    junction_maximum_iterations: int = 10_000
    junction_minimum_phase_voxels: int = 8
    junction_preserve_outer_layer: bool = False
    junction_phase_change_penalties: Mapping[int, float] = field(default_factory=dict)
    consistency_check: bool = True
    conforming_delaunay: bool = True
    preserve_boundary_facets: bool = False
    retry_preserve_boundary_on_facet_error: bool = True
    freeform_minimum_clearance_factor: float = 0.35
    freeform_minimum_clearance_m: float | None = None
    freeform_strict_clearance: bool = True
    quiet: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.radius_edge_ratio) or self.radius_edge_ratio <= 1.0:
            raise ValueError("radius_edge_ratio must be finite and greater than one")
        if (
            not np.isfinite(self.minimum_dihedral_degrees)
            or not 0.0 <= self.minimum_dihedral_degrees < 60.0
        ):
            raise ValueError("minimum_dihedral_degrees must lie in [0, 60)")
        if self.optimization_level not in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}:
            raise ValueError("optimization_level must lie in [0, 10]")
        if self.maximum_steiner_points < -1:
            raise ValueError("maximum_steiner_points must be -1 or non-negative")
        if self.global_maximum_tetra_volume_m3 is not None and (
            not np.isfinite(self.global_maximum_tetra_volume_m3)
            or self.global_maximum_tetra_volume_m3 <= 0.0
        ):
            raise ValueError("global_maximum_tetra_volume_m3 must be positive")
        phase_limits = {int(key): float(value) for key, value in self.phase_maximum_tetra_volume_m3.items()}
        if any(not np.isfinite(value) or value <= 0.0 for value in phase_limits.values()):
            raise ValueError("phase maximum tetra volumes must be positive and finite")
        facet_limits = dict(self.facet_maximum_area_m2)
        if any(not np.isfinite(float(value)) or float(value) <= 0.0 for value in facet_limits.values()):
            raise ValueError("facet maximum areas must be positive and finite")
        zones = tuple(
            value if isinstance(value, LocalRefinementZone) else LocalRefinementZone(**value)
            for value in self.local_refinement_zones
        )
        fixed_pairs: list[tuple[int, int]] = []
        for pair in self.fixed_interface_pairs:
            if len(pair) != 2:
                raise ValueError("fixed_interface_pairs entries must contain two region IDs")
            left, right = int(pair[0]), int(pair[1])
            if left == right:
                raise ValueError("fixed_interface_pairs region IDs must be distinct")
            normalized_pair = tuple(sorted((left, right)))
            if normalized_pair not in fixed_pairs:
                fixed_pairs.append(normalized_pair)
        if self.smoothing_iterations < 0:
            raise ValueError("smoothing_iterations cannot be negative")
        if not 0.0 <= self.smoothing_relaxation < 1.0:
            raise ValueError("smoothing_relaxation must lie in [0, 1)")
        if not -1.0 < self.smoothing_taubin_mu <= 0.0:
            raise ValueError("smoothing_taubin_mu must lie in (-1, 0]")
        if self.maximum_surface_displacement_voxels < 0.0:
            raise ValueError("maximum_surface_displacement_voxels cannot be negative")
        if not 0.0 <= self.junction_maximum_changed_fraction <= 1.0:
            raise ValueError("junction_maximum_changed_fraction must lie in [0, 1]")
        if not 0.0 <= self.junction_hard_maximum_changed_fraction <= 1.0:
            raise ValueError("junction_hard_maximum_changed_fraction must lie in [0, 1]")
        if self.junction_hard_maximum_changed_fraction < self.junction_maximum_changed_fraction:
            raise ValueError(
                "junction_hard_maximum_changed_fraction cannot be smaller than "
                "junction_maximum_changed_fraction"
            )
        if self.junction_adaptive_budget and (
            not np.isfinite(self.junction_budget_growth_factor)
            or self.junction_budget_growth_factor <= 1.0
        ):
            raise ValueError(
                "junction_budget_growth_factor must be finite and greater than one"
            )
        if self.junction_maximum_iterations < 1:
            raise ValueError("junction_maximum_iterations must be positive")
        if self.junction_minimum_phase_voxels < 1:
            raise ValueError("junction_minimum_phase_voxels must be positive")
        if (
            not np.isfinite(self.freeform_minimum_clearance_factor)
            or self.freeform_minimum_clearance_factor < 0.0
        ):
            raise ValueError(
                "freeform_minimum_clearance_factor must be finite and non-negative"
            )
        if self.freeform_minimum_clearance_m is not None and (
            not np.isfinite(self.freeform_minimum_clearance_m)
            or self.freeform_minimum_clearance_m < 0.0
        ):
            raise ValueError(
                "freeform_minimum_clearance_m must be finite and non-negative"
            )
        penalties = {
            int(key): float(value)
            for key, value in self.junction_phase_change_penalties.items()
        }
        object.__setattr__(self, "radius_edge_ratio", float(self.radius_edge_ratio))
        object.__setattr__(
            self, "minimum_dihedral_degrees", float(self.minimum_dihedral_degrees)
        )
        object.__setattr__(
            self,
            "global_maximum_tetra_volume_m3",
            None
            if self.global_maximum_tetra_volume_m3 is None
            else float(self.global_maximum_tetra_volume_m3),
        )
        object.__setattr__(self, "phase_maximum_tetra_volume_m3", phase_limits)
        object.__setattr__(self, "facet_maximum_area_m2", facet_limits)
        object.__setattr__(self, "local_refinement_zones", zones)
        object.__setattr__(self, "fixed_interface_pairs", tuple(fixed_pairs))
        object.__setattr__(self, "junction_phase_change_penalties", penalties)
        object.__setattr__(
            self,
            "freeform_minimum_clearance_factor",
            float(self.freeform_minimum_clearance_factor),
        )
        object.__setattr__(
            self,
            "freeform_minimum_clearance_m",
            None
            if self.freeform_minimum_clearance_m is None
            else float(self.freeform_minimum_clearance_m),
        )


@dataclass(frozen=True, slots=True)
class RegionSeed:
    point_m_xyz: tuple[float, float, float]
    phase: int
    component: int
    component_voxels: int
    maximum_tetra_volume_m3: float | None


@dataclass(frozen=True, slots=True)
class TetGenNativeStatus:
    available: bool
    version: str
    reason: str | None
    module_path: Path | None
    vendored_source_path: Path | None
    license: str


@dataclass(frozen=True, slots=True)
class TetGenMeshResult:
    mesh: VolumeMesh
    boundary: TriangleMesh
    interface_faces: Mapping[tuple[int, int], np.ndarray]
    quality: TetraQuality
    input_plc: MultiphasePLC
    input_plc_audit: SurfacePLCAudit
    output_surface: TriangleMesh | None
    region_seeds: tuple[RegionSeed, ...]
    meshed_volume: MicrostructureVolume
    junction_report: JunctionRegularizationReport | None
    switches: str
    native_version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


_TETGEN_NATIVE_MODULE: Any | None = None
_TETGEN_NATIVE_IMPORT_ERROR: str | None = None


def _candidate_tetgen_extension_paths() -> tuple[Path, ...]:
    """Return plausible installed/build-tree TetGen extension paths for diagnostics/fallback."""

    names = tuple(f"_zynmorph_tetgen_native{suffix}" for suffix in importlib.machinery.EXTENSION_SUFFIXES)
    roots: list[Path] = []

    try:
        import zynnova as _zynnova
        for package_root in getattr(_zynnova, "__path__", ()):
            roots.append(Path(package_root) / "_native")
    except Exception:
        pass

    for value in sys.path:
        if value:
            roots.append(Path(value) / "zynnova" / "_native")
    try:
        roots.extend(Path(value) / "zynnova" / "_native" for value in site.getsitepackages())
    except Exception:
        pass
    try:
        roots.append(Path(site.getusersitepackages()) / "zynnova" / "_native")
    except Exception:
        pass

    # Source-checkout fallback for editable development builds.  This is only
    # used after the normal import path fails and is ABI/suffix filtered.
    source_file = Path(__file__).resolve()
    project_root = next(
        (parent for parent in source_file.parents if (parent / "pyproject.toml").is_file()),
        None,
    )
    if project_root is not None:
        build_root = project_root / "build"
        if build_root.is_dir():
            for pattern in ("*/cpp", "*/cpp/Release", "*/cpp/RelWithDebInfo", "*/cpp/Debug"):
                roots.extend(build_root.glob(pattern))

    found: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        for name in names:
            candidate = root / name
            if not candidate.is_file():
                continue
            try:
                key = str(candidate.resolve()).casefold()
            except OSError:
                continue
            if key not in seen:
                seen.add(key)
                found.append(candidate.resolve())
    return tuple(found)


def _validate_tetgen_native_module(native: Any) -> None:
    if not callable(getattr(native, "tetrahedralize", None)):
        raise ImportError("TetGen native module does not expose tetrahedralize()")
    version = str(getattr(native, "tetgen_version", ""))
    if "1.6" not in version:
        raise ImportError(f"unexpected TetGen native version: {version!r}")
    binding_abi = int(getattr(native, "tetgen_binding_abi", 0))
    if binding_abi < 2:
        raise ImportError(
            "TetGen native extension ABI is too old for robust free-form boundary "
            f"recovery (found ABI {binding_abi}, require >=2). Rebuild ZynNova with "
            "`python -m pip install -e .[zynmorph-all] -v` and restart Python/Jupyter."
        )


def _load_tetgen_native() -> Any:
    """Load the pip-built TetGen extension, with a transparent editable-build fallback."""

    global _TETGEN_NATIVE_MODULE, _TETGEN_NATIVE_IMPORT_ERROR
    if _TETGEN_NATIVE_MODULE is not None:
        return _TETGEN_NATIVE_MODULE

    errors: list[str] = []
    module_name = "zynnova._native._zynmorph_tetgen_native"
    try:
        native = importlib.import_module(module_name)
        _validate_tetgen_native_module(native)
        _TETGEN_NATIVE_MODULE = native
        _TETGEN_NATIVE_IMPORT_ERROR = None
        return native
    except Exception as exc:
        errors.append(f"normal import: {type(exc).__name__}: {exc}")

    # A redirect editable can leave the extension in the CMake install/build
    # tree while Python sources are imported from ``src``.  Search only paths
    # matching this Python's native extension suffix and load under the exact
    # package-qualified module name.
    for candidate in _candidate_tetgen_extension_paths():
        try:
            if os.name == "nt":
                try:
                    native_pkg = importlib.import_module("zynnova._native")
                    register = getattr(native_pkg, "_register_windows_dll_directory", None)
                    if callable(register):
                        register(candidate.parent)
                except Exception:
                    pass
            spec = importlib.util.spec_from_file_location(module_name, candidate)
            if spec is None or spec.loader is None:
                raise ImportError("could not create extension-module spec")
            native = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = native
            try:
                spec.loader.exec_module(native)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
            _validate_tetgen_native_module(native)
            _TETGEN_NATIVE_MODULE = native
            _TETGEN_NATIVE_IMPORT_ERROR = None
            return native
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")

    _TETGEN_NATIVE_IMPORT_ERROR = " | ".join(errors)
    raise ImportError(_TETGEN_NATIVE_IMPORT_ERROR)


def tetgen_native_status() -> TetGenNativeStatus:
    """Report whether the pip-built TetGen pybind11 extension is importable."""

    try:
        native = _load_tetgen_native()
    except Exception as exc:
        candidates = _candidate_tetgen_extension_paths()
        suffix = ""
        if candidates:
            suffix = "; candidates=" + ", ".join(str(path) for path in candidates)
        return TetGenNativeStatus(
            available=False,
            version="TetGen 1.6.0",
            reason=f"native extension is unavailable: {type(exc).__name__}: {exc}{suffix}",
            module_path=None,
            vendored_source_path=None,
            license="AGPL-3.0-or-later (TetGen); ZynNova binding is MIT",
        )
    module_path = Path(native.__file__).resolve()
    source_root = module_path.parent.parent / "_third_party" / "tetgen"
    return TetGenNativeStatus(
        available=True,
        version=str(getattr(native, "tetgen_version", "TetGen 1.6.0")),
        reason=None,
        module_path=module_path,
        vendored_source_path=source_root if source_root.exists() else None,
        license="AGPL-3.0-or-later (TetGen); ZynNova binding is MIT",
    )


def tetgen_native_diagnostics() -> Mapping[str, Any]:
    """Return actionable import/path diagnostics without hiding the original loader error."""

    status = tetgen_native_status()
    try:
        import zynnova as _zynnova
        package_file = str(Path(_zynnova.__file__).resolve()) if _zynnova.__file__ else None
        package_paths = [str(Path(value).resolve()) for value in getattr(_zynnova, "__path__", ())]
    except Exception as exc:
        package_file = None
        package_paths = [f"zynnova import failed: {type(exc).__name__}: {exc}"]
    try:
        native_spec = importlib.util.find_spec("zynnova._native")
        native_spec_text = None if native_spec is None else str(native_spec)
    except Exception as exc:
        native_spec_text = f"{type(exc).__name__}: {exc}"
    try:
        extension_spec = importlib.util.find_spec("zynnova._native._zynmorph_tetgen_native")
        extension_spec_text = None if extension_spec is None else str(extension_spec)
    except Exception as exc:
        extension_spec_text = f"{type(exc).__name__}: {exc}"
    try:
        native_pkg = importlib.import_module("zynnova._native")
        native_paths = [str(Path(value).resolve()) for value in getattr(native_pkg, "__path__", ())]
        runtime_dirs = list(getattr(native_pkg, "WINDOWS_RUNTIME_DIRECTORIES", ()))
    except Exception as exc:
        native_paths = [f"native package import failed: {type(exc).__name__}: {exc}"]
        runtime_dirs = []
    return {
        "available": status.available,
        "version": status.version,
        "reason": status.reason,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": sys.platform,
        "zynnova_file": package_file,
        "zynnova_path": package_paths,
        "native_package_path": native_paths,
        "native_spec": native_spec_text,
        "extension_spec": extension_spec_text,
        "candidate_extensions": [str(path) for path in _candidate_tetgen_extension_paths()],
        "windows_runtime_directories": runtime_dirs,
    }


def build_region_seeds(
    volume: MicrostructureVolume,
    config: TetGenMeshingConfig | None = None,
) -> tuple[RegionSeed, ...]:
    """Find one strictly interior seed for every 6-connected phase component."""

    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - optional dependency gate
        raise RuntimeError("adaptive TetGen meshing requires scipy>=1.10") from exc

    resolved = config or TetGenMeshingConfig()
    labels = volume.labels
    structure = ndimage.generate_binary_structure(3, 1)
    spacing_zyx = tuple(map(float, volume.voxel_size_m))
    oz, oy, ox = map(float, volume.origin_m)
    dz, dy, dx = spacing_zyx
    seeds: list[RegionSeed] = []
    for phase in map(int, np.unique(labels)):
        components, count = ndimage.label(labels == phase, structure=structure)
        slices = ndimage.find_objects(components)
        for component_id in range(1, count + 1):
            bounds = slices[component_id - 1]
            if bounds is None:
                continue
            component = components[bounds] == component_id
            padded = np.pad(component, 1, mode="constant", constant_values=False)
            distance = ndimage.distance_transform_edt(
                padded, sampling=spacing_zyx
            )[1:-1, 1:-1, 1:-1]
            distance[~component] = -1.0
            local_flat = int(np.argmax(distance))
            local_zyx = np.unravel_index(local_flat, component.shape)
            global_zyx = tuple(
                int(bounds[axis].start + local_zyx[axis]) for axis in range(3)
            )
            z, y, x = global_zyx
            maximum = resolved.phase_maximum_tetra_volume_m3.get(
                phase, resolved.global_maximum_tetra_volume_m3
            )
            # A zone containing the seed can tighten this connected component;
            # the C++ -u callback still supplies true local refinement inside it.
            seed_xyz = (
                ox + (x + 0.5) * dx,
                oy + (y + 0.5) * dy,
                oz + (z + 0.5) * dz,
            )
            for zone in resolved.local_refinement_zones:
                if np.linalg.norm(np.asarray(seed_xyz) - np.asarray(zone.center_m_xyz)) <= zone.radius_m:
                    maximum = (
                        zone.maximum_tetra_volume_m3
                        if maximum is None
                        else min(maximum, zone.maximum_tetra_volume_m3)
                    )
            seeds.append(
                RegionSeed(
                    point_m_xyz=tuple(map(float, seed_xyz)),
                    phase=phase,
                    component=component_id,
                    component_voxels=int(np.count_nonzero(component)),
                    maximum_tetra_volume_m3=maximum,
                )
            )
    if not seeds:
        raise GeometryError("no TetGen region seeds could be constructed")
    return tuple(seeds)


def mesh_microstructure_tetgen(
    volume: MicrostructureVolume,
    *,
    config: TetGenMeshingConfig | None = None,
    maximum_tetrahedra: int = 12_000_000,
) -> TetGenMeshResult:
    """Generate an adaptive conforming Tet4 mesh with the native TetGen core."""

    resolved = config or TetGenMeshingConfig()
    if maximum_tetrahedra < 1:
        raise ValueError("maximum_tetrahedra must be positive")
    status = tetgen_native_status()
    if not status.available:
        raise RuntimeError(
            "TetGen meshing was selected, but the native extension is unavailable. "
            "Vendor the pinned source with `python scripts/vendor_tetgen.py --accept-agpl` "
            "and reinstall with `python -m pip install -e .`. "
            f"Details: {status.reason}"
        )
    native = _load_tetgen_native()

    junction_report: JunctionRegularizationReport | None
    if resolved.regularize_junctions:
        meshed_volume, junction_report = regularize_nonmanifold_junctions(
            volume,
            maximum_changed_fraction=resolved.junction_maximum_changed_fraction,
            adaptive_budget=resolved.junction_adaptive_budget,
            hard_maximum_changed_fraction=resolved.junction_hard_maximum_changed_fraction,
            budget_growth_factor=resolved.junction_budget_growth_factor,
            maximum_iterations=resolved.junction_maximum_iterations,
            minimum_phase_voxels=resolved.junction_minimum_phase_voxels,
            preserve_outer_layer=resolved.junction_preserve_outer_layer,
            phase_change_penalties=resolved.junction_phase_change_penalties,
            strict=True,
        )
    else:
        ambiguous = count_nonmanifold_voxel_edges(volume.labels)
        if ambiguous:
            raise GeometryError(
                f"volume contains {ambiguous} non-manifold diagonal voxel junctions; "
                "enable regularize_junctions before TetGen"
            )
        meshed_volume = volume
        junction_report = None

    plc = extract_multiphase_plc(
        meshed_volume,
        checkerboard_diagonals=resolved.checkerboard_diagonals,
        preserve_outer_boundary=resolved.preserve_outer_boundary,
        preserve_multiphase_junctions=resolved.preserve_multiphase_junctions,
        strict=True,
    )
    if resolved.fixed_interface_pairs:
        plc = lock_plc_interfaces(
            plc,
            resolved.fixed_interface_pairs,
            require_present=True,
        )
    maximum_displacement = (
        resolved.maximum_surface_displacement_voxels
        * min(map(float, meshed_volume.voxel_size_m))
    )
    if resolved.smoothing_iterations:
        plc = smooth_multiphase_plc(
            plc,
            iterations=resolved.smoothing_iterations,
            relaxation=resolved.smoothing_relaxation,
            taubin_mu=resolved.smoothing_taubin_mu,
            maximum_displacement_m=maximum_displacement
            if maximum_displacement > 0.0
            else None,
        )
    plc_audit = audit_multiphase_plc(plc)
    if not plc_audit.valid:
        raise GeometryError(f"TetGen input PLC failed its final audit: {plc_audit}")

    seeds = build_region_seeds(meshed_volume, resolved)
    seed_array = np.asarray(
        [
            (*seed.point_m_xyz, float(seed.phase), -1.0 if seed.maximum_tetra_volume_m3 is None else seed.maximum_tetra_volume_m3)
            for seed in seeds
        ],
        dtype=np.float64,
    )
    facet_constraints = _resolve_facet_constraints(plc, resolved.facet_maximum_area_m2)
    zones = np.asarray(
        [
            (*zone.center_m_xyz, zone.radius_m, zone.maximum_tetra_volume_m3)
            for zone in resolved.local_refinement_zones
        ],
        dtype=np.float64,
    ).reshape(-1, 5)

    native_points, native_seeds, native_constraints, native_zones, transform = (
        _normalize_tetgen_input(plc.vertices, seed_array, facet_constraints, zones)
        if resolved.normalize_coordinates
        else (plc.vertices, seed_array, facet_constraints, zones, None)
    )

    raw = native.tetrahedralize(
        native_points,
        plc.triangles,
        plc.facet_markers,
        native_seeds,
        np.empty((0, 3), dtype=np.float64),
        native_constraints,
        native_zones,
        float(resolved.radius_edge_ratio),
        float(resolved.minimum_dihedral_degrees),
        int(resolved.optimization_level),
        int(resolved.maximum_steiner_points),
        bool(resolved.consistency_check),
        bool(resolved.conforming_delaunay),
        bool(resolved.quiet),
    )
    nodes = np.ascontiguousarray(raw["points"], dtype=np.float64)
    if transform is not None:
        offset, scale = transform
        nodes = np.ascontiguousarray(nodes * scale + offset, dtype=np.float64)
    tetrahedra = np.ascontiguousarray(raw["tetrahedra"], dtype=np.int64)
    if len(tetrahedra) > maximum_tetrahedra:
        raise GeometryError(
            f"TetGen produced {len(tetrahedra):,} tetrahedra, exceeding "
            f"maximum_tetrahedra={maximum_tetrahedra:,}"
        )
    attributes = np.asarray(raw["region_attributes"], dtype=np.float64)
    if attributes.shape != (len(tetrahedra),):
        raise GeometryError("TetGen did not return one region attribute per tetrahedron")
    rounded = np.rint(attributes)
    if not np.allclose(attributes, rounded, rtol=0.0, atol=1.0e-8):
        raise GeometryError("TetGen returned non-integral material attributes")
    cell_regions = np.ascontiguousarray(rounded, dtype=np.int32)

    tentative = VolumeMesh(
        nodes=nodes,
        tetrahedra=tetrahedra,
        cell_regions=cell_regions,
        region_names=meshed_volume.phase_names,
        metadata={"source": "tetgen-1.6.0", "switches": str(raw["switches"])},
    )
    signed = tetrahedron_signed_volumes(tentative)
    negative = signed < 0.0
    if np.any(negative):
        tetrahedra = tetrahedra.copy()
        first = tetrahedra[negative, 0].copy()
        tetrahedra[negative, 0] = tetrahedra[negative, 1]
        tetrahedra[negative, 1] = first
    mesh = VolumeMesh(
        nodes=nodes,
        tetrahedra=tetrahedra,
        cell_regions=cell_regions,
        region_names=meshed_volume.phase_names,
        metadata={
            "source": "tetgen-1.6.0",
            "switches": str(raw["switches"]),
            "adaptive": True,
            "input_voxel_shape_zyx": meshed_volume.shape,
        },
    )
    quality = tetra_quality(mesh)
    if not quality.fem_ready:
        raise GeometryError(
            "TetGen output failed FEM readiness: "
            f"inverted={quality.inverted_cells}, degenerate={quality.degenerate_cells}"
        )
    input_phases = set(map(int, np.unique(meshed_volume.labels)))
    output_phases = set(map(int, np.unique(cell_regions)))
    if input_phases != output_phases:
        raise GeometryError(
            "TetGen region coverage mismatch: "
            f"input={sorted(input_phases)}, output={sorted(output_phases)}"
        )

    boundary, interfaces = extract_material_surfaces(mesh)
    trifaces = np.asarray(raw.get("trifaces", np.empty((0, 3))), dtype=np.int64)
    trimarkers = np.asarray(raw.get("triface_markers", np.empty(0)), dtype=np.int32)
    output_surface = None
    if trifaces.ndim == 2 and trifaces.shape[1:] == (3,) and len(trifaces):
        output_surface = TriangleMesh(
            vertices=nodes,
            faces=trifaces,
            face_materials=trimarkers if trimarkers.shape == (len(trifaces),) else None,
            metadata={"source": "tetgen-output-subfaces"},
        )

    return TetGenMeshResult(
        mesh=mesh,
        boundary=boundary,
        interface_faces=interfaces,
        quality=quality,
        input_plc=plc,
        input_plc_audit=plc_audit,
        output_surface=output_surface,
        region_seeds=seeds,
        meshed_volume=meshed_volume,
        junction_report=junction_report,
        switches=str(raw["switches"]),
        native_version=str(raw.get("version", status.version)),
        metadata={
            "tetgen_stdout_suppressed": bool(resolved.quiet),
            "local_refinement_zone_count": len(resolved.local_refinement_zones),
            "facet_constraint_count": len(facet_constraints),
            "region_seed_count": len(seeds),
            "fixed_interface_pairs": resolved.fixed_interface_pairs,
            "coordinate_normalization": None
            if transform is None
            else {
                "offset_m_xyz": transform[0].tolist(),
                "scale_m": float(transform[1]),
            },
        },
    )


def _normalize_tetgen_input(
    points: np.ndarray,
    region_seeds: np.ndarray,
    facet_constraints: np.ndarray,
    local_zones: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, float],
]:
    """Map SI-scale geometry to an O(1) box before robust predicates run."""

    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    scale = float(np.max(maximum - minimum))
    if not np.isfinite(scale) or scale <= 0.0:
        raise GeometryError("TetGen PLC has zero or non-finite physical extent")
    offset = minimum.astype(np.float64, copy=True)
    normalized_points = np.ascontiguousarray((points - offset) / scale)
    normalized_seeds = np.ascontiguousarray(region_seeds.copy(), dtype=np.float64)
    normalized_seeds[:, :3] = (normalized_seeds[:, :3] - offset) / scale
    finite_volume = normalized_seeds[:, 4] > 0.0
    normalized_seeds[finite_volume, 4] /= scale**3

    normalized_constraints = np.ascontiguousarray(
        facet_constraints.copy(), dtype=np.float64
    )
    if len(normalized_constraints):
        normalized_constraints[:, 1] /= scale**2

    normalized_zones = np.ascontiguousarray(local_zones.copy(), dtype=np.float64)
    if len(normalized_zones):
        normalized_zones[:, :3] = (normalized_zones[:, :3] - offset) / scale
        normalized_zones[:, 3] /= scale
        normalized_zones[:, 4] /= scale**3
    return (
        normalized_points,
        normalized_seeds,
        normalized_constraints,
        normalized_zones,
        (offset, scale),
    )



def _resolve_facet_constraints(
    plc: MultiphasePLC,
    configured: Mapping[object, float],
) -> np.ndarray:
    rows: dict[int, float] = {}
    for key, raw_value in configured.items():
        area = float(raw_value)
        matched: list[int] = []
        if isinstance(key, (int, np.integer)):
            matched = [int(key)] if int(key) in plc.marker_names else []
        elif isinstance(key, str):
            matched = [marker for marker, name in plc.marker_names.items() if name == key]
        elif isinstance(key, tuple) and len(key) == 2:
            pair = tuple(sorted(map(int, key)))
            matched = [
                marker
                for marker, value in plc.marker_region_pairs.items()
                if value[0] is not None
                and value[1] is not None
                and tuple(sorted((int(value[0]), int(value[1])))) == pair
            ]
        if not matched:
            raise ValueError(f"facet area constraint key did not match a PLC marker: {key!r}")
        for marker in matched:
            rows[marker] = min(rows.get(marker, area), area)
    return np.asarray(sorted(rows.items()), dtype=np.float64).reshape(-1, 2)


__all__ = [
    "LocalRefinementZone",
    "RegionSeed",
    "TetGenMeshResult",
    "TetGenMeshingConfig",
    "TetGenNativeStatus",
    "build_region_seeds",
    "mesh_microstructure_tetgen",
    "tetgen_native_status",
    "tetgen_native_diagnostics",
]
