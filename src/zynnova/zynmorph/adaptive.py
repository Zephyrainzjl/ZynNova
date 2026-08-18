"""General adaptive non-structured TetGen sizing for arbitrary complex regions.

This module provides the high-level production policy used when users want a
quality unstructured tetrahedral mesh without manually converting physical edge
length targets into TetGen volume/area constraints.  It deliberately never
falls back to the structured six-tetrahedra-per-voxel path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .tetgen import LocalRefinementZone, TetGenMeshingConfig
from .volume import MicrostructureVolume


def regular_tetrahedron_volume_from_edge(edge_length_m: float) -> float:
    """Return the regular-tetrahedron volume for a physical edge length."""

    edge = float(edge_length_m)
    if not np.isfinite(edge) or edge <= 0.0:
        raise ValueError("edge_length_m must be positive and finite")
    return float(np.sqrt(2.0) / 12.0 * edge**3)


def equilateral_triangle_area_from_edge(edge_length_m: float) -> float:
    """Return the equilateral-triangle area for a physical edge length."""

    edge = float(edge_length_m)
    if not np.isfinite(edge) or edge <= 0.0:
        raise ValueError("edge_length_m must be positive and finite")
    return float(np.sqrt(3.0) / 4.0 * edge**2)


def _surface_median_edge_length(vertices: np.ndarray, triangles: np.ndarray) -> float:
    xyz = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
        raise ValueError("cannot infer mesh scale from an empty/non-triangular surface")
    tri = xyz[faces]
    edges = np.concatenate(
        (
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        )
    )
    edges = edges[np.isfinite(edges) & (edges > 0.0)]
    if not len(edges):
        raise ValueError("surface does not contain a finite positive edge length")
    return float(np.median(edges))


def infer_irregular_base_edge_length_m(source: object) -> float:
    """Infer a conservative bulk Tet4 scale for a volume, PLC, or shell set."""

    if isinstance(source, MicrostructureVolume):
        # The TetGen grid is intentionally decoupled from the voxel lattice.
        # A two-voxel bulk target gives the Delaunay optimizer room to create
        # genuinely irregular tetrahedra; interface constraints may be finer.
        return float(2.0 * min(map(float, source.voxel_size_m)))

    from .freeform import SurfaceShell
    from .surface import MultiphasePLC

    if isinstance(source, MultiphasePLC):
        return 1.35 * _surface_median_edge_length(source.vertices, source.triangles)

    if isinstance(source, Sequence) and source and all(
        isinstance(item, SurfaceShell) for item in source
    ):
        medians = [
            _surface_median_edge_length(item.surface.vertices, item.surface.faces)
            for item in source
        ]
        return 1.35 * float(np.median(medians))

    raise TypeError(
        "cannot infer an irregular mesh scale from this source; supply "
        "IrregularMeshPolicy(base_edge_length_m=...)"
    )


@dataclass(frozen=True, slots=True)
class IrregularMeshPolicy:
    """Physical sizing/quality policy for arbitrary multi-domain TetGen meshes.

    The API is expressed in target edge lengths rather than TetGen's raw volume
    constraints.  Material IDs may represent disconnected objects; e.g. thirty
    cathode particles may all use region ``1`` and therefore become one FEM /
    COMSOL material domain while still receiving multiple interior seeds.
    """

    base_edge_length_m: float | None = None
    region_edge_lengths_m: Mapping[int, float] = field(default_factory=dict)
    interface_edge_lengths_m: Mapping[object, float] = field(default_factory=dict)
    local_refinement_zones: tuple[LocalRefinementZone, ...] = ()
    fixed_interface_pairs: tuple[tuple[int, int], ...] = ()

    radius_edge_ratio: float = 1.45
    minimum_dihedral_degrees: float = 8.0
    optimization_level: int = 2
    maximum_steiner_points: int = -1

    smoothing_iterations: int = 8
    smoothing_relaxation: float = 0.34
    smoothing_taubin_mu: float = -0.36
    maximum_surface_displacement_voxels: float = 0.42

    checkerboard_diagonals: bool = True
    preserve_outer_boundary: bool = True
    preserve_multiphase_junctions: bool = True

    regularize_junctions: bool = True
    junction_maximum_changed_fraction: float = 0.005
    junction_adaptive_budget: bool = True
    junction_hard_maximum_changed_fraction: float = 0.05
    junction_budget_growth_factor: float = 2.0
    junction_maximum_iterations: int = 10_000
    junction_minimum_phase_voxels: int = 8
    junction_preserve_outer_layer: bool = False
    junction_phase_change_penalties: Mapping[int, float] = field(default_factory=dict)

    normalize_coordinates: bool = True
    consistency_check: bool = True
    conforming_delaunay: bool = True
    preserve_boundary_facets: bool = False
    retry_preserve_boundary_on_facet_error: bool = True
    freeform_minimum_clearance_factor: float = 0.35
    freeform_minimum_clearance_m: float | None = None
    freeform_strict_clearance: bool = True
    quiet: bool = True

    def __post_init__(self) -> None:
        if self.base_edge_length_m is not None and (
            not np.isfinite(self.base_edge_length_m)
            or float(self.base_edge_length_m) <= 0.0
        ):
            raise ValueError("base_edge_length_m must be positive and finite")
        for label, value in self.region_edge_lengths_m.items():
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    f"invalid region edge length for region {int(label)}: {value}"
                )
        for label, value in self.interface_edge_lengths_m.items():
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"invalid interface edge length for {label!r}: {value}")

        zones = tuple(
            zone
            if isinstance(zone, LocalRefinementZone)
            else LocalRefinementZone(**dict(zone))
            for zone in self.local_refinement_zones
        )
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
        object.__setattr__(self, "local_refinement_zones", zones)

    def to_tetgen_config(self, source: object) -> TetGenMeshingConfig:
        base = (
            infer_irregular_base_edge_length_m(source)
            if self.base_edge_length_m is None
            else float(self.base_edge_length_m)
        )
        return TetGenMeshingConfig(
            radius_edge_ratio=self.radius_edge_ratio,
            minimum_dihedral_degrees=self.minimum_dihedral_degrees,
            optimization_level=self.optimization_level,
            maximum_steiner_points=self.maximum_steiner_points,
            global_maximum_tetra_volume_m3=regular_tetrahedron_volume_from_edge(base),
            phase_maximum_tetra_volume_m3={
                int(region): regular_tetrahedron_volume_from_edge(float(edge))
                for region, edge in self.region_edge_lengths_m.items()
            },
            facet_maximum_area_m2={
                key: equilateral_triangle_area_from_edge(float(edge))
                for key, edge in self.interface_edge_lengths_m.items()
            },
            local_refinement_zones=self.local_refinement_zones,
            fixed_interface_pairs=self.fixed_interface_pairs,
            smoothing_iterations=self.smoothing_iterations,
            smoothing_relaxation=self.smoothing_relaxation,
            smoothing_taubin_mu=self.smoothing_taubin_mu,
            maximum_surface_displacement_voxels=self.maximum_surface_displacement_voxels,
            checkerboard_diagonals=self.checkerboard_diagonals,
            preserve_outer_boundary=self.preserve_outer_boundary,
            preserve_multiphase_junctions=self.preserve_multiphase_junctions,
            regularize_junctions=self.regularize_junctions,
            junction_maximum_changed_fraction=self.junction_maximum_changed_fraction,
            junction_adaptive_budget=self.junction_adaptive_budget,
            junction_hard_maximum_changed_fraction=self.junction_hard_maximum_changed_fraction,
            junction_budget_growth_factor=self.junction_budget_growth_factor,
            junction_maximum_iterations=self.junction_maximum_iterations,
            junction_minimum_phase_voxels=self.junction_minimum_phase_voxels,
            junction_preserve_outer_layer=self.junction_preserve_outer_layer,
            junction_phase_change_penalties=self.junction_phase_change_penalties,
            normalize_coordinates=self.normalize_coordinates,
            consistency_check=self.consistency_check,
            conforming_delaunay=self.conforming_delaunay,
            preserve_boundary_facets=self.preserve_boundary_facets,
            retry_preserve_boundary_on_facet_error=self.retry_preserve_boundary_on_facet_error,
            freeform_minimum_clearance_factor=self.freeform_minimum_clearance_factor,
            freeform_minimum_clearance_m=self.freeform_minimum_clearance_m,
            freeform_strict_clearance=self.freeform_strict_clearance,
            quiet=self.quiet,
        )


def mesh_complex_regions(
    source: object,
    *,
    regions: Sequence[object] | None = None,
    policy: IrregularMeshPolicy | None = None,
    tetgen_config: TetGenMeshingConfig | Mapping[str, object] | None = None,
    material_region_map: Mapping[int, int] | None = None,
    material_region_names: Mapping[int, str] | None = None,
    require_complete_region_map: bool = False,
    holes_m_xyz: Sequence[tuple[float, float, float]] = (),
    void_regions: Sequence[int] = (),
    maximum_tetrahedra: int = 20_000_000,
    weld_tolerance_m: float | None = None,
):
    """Generate quality non-structured Tet4 for arbitrary complex regions.

    ``source`` may be a labelled :class:`MicrostructureVolume`, a conforming
    :class:`MultiphasePLC`, or a sequence of watertight ``SurfaceShell``
    objects.  The function always dispatches to native TetGen and never silently
    uses the structured voxel splitter.

    For labelled volumes, ``material_region_map`` is applied *before* PLC
    extraction.  This is the supported way to keep per-particle tracking IDs in
    the voxel data while collapsing every cathode particle into one final FEM
    region and every anode particle into another.
    """

    if tetgen_config is not None and policy is not None:
        raise ValueError("supply either policy or tetgen_config, not both")

    if tetgen_config is None:
        resolved = (policy or IrregularMeshPolicy()).to_tetgen_config(source)
    elif isinstance(tetgen_config, TetGenMeshingConfig):
        resolved = tetgen_config
    else:
        resolved = TetGenMeshingConfig(**dict(tetgen_config))

    from .meshing import mesh_unstructured_regions

    return mesh_unstructured_regions(
        source,
        regions=regions,
        tetgen_config=resolved,
        material_region_map=material_region_map,
        material_region_names=material_region_names,
        require_complete_region_map=require_complete_region_map,
        holes_m_xyz=holes_m_xyz,
        void_regions=void_regions,
        maximum_tetrahedra=maximum_tetrahedra,
        weld_tolerance_m=weld_tolerance_m,
    )


__all__ = [
    "IrregularMeshPolicy",
    "equilateral_triangle_area_from_edge",
    "infer_irregular_base_edge_length_m",
    "mesh_complex_regions",
    "regular_tetrahedron_volume_from_edge",
]
