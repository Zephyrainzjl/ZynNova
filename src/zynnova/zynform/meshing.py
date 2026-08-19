"""Watertight surface-to-Tet4 conversion with TetGen, Gmsh, and voxel fallbacks."""

from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ..core import BackendUnavailableError, GeometryError
from ..geometry import (
    TriangleMesh,
    VolumeMesh,
    export_triangle_mesh,
    select_volume_regions,
    tetra_quality,
    tetrahedron_signed_volumes,
    triangle_quality,
    voxel_to_tetrahedra,
)
from .schema import FEMConfig, FEMMethod


class FEMMeshingError(GeometryError):
    """Raised when every requested tetrahedralization strategy fails."""


def tetrahedralize_surface(
    mesh: TriangleMesh,
    config: FEMConfig | None = None,
) -> VolumeMesh:
    """Generate and quality-check a conforming first-order tetrahedral mesh."""

    config = config or FEMConfig()
    surface_quality = triangle_quality(mesh)
    if config.require_watertight and not surface_quality.watertight:
        raise FEMMeshingError(
            "surface is not watertight: "
            f"boundary_edges={surface_quality.boundary_edges}, "
            f"nonmanifold_edges={surface_quality.nonmanifold_edges}"
        )
    methods = _method_order(config.method)
    failures: list[str] = []
    for method in methods:
        try:
            if method == FEMMethod.TETGEN:
                volume = _with_tetgen(mesh, config)
            elif method == FEMMethod.GMSH:
                volume = _with_gmsh(mesh, config)
            elif method == FEMMethod.VOXEL:
                volume = _with_voxels(mesh, config)
            else:  # pragma: no cover - enum guards this branch
                continue
            volume = _orient_positive(volume)
            quality = tetra_quality(volume)
            if not quality.fem_ready:
                raise FEMMeshingError(
                    f"{method.value} returned invalid Tet4 cells: "
                    f"inverted={quality.inverted_cells}, degenerate={quality.degenerate_cells}"
                )
            if quality.minimum_mean_ratio < config.minimum_mean_ratio:
                raise FEMMeshingError(
                    f"{method.value} minimum mean-ratio={quality.minimum_mean_ratio:.6g} "
                    f"is below requested {config.minimum_mean_ratio:.6g}"
                )
            if volume.n_cells > config.maximum_cells:
                raise FEMMeshingError(
                    f"{method.value} created {volume.n_cells} cells, above "
                    f"maximum_cells={config.maximum_cells}"
                )
            return VolumeMesh(
                nodes=volume.nodes,
                tetrahedra=volume.tetrahedra,
                cell_regions=np.full(volume.n_cells, config.region_id, dtype=np.int32),
                region_names={config.region_id: config.region_name},
                metadata={
                    **volume.metadata,
                    "meshing_method": method.value,
                    "surface_watertight": surface_quality.watertight,
                    "minimum_mean_ratio": quality.minimum_mean_ratio,
                },
            )
        except (ImportError, BackendUnavailableError, GeometryError, RuntimeError, ValueError) as exc:
            failures.append(f"{method.value}: {exc}")
            if config.method != FEMMethod.AUTO:
                raise FEMMeshingError(failures[-1]) from exc
    raise FEMMeshingError("all FEM meshing methods failed; " + "; ".join(failures))


def _method_order(method: FEMMethod) -> tuple[FEMMethod, ...]:
    if method == FEMMethod.AUTO:
        return (FEMMethod.TETGEN, FEMMethod.GMSH, FEMMethod.VOXEL)
    return (method,)


def _with_tetgen(mesh: TriangleMesh, config: FEMConfig) -> VolumeMesh:
    failures: list[str] = []
    if config.prefer_native_tetgen:
        try:
            return _with_zynmorph_native_tetgen(mesh, config)
        except (ImportError, BackendUnavailableError, GeometryError, RuntimeError, ValueError) as exc:
            failures.append(f"native-zynmorph: {exc}")
    try:
        import tetgen
    except ImportError as exc:
        detail = "; ".join(failures)
        raise BackendUnavailableError(
            "no TetGen backend is available" + (f"; {detail}" if detail else "")
        ) from exc
    generator = tetgen.TetGen(mesh.vertices, mesh.faces)
    result = generator.tetrahedralize(
        order=1,
        mindihedral=config.minimum_dihedral_degrees,
        minratio=config.minimum_radius_edge_ratio,
        maxvolume=_regular_tetra_volume(config.target_edge_length),
        fixedvolume=True,
        opt_iterations=max(0, config.optimization_level),
    )
    if isinstance(result, tuple) and len(result) >= 2:
        nodes, tetrahedra = result[0], result[1]
    elif hasattr(generator, "grid"):
        nodes = np.asarray(generator.grid.points)
        tetrahedra = np.asarray(generator.grid.cells_dict["tetra"])
    else:  # pragma: no cover - API compatibility guard
        raise RuntimeError("unrecognized tetgen return value")
    return VolumeMesh(
        nodes=np.asarray(nodes),
        tetrahedra=np.asarray(tetrahedra),
        cell_regions=np.full(len(tetrahedra), config.region_id, dtype=np.int32),
        region_names={config.region_id: config.region_name},
        metadata={"backend": "tetgen-python", "native_failures": tuple(failures)},
    )


def _with_zynmorph_native_tetgen(mesh: TriangleMesh, config: FEMConfig) -> VolumeMesh:
    from ..zynmorph.freeform import mesh_closed_surface_tetgen
    from ..zynmorph.tetgen import TetGenMeshingConfig, tetgen_native_status

    status = tetgen_native_status()
    if not status.available:
        raise BackendUnavailableError(status.reason or "ZynMorph native TetGen is unavailable")
    seed = _interior_seed(mesh, config.target_edge_length)
    maximum_volume = _regular_tetra_volume(config.target_edge_length)
    result = mesh_closed_surface_tetgen(
        mesh,
        region=config.region_id,
        region_name=config.region_name,
        seed_m_xyz=seed,
        maximum_tetra_volume_m3=maximum_volume,
        config=TetGenMeshingConfig(
            radius_edge_ratio=max(1.000001, config.minimum_radius_edge_ratio),
            minimum_dihedral_degrees=config.minimum_dihedral_degrees,
            optimization_level=config.optimization_level,
            global_maximum_tetra_volume_m3=maximum_volume,
            preserve_boundary_facets=True,
        ),
        maximum_tetrahedra=config.maximum_cells,
    )
    volume = result.mesh
    return VolumeMesh(
        nodes=volume.nodes,
        tetrahedra=volume.tetrahedra,
        cell_regions=volume.cell_regions,
        region_names=volume.region_names,
        metadata={**volume.metadata, "backend": "zynnova-native-tetgen"},
    )


def _regular_tetra_volume(edge_length: float) -> float:
    return float(edge_length**3 / (6.0 * math.sqrt(2.0)))


def _interior_seed(mesh: TriangleMesh, target_edge: float) -> tuple[float, float, float]:
    centre = 0.5 * (np.min(mesh.vertices, axis=0) + np.max(mesh.vertices, axis=0))
    try:
        import trimesh

        surface = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
        candidates = [np.asarray(surface.center_mass, dtype=float), centre]
        for candidate in candidates:
            try:
                if bool(surface.contains(candidate.reshape(1, 3))[0]):
                    return tuple(float(v) for v in candidate)
            except Exception:
                pass
        extent = float(np.max(np.ptp(mesh.vertices, axis=0)))
        pitch = max(min(float(target_edge), extent / 24.0), extent / 128.0)
        filled = surface.voxelized(pitch).fill()
        points = np.asarray(filled.points, dtype=float)
        if len(points):
            index = int(np.argmin(np.linalg.norm(points - centre[None, :], axis=1)))
            return tuple(float(v) for v in points[index])
    except Exception:
        pass
    return tuple(float(v) for v in centre)


def _with_gmsh(mesh: TriangleMesh, config: FEMConfig) -> VolumeMesh:
    try:
        import gmsh
    except ImportError as exc:
        raise BackendUnavailableError("Gmsh Python API is not installed") from exc
    with TemporaryDirectory(prefix="zynnova_gmsh_") as temporary:
        surface_path = Path(temporary) / "surface.stl"
        export_triangle_mesh(surface_path, mesh)
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("zynnova_object")
            gmsh.merge(str(surface_path))
            angle = 40.0 * math.pi / 180.0
            gmsh.model.mesh.classifySurfaces(angle, True, True, math.pi)
            gmsh.model.mesh.createGeometry()
            surfaces = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]
            if not surfaces:
                raise RuntimeError("Gmsh did not recover any surfaces")
            loop = gmsh.model.geo.addSurfaceLoop(surfaces)
            gmsh.model.geo.addVolume([loop])
            gmsh.model.geo.synchronize()
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", config.target_edge_length)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", config.target_edge_length)
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.model.mesh.generate(3)
            node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
            element_tags, element_nodes = gmsh.model.mesh.getElementsByType(4)
            if len(element_tags) == 0:
                raise RuntimeError("Gmsh generated no first-order tetrahedra")
            nodes = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
            tags = np.asarray(node_tags, dtype=np.int64)
            order = np.argsort(tags)
            sorted_tags = tags[order]
            references = np.asarray(element_nodes, dtype=np.int64).reshape(-1, 4)
            positions = np.searchsorted(sorted_tags, references)
            if np.any(sorted_tags[positions] != references):
                raise RuntimeError("Gmsh returned unknown node references")
            tetrahedra = order[positions]
        finally:
            gmsh.finalize()
    return VolumeMesh(
        nodes=nodes,
        tetrahedra=tetrahedra,
        cell_regions=np.full(len(tetrahedra), config.region_id, dtype=np.int32),
        region_names={config.region_id: config.region_name},
        metadata={"backend": "gmsh"},
    )


def _with_voxels(mesh: TriangleMesh, config: FEMConfig) -> VolumeMesh:
    try:
        import trimesh
    except ImportError as exc:
        raise BackendUnavailableError(
            "voxel FEM fallback requires trimesh; install zynnova[zynnova-object]"
        ) from exc
    surface = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        process=False,
    )
    grid = surface.voxelized(config.voxel_pitch).fill()
    matrix_xyz = np.asarray(grid.matrix, dtype=bool)
    if not np.any(matrix_xyz):
        raise RuntimeError("surface voxelization returned an empty solid")
    # trimesh matrix axes are x/y/z; ZynNova labels are z/y/x.
    labels = np.transpose(matrix_xyz.astype(np.int32), (2, 1, 0))
    # ``VoxelGrid.transform`` maps integer matrix indices to voxel centres and is
    # stable across trimesh releases, unlike the removed ``origin`` convenience
    # attribute. ZynNova's structured mesh expects the lower corner of voxel (0,0,0).
    transform = np.asarray(grid.transform, dtype=float)
    pitch_xyz = np.asarray(grid.pitch, dtype=float).reshape(3)
    origin_xyz = transform[:3, 3] - 0.5 * pitch_xyz
    result = voxel_to_tetrahedra(
        labels,
        spacing=(pitch_xyz[2], pitch_xyz[1], pitch_xyz[0]),
        origin=(origin_xyz[2], origin_xyz[1], origin_xyz[0]),
        region_names={1: config.region_name},
    )
    return select_volume_regions(result.volume_mesh, {1})


def _orient_positive(mesh: VolumeMesh) -> VolumeMesh:
    tetrahedra = mesh.tetrahedra.copy()
    signed = tetrahedron_signed_volumes(mesh)
    negative = signed < 0.0
    if np.any(negative):
        first = tetrahedra[negative, 0].copy()
        tetrahedra[negative, 0] = tetrahedra[negative, 1]
        tetrahedra[negative, 1] = first
    return VolumeMesh(
        nodes=mesh.nodes,
        tetrahedra=tetrahedra,
        cell_regions=mesh.cell_regions,
        region_names=mesh.region_names,
        metadata=mesh.metadata,
    )


__all__ = ["FEMMeshingError", "tetrahedralize_surface"]
