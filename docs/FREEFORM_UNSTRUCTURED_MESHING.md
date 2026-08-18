# ZynMorph Free-form Unstructured Tet4 Meshing

## Why this exists

The structured voxel path is intentionally limited: a labeled array has a rectangular outer envelope, and the compatibility mesher splits each voxel into six tetrahedra. That is useful for debugging but too restrictive for curved particles, pores, CAD-like objects, hollow geometries, reconstructed surfaces, and spatially graded FEM meshes.

The free-form path removes the rectangular assumption. Geometry is represented as one or more **closed triangular shells**. Each shell separates an inside region from an outside region. The outer shell uses `outside_region=None`; internal shells can represent inclusions, pores-as-materials, coatings, cracks represented as finite-thickness regions, or nested layers.

```python
from zynnova.zynmorph import (
    SurfaceShell,
    FreeformRegion,
    mesh_freeform_geometry,
    TetGenMeshingConfig,
)

result = mesh_freeform_geometry(
    shells,
    regions,
    tetgen_config=TetGenMeshingConfig(...),
)
```

The production backend is the compiled TetGen 1.6 C++ core. No structured fallback occurs if the native extension is unavailable.

## Supported geometry sources

`load_surface_shell()` accepts the common triangle formats already supported by `zynnova.geometry.load_triangle_mesh`: OBJ, PLY, STL, NPZ, plus formats handled by the optional trimesh backend.

The shell itself can also be created in Python from arbitrary vertices and triangles. Therefore the geometry can come from image reconstruction, marching cubes, CAD triangulation, tomography, procedural geometry, or an external mesher.

## Multi-domain PLC

```python
outer = SurfaceShell(
    outer_surface,
    inside_region=1,
    outside_region=None,
    name="outer_boundary",
)

particle = SurfaceShell(
    particle_surface,
    inside_region=2,
    outside_region=1,
    name="particle_matrix_interface",
)

coating = SurfaceShell(
    coating_surface,
    inside_region=3,
    outside_region=2,
    name="coating_particle_interface",
)
```

`assemble_freeform_plc()`:

- checks every shell is closed and 2-manifold;
- repairs inconsistent local triangle orientation by graph propagation;
- orients each closed component outward using signed enclosed volume;
- welds coincident vertices within a scale-aware tolerance;
- preserves one facet marker per interface shell;
- records the material region on both sides of every interface;
- runs the same multi-region PLC audit used by the voxel-derived TetGen path.

Self-intersecting shells or overlapping duplicate interfaces are not silently repaired; they should be fixed at the geometry stage or rejected by TetGen consistency checks.

## Region seeds and sizing

Every meshed region has an explicit interior seed:

```python
regions = (
    FreeformRegion(1, (x1, y1, z1), "matrix", maximum_tetra_volume_m3=...),
    FreeformRegion(2, (x2, y2, z2), "particle", maximum_tetra_volume_m3=...),
)
```

Sizing can combine:

- global maximum tetrahedron volume;
- per-region maximum tetrahedron volume;
- per-interface maximum triangle area;
- spherical local refinement zones;
- TetGen radius-edge quality control;
- minimum dihedral target;
- TetGen optimization and consistency checks.

This creates genuinely unstructured, spatially graded Delaunay Tet4 meshes rather than six repeated tetrahedra per voxel.

## Voids

A closed shell can identify a void region. Supply its region ID through `void_regions` and provide a TetGen hole point inside the void:

```python
result = mesh_freeform_geometry(
    shells,
    meshed_regions,
    void_regions=(99,),
    holes_m_xyz=((xh, yh, zh),),
)
```

The void is removed from the tetrahedralization instead of receiving material Tet4 cells.

## Reference-style meshing from an existing COMSOL MPHTXT

```python
from zynnova.zynmorph import (
    load_comsol_tet4_mphtxt,
    profile_reference_mesh,
    tetgen_config_from_reference,
)

reference = load_comsol_tet4_mphtxt("TetMesh-cell_nmc_grp.mphtxt")
profile = profile_reference_mesh(reference)
config = tetgen_config_from_reference(
    profile,
    region_map={1: 1, 2: 4, 3: 2},
    volume_quantile=0.95,
)
```

The profile records global and per-domain edge-length and tetra-volume distributions. The target geometry may be completely different; only the mesh scale/style is transferred. `linear_scale` scales target edge lengths, with volume constraints scaled cubically.

For the supplied `TetMesh-cell_nmc_grp.mphtxt` reference, the parser finds 3,606 nodes, 18,532 positive Tet4 cells, four domains, and a broad nonuniform edge-length distribution. This makes it a useful sizing reference rather than a geometry template.

## COMSOL

The resulting `VolumeMesh` can use the existing Tet4 MPHTXT exporter directly:

```python
from zynnova.zynmorph import export_fem_mesh

result = export_fem_mesh(
    result,
    "outputs/freeform",
    formats=("mphtxt", "vtk", "msh", "inp"),
    comsol_options={
        "include_default_battery_selections": False,
        "include_default_boundary_unions": False,
    },
)
```

The COMSOL exporter receives an already unstructured Tet4 mesh, so there is no Hex8 orientation issue and no rectangular-domain assumption.

## One-call reference-style meshing

When the target should look statistically like an existing unstructured Tet4
mesh but use a completely different outer shape, use the convenience wrapper:

```python
from zynnova.zynmorph import mesh_freeform_like_reference

fem = mesh_freeform_like_reference(
    shells,
    regions,
    "TetMesh-cell_nmc_grp.mphtxt",
    region_map={1: 3, 2: 1, 3: 2},
    volume_quantile=0.95,
    linear_scale=0.85,
    tetgen_config=base_config,
    void_regions=(99,),
    holes_m_xyz=((xh, yh, zh),),
)
```

The reference contributes per-domain tetrahedron-volume statistics only.  Its
rectangular bounding box is not copied, and the generated target mesh records
`rectangular_domain_assumed=False`.

## Existing Tet4 meshes as geometry diagnostics

`plc_from_volume_mesh(mesh)` reconstructs the exterior and all material
interfaces from an existing conforming Tet4 volume mesh.  Strict mode requires
each recovered material shell to pass the manifold PLC gate.  Some valid legacy
FEM volume meshes contain *pinched* material interfaces where several interface
triangles meet on one edge; these are valid as volume meshes but are not safe to
silently pass back to TetGen as a strict shell PLC.  For inspection only:

```python
plc = plc_from_volume_mesh(mesh, strict=False)
print(plc.metadata["plc_audit"])
```

This distinction is deliberate: reference mesh *sizing* is always safe, while
reference mesh *geometry remeshing* is admitted only when the recovered PLC is
strictly valid.
