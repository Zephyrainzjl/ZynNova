"""End-to-end conditional generation, validation, FEM meshing, and export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..core.artifacts import RunManifest
from ..core.serialization import dump_json, to_jsonable
from .generation import GenerationResult, enforce_generation_constraints
from .meshing import FEMMeshResult, export_fem_mesh, mesh_microstructure
from .metrics import MicrostructureMetrics, analyze_microstructure
from .registry import BACKENDS
from .schema import GenerationConfig, MicrostructureCondition


@dataclass(frozen=True, slots=True)
class ZynMorphRun:
    directory: Path
    generation: GenerationResult
    metrics: MicrostructureMetrics
    fem: FEMMeshResult
    manifest: Path
    artifacts: Mapping[str, Path]


def run_zynmorph(
    condition: MicrostructureCondition,
    config: GenerationConfig | None = None,
    *,
    backend_options: Mapping[str, object] | None = None,
) -> ZynMorphRun:
    resolved = config or GenerationConfig()
    options = dict(backend_options or {})
    backend = BACKENDS.choose(resolved.backend, **options)
    manifest = RunManifest(
        workflow="zynnova.zynmorph.generate",
        backend=backend.name,
        configuration={"condition": to_jsonable(condition), "config": to_jsonable(resolved)},
    )
    root = Path(resolved.output_directory).resolve() / manifest.run_id
    root.mkdir(parents=True, exist_ok=False)
    try:
        manifest.event("generation_started")
        generation = backend.generate(
            condition,
            refinement_steps=resolved.refinement_steps,
            temperature=resolved.temperature,
        )
        generation = enforce_generation_constraints(
            generation,
            condition,
            preserve_exact_fractions=resolved.preserve_exact_fractions,
        )
        volume_exports = generation.volume.export(
            root / "volume",
            resolved.export_volume_formats,
        )
        metrics = analyze_microstructure(generation.volume)
        metrics_path = dump_json(root / "metrics.json", metrics)
        condition_path = dump_json(root / "condition.json", condition)
        generation_path = dump_json(root / "generation.json", generation)
        manifest.event(
            "generation_completed",
            achieved_counts=generation.achieved_counts,
            percolation=generation.metadata.get("percolation_repairs", []),
        )
        manifest.event("meshing_started", voxels=generation.volume.labels.size)
        fem = mesh_microstructure(
            generation.volume,
            maximum_tetrahedra=resolved.maximum_tetrahedra,
        )
        fem = export_fem_mesh(fem, root / "mesh", formats=resolved.export_mesh_formats)
        quality_path = dump_json(root / "mesh" / "quality.json", fem.quality)
        artifacts: dict[str, Path] = {
            **{f"volume-{key}": value for key, value in volume_exports.items()},
            "metrics": metrics_path,
            "condition": condition_path,
            "generation": generation_path,
            "mesh-quality": quality_path,
            **{f"mesh-{key}": value for key, value in fem.exports.items()},
        }
        media_types = {
            ".json": "application/json",
            ".npz": "application/x-numpy",
            ".npy": "application/x-numpy",
            ".raw": "application/octet-stream",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
            ".vtk": "model/vnd.vtk",
            ".msh": "application/x-gmsh",
            ".inp": "text/plain",
            ".ply": "model/ply",
            ".stl": "model/stl",
        }
        for role, path in artifacts.items():
            manifest.add_artifact(
                path,
                role=role,
                media_type=media_types.get(path.suffix.lower(), "application/octet-stream"),
            )
        manifest.event(
            "meshing_completed",
            tetrahedra=int(fem.mesh.tetrahedra.shape[0]),
            inverted=int(fem.quality.inverted_cells),
            degenerate=int(fem.quality.degenerate_cells),
        )
        manifest.finish()
    except Exception as exc:
        manifest.event("error", type=type(exc).__name__, message=str(exc))
        manifest.finish(status="failed")
        manifest.save(root / "manifest.json")
        raise
    manifest_path = manifest.save(root / "manifest.json")
    return ZynMorphRun(
        directory=root,
        generation=generation,
        metrics=metrics,
        fem=fem,
        manifest=manifest_path,
        artifacts={**artifacts, "manifest": manifest_path},
    )


__all__ = ["ZynMorphRun", "run_zynmorph"]
