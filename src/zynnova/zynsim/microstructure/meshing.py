"""Quality-gated, interface-conforming meshing for generated microstructures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..core import (
    VoxelFEMReconstructionConfig,
    VoxelFEMReconstructionResult,
    reconstruct_voxel_fem_mesh,
)
from .generation import GeneratedMicrostructure
from .schema import DEFAULT_PHASE_NAMES, ManufacturingProcessControl, validate_phase_labels


@dataclass(frozen=True, slots=True)
class InterfaceConformityReport:
    shared_face_pairs: int
    orphan_interface_faces: int
    nonmanifold_interface_faces: int
    conforming: bool


@dataclass(frozen=True, slots=True)
class SimulationReadyMicrostructure:
    generated: GeneratedMicrostructure
    reconstruction: VoxelFEMReconstructionResult
    interface_conformity: InterfaceConformityReport

    @property
    def mesh(self):
        return self.reconstruction.mesh


class ConformalMicrostructureMesher:
    def __init__(
        self,
        *,
        minimum_mean_ratio: float = 0.20,
        maximum_region_volume_change: float = 0.03,
        maximum_tetrahedra: int = 20_000_000,
    ) -> None:
        self.minimum_mean_ratio = float(minimum_mean_ratio)
        self.maximum_region_volume_change = float(maximum_region_volume_change)
        self.maximum_tetrahedra = int(maximum_tetrahedra)

    def mesh(
        self,
        microstructure: GeneratedMicrostructure | np.ndarray,
        *,
        voxel_size_m: float | tuple[float, float, float],
        process: ManufacturingProcessControl | None = None,
        region_names: Mapping[int, str] | None = None,
    ) -> SimulationReadyMicrostructure:
        if isinstance(microstructure, GeneratedMicrostructure):
            generated = microstructure
            resolved_process = process or generated.process
        else:
            labels = validate_phase_labels(microstructure)
            resolved_process = process or ManufacturingProcessControl()
            unique, counts = np.unique(labels, return_counts=True)
            generated = GeneratedMicrostructure(
                labels,
                "external-labels",
                resolved_process,
                tuple(map(int, labels.shape)),
                {int(value): float(count / labels.size) for value, count in zip(unique, counts, strict=True)},
                {},
            )
        config = VoxelFEMReconstructionConfig(
            smoothing=resolved_process.smoothing,
            refinement=resolved_process.refinement,
            minimum_mean_ratio=self.minimum_mean_ratio,
            maximum_region_volume_change=self.maximum_region_volume_change,
            maximum_tetrahedra=self.maximum_tetrahedra,
            quality_failure="raise",
            validate_manifold=True,
        )
        available_labels = {int(value) for value in np.unique(generated.phase_labels)}
        supplied_names = dict(
            DEFAULT_PHASE_NAMES if region_names is None else region_names
        )
        active_region_names = {
            int(label): str(name)
            for label, name in supplied_names.items()
            if int(label) in available_labels
        }
        reconstruction = reconstruct_voxel_fem_mesh(
            generated.phase_labels,
            voxel_size_m=voxel_size_m,
            region_names=active_region_names,
            config=config,
        )
        conformity = audit_interface_conformity(reconstruction)
        if not conformity.conforming:
            raise ValueError("reconstructed mesh failed interface-conformity audit")
        return SimulationReadyMicrostructure(generated, reconstruction, conformity)


def audit_interface_conformity(
    reconstruction: VoxelFEMReconstructionResult,
) -> InterfaceConformityReport:
    cells = reconstruction.mesh.cells
    regions = reconstruction.mesh.cell_regions
    face_owner: dict[tuple[int, int, int], list[int]] = {}
    local_faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    for cell_index, cell in enumerate(cells):
        for local in local_faces:
            face = tuple(sorted(int(cell[index]) for index in local))
            face_owner.setdefault(face, []).append(cell_index)
    shared_pairs = 0
    nonmanifold = 0
    actual_interfaces: set[tuple[int, int, int]] = set()
    for face, owners in face_owner.items():
        if len(owners) > 2:
            nonmanifold += 1
        if len(owners) == 2 and regions is not None and regions[owners[0]] != regions[owners[1]]:
            shared_pairs += 1
            actual_interfaces.add(face)
    expected = {
        tuple(sorted(map(int, face)))
        for faces in reconstruction.interface_faces.values()
        for face in np.asarray(faces)
    }
    orphan = len(expected - actual_interfaces)
    return InterfaceConformityReport(
        shared_face_pairs=shared_pairs,
        orphan_interface_faces=orphan,
        nonmanifold_interface_faces=nonmanifold,
        conforming=orphan == 0 and nonmanifold == 0 and reconstruction.report.fem_ready,
    )


__all__ = [
    "ConformalMicrostructureMesher",
    "InterfaceConformityReport",
    "SimulationReadyMicrostructure",
    "audit_interface_conformity",
]
