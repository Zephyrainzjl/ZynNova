from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..record import MaterialSample, MaterialType
from ..source import DatasetSource
from ..registry import DATASETS


@DATASETS.register("trajectory", aliases=("ase_trajectory",))
class TrajectoryDatasetSource(DatasetSource):
    """Turn ASE trajectories into energy/force/stress potential-training samples."""

    name = "trajectory"

    def __init__(
        self,
        path: str | Path,
        *,
        material_type: MaterialType | str = MaterialType.SPECIAL,
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str | None = "stress",
        stride: int = 1,
        start: int | None = None,
        stop: int | None = None,
        root: str | Path | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.material_type = MaterialType(material_type)
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.stress_key = stress_key
        self.stride = int(stride)
        self.start = start
        self.stop = stop
        super().__init__(root or self.path.parent, download=False, prepare=False)

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        try:
            from ase.io import iread
        except ImportError as exc:
            raise ImportError("ASE is required; install zynnova[io]") from exc
        from ...structure import StructureData

        for frame_index, atoms in enumerate(iread(self.path, index=":")):
            if self.start is not None and frame_index < self.start:
                continue
            if self.stop is not None and frame_index >= self.stop:
                break
            if frame_index % self.stride:
                continue
            labels: dict[str, Any] = {}
            energy = atoms.info.get(self.energy_key)
            if energy is None and atoms.calc is not None:
                try:
                    energy = atoms.get_potential_energy()
                except Exception:
                    energy = None
            if energy is not None:
                labels["energy"] = float(energy)
            forces = atoms.arrays.get(self.forces_key)
            if forces is None and atoms.calc is not None:
                try:
                    forces = atoms.get_forces()
                except Exception:
                    forces = None
            if forces is not None:
                labels["forces"] = np.asarray(forces, dtype=np.float64)
            if self.stress_key:
                stress = atoms.info.get(self.stress_key)
                if stress is not None:
                    labels["stress"] = np.asarray(stress, dtype=np.float64)
            sample = MaterialSample(
                id=f"{self.path.stem}:{frame_index}",
                material_type=self.material_type,
                structure=StructureData.from_ase(atoms, source=str(self.path)),
                labels=labels,
                metadata={"frame_index": frame_index},
                provenance={"dataset": self.path.name, "source": str(self.path)},
            )
            if split is None or sample.split == split:
                yield sample
