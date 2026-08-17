from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..download import DownloadSpec
from ..local_input import require_local_file
from ..record import MaterialSample
from ..registry import DATASETS
from .base import MolecularDatasetSource

HARTREE_TO_EV = 27.211386245988
HARTREE_PER_ANGSTROM_TO_EV_PER_ANGSTROM = HARTREE_TO_EV


@DATASETS.register("ani1x", aliases=("ani_1x",))
class ANI1xDatasetSource(MolecularDatasetSource):
    name = "ani1x"
    homepage = (
        "https://springernature.figshare.com/articles/dataset/"
        "ANI-1x_Dataset_Release/10047041"
    )
    license = "CC0"
    URL = "https://springernature.figshare.com/ndownloader/files/18112775"

    def __init__(
        self,
        root: str | Path,
        *,
        url: str | None = None,
        h5_path: str | Path | None = None,
        local_file: str | Path | None = None,
        local_dir: str | Path | None = None,
        conformations_per_molecule: int | None = None,
        limit: int | None = None,
        convert_to_ev: bool = True,
        **kwargs: Any,
    ) -> None:
        if h5_path is not None and local_file is not None:
            raise ValueError("h5_path and local_file are aliases; pass only one")
        self.url = url or self.URL
        self._h5_path: Path | None = None
        self.conformations_per_molecule = conformations_per_molecule
        self.limit = limit
        self.convert_to_ev = convert_to_ev
        super().__init__(
            root,
            local_file=h5_path or local_file,
            local_dir=local_dir,
            **kwargs,
        )

    @property
    def h5_path(self) -> Path:
        return self._h5_path or (self.raw_dir / "ani1x-release.h5")

    def download(self, *, force: bool = False) -> None:
        local = self.materialize_local_input(
            force=force,
            extract_subdir="ani1x-local",
        )
        if local is not None:
            self._h5_path = require_local_file(
                local,
                ("*.h5", "*.hdf5"),
                description="ANI-1x HDF5 file",
            )
            return
        self.download_manager.fetch(
            DownloadSpec(
                urls=(self.url,),
                filename="ani1x-release.h5",
                archive=None,
                description="ANI-1x HDF5 release",
            ),
            force=force,
        )

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required; install zynnova[data-hdf5]") from exc
        from ...structure import StructureData

        count = 0
        with h5py.File(self.h5_path, "r") as handle:
            for molecule_name, group in handle.items():
                coordinates = _dataset(group, "coordinates")
                species = _dataset(group, "atomic_numbers", "species")
                energies = _dataset(
                    group,
                    "wb97x_dz.energy",
                    "energies",
                    "energy",
                )
                forces = _dataset(
                    group,
                    "wb97x_dz.forces",
                    "forces",
                    required=False,
                )
                if coordinates is None or species is None or energies is None:
                    continue
                atomic_numbers = [_atomic_number(item) for item in species[...]]
                number = coordinates.shape[0]
                if self.conformations_per_molecule is not None:
                    number = min(number, self.conformations_per_molecule)
                for conformation in range(number):
                    if self.limit is not None and count >= self.limit:
                        return
                    energy = float(np.asarray(energies[conformation]).reshape(-1)[0])
                    labels: dict[str, Any] = {
                        "energy": energy * HARTREE_TO_EV if self.convert_to_ev else energy
                    }
                    if forces is not None:
                        force = np.asarray(forces[conformation], dtype=np.float64)
                        labels["forces"] = (
                            force * HARTREE_PER_ANGSTROM_TO_EV_PER_ANGSTROM
                            if self.convert_to_ev
                            else force
                        )
                    yield MaterialSample(
                        id=f"ani1x:{molecule_name}:{conformation}",
                        material_type=self.material_type,
                        structure=StructureData(
                            atomic_numbers=atomic_numbers,
                            positions=np.asarray(coordinates[conformation], dtype=np.float64),
                        ),
                        labels=labels,
                        metadata={"molecule": molecule_name},
                        provenance={"dataset": "ANI-1x", "group": molecule_name},
                    )
                    count += 1


def _dataset(group: Any, *names: str, required: bool = True):
    for name in names:
        if name in group:
            return group[name]
    if required:
        return None
    return None


def _atomic_number(value: Any) -> int:
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, (np.integer, int)):
        return int(value)
    try:
        from ase.data import atomic_numbers
    except ImportError as exc:
        raise ImportError("ASE is required to parse ANI element labels") from exc
    return int(atomic_numbers[str(value)])
