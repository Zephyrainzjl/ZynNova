from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ...structure import StructureData
from ..adapters import as_structure_data
from ..local_input import iter_records
from ..record import MaterialSample
from ..registry import DATASETS
from .base import CrystalDatasetSource


@DATASETS.register("jarvis_dft", aliases=("jarvis",))
class JarvisDFTDatasetSource(CrystalDatasetSource):
    name = "jarvis_dft"
    homepage = "https://jarvis.nist.gov/"

    def __init__(
        self,
        root: str | Path,
        *,
        dataset: str = "dft_3d",
        targets: Sequence[str] | None = None,
        limit: int | None = None,
        local_file: str | Path | None = None,
        local_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.dataset = dataset
        self.targets = tuple(targets or ())
        self.limit = limit
        self._local_root: Path | None = None
        super().__init__(
            root,
            local_file=local_file,
            local_dir=local_dir,
            **kwargs,
        )

    def download(self, *, force: bool = False) -> None:
        local = self.materialize_local_input(
            force=force,
            extract_subdir="jarvis-local",
        )
        if local is not None:
            self._local_root = local

    def _records(self) -> Iterator[dict[str, Any]]:
        if self._local_root is not None:
            yield from iter_records(self._local_root)
            return
        try:
            from jarvis.db.figshare import data
        except ImportError as exc:
            raise ImportError(
                "jarvis-tools is required for automatic JARVIS download; "
                "install zynnova[data-crystal] or pass local_file/local_dir"
            ) from exc
        yield from data(self.dataset)

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        del split
        for index, record in enumerate(self._records()):
            if self.limit is not None and index >= self.limit:
                break
            atoms_payload = record.get("atoms") or record.get("structure")
            if atoms_payload is None:
                continue
            structure = _jarvis_structure(atoms_payload)
            identifier = str(record.get("jid", record.get("id", index)))
            target_names = self.targets or tuple(
                key
                for key, value in record.items()
                if key not in {"atoms", "structure", "jid", "id"}
                and isinstance(value, (int, float, bool))
            )
            labels = {name: record.get(name) for name in target_names}
            metadata = {
                key: value
                for key, value in record.items()
                if key not in set(target_names) | {"atoms", "structure"}
            }
            yield MaterialSample(
                id=identifier,
                material_type=self.material_type,
                structure=structure,
                labels=labels,
                metadata=metadata,
                provenance={"dataset": f"JARVIS-{self.dataset}", "jid": identifier},
            )


def _jarvis_structure(payload: Any) -> StructureData:
    if isinstance(payload, str) and payload.strip().startswith("{"):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        return as_structure_data(payload)
    try:
        return as_structure_data(payload)
    except (TypeError, ValueError, FileNotFoundError):
        pass

    elements = payload.get("elements") or payload.get("species")
    coordinates = payload.get("coords") or payload.get("positions")
    lattice = payload.get("lattice_mat") or payload.get("lattice") or np.zeros((3, 3))
    if elements is None or coordinates is None:
        try:
            from jarvis.core.atoms import Atoms
        except ImportError as exc:
            raise ValueError(
                "local JARVIS atoms payload must contain elements and coords, "
                "or jarvis-tools must be installed"
            ) from exc
        return as_structure_data(Atoms.from_dict(payload))

    atomic_numbers = [_atomic_number(value) for value in elements]
    cell = np.asarray(lattice, dtype=np.float64)
    positions = np.asarray(coordinates, dtype=np.float64)
    cartesian = bool(payload.get("cartesian", True))
    if not cartesian:
        positions = positions @ cell
    return StructureData(
        atomic_numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=[True, True, True],
    )


def _atomic_number(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        from ase.data import atomic_numbers
    except ImportError as exc:
        raise ImportError("ASE is required to parse element symbols") from exc
    return int(atomic_numbers[str(value)])
