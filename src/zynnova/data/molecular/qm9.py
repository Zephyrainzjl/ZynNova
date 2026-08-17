from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ...structure import StructureData
from ..adapters import pyg_atoms_to_structure
from ..local_input import find_local_files, link_or_copy_directory
from ..record import MaterialSample
from ..registry import DATASETS
from .base import MolecularDatasetSource

QM9_TARGETS = (
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "u0",
    "u",
    "h",
    "g",
    "cv",
    "u0_atom",
    "u_atom",
    "h_atom",
    "g_atom",
    "a",
    "b",
    "c",
)


@DATASETS.register("qm9")
class QM9DatasetSource(MolecularDatasetSource):
    name = "qm9"
    homepage = "http://quantum-machine.org/datasets/"

    def __init__(
        self,
        root: str | Path,
        *,
        targets: tuple[str, ...] | None = None,
        limit: int | None = None,
        local_file: str | Path | None = None,
        local_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.targets = tuple(targets or QM9_TARGETS)
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
            extract_subdir="qm9-local",
        )
        if local is not None:
            self._local_root = local

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        del split
        if self._local_root is not None and self._local_root.is_file():
            if self._local_root.suffix.lower() == ".npz":
                yield from self._iter_npz(self._local_root)
                return
        dataset = self._pyg_dataset()
        for index, data in enumerate(dataset):
            if self.limit is not None and index >= self.limit:
                break
            target_values = data.y.reshape(-1).detach().cpu().tolist()
            labels = {
                name: target_values[target_index]
                for target_index, name in enumerate(QM9_TARGETS)
                if name in self.targets and target_index < len(target_values)
            }
            smiles = getattr(data, "smiles", None)
            yield MaterialSample(
                id=f"qm9:{index}",
                material_type=self.material_type,
                structure=pyg_atoms_to_structure(data),
                labels=labels,
                metadata={"smiles": smiles} if smiles else {},
                provenance={"dataset": "QM9", "index": index},
            )

    def _pyg_dataset(self):
        try:
            from torch_geometric.datasets import QM9
        except ImportError as exc:
            raise ImportError("torch-geometric is required; install zynnova[graph]") from exc
        if self._local_root is None:
            return QM9(str(self.raw_dir))
        pyg_root = _prepare_pyg_root(self._local_root, self.raw_dir)
        return QM9(str(pyg_root))

    def _iter_npz(self, path: Path) -> Iterator[MaterialSample]:
        with np.load(path, allow_pickle=True) as payload:
            z_values = _npz_value(payload, "z", "atomic_numbers")
            positions = _npz_value(payload, "pos", "positions", "coordinates")
            targets = _npz_value(payload, "y", "targets", required=False)
            num_atoms = _npz_value(payload, "num_atoms", "natoms", required=False)
            smiles = _npz_value(payload, "smiles", required=False)
            count = len(z_values)
            if self.limit is not None:
                count = min(count, self.limit)
            for index in range(count):
                z = np.asarray(z_values[index], dtype=np.int64).reshape(-1)
                pos = np.asarray(positions[index], dtype=np.float64).reshape(-1, 3)
                if num_atoms is not None:
                    n_atoms = int(np.asarray(num_atoms[index]).reshape(-1)[0])
                    z = z[:n_atoms]
                    pos = pos[:n_atoms]
                else:
                    keep = z > 0
                    if keep.any() and len(z) == len(pos):
                        z = z[keep]
                        pos = pos[keep]
                row_targets = (
                    []
                    if targets is None
                    else np.asarray(targets[index]).reshape(-1).tolist()
                )
                labels = {
                    name: row_targets[target_index]
                    for target_index, name in enumerate(QM9_TARGETS)
                    if name in self.targets and target_index < len(row_targets)
                }
                smile = None if smiles is None else str(smiles[index])
                yield MaterialSample(
                    id=f"qm9:{index}",
                    material_type=self.material_type,
                    structure=StructureData(z, pos, source=str(path)),
                    labels=labels,
                    metadata={"smiles": smile} if smile else {},
                    provenance={"dataset": "QM9-local-NPZ", "index": index},
                )


def _prepare_pyg_root(source: Path, cache_root: Path) -> Path:
    source = source.resolve()
    if source.is_dir():
        if (source / "raw").is_dir() or (source / "processed").is_dir():
            return source
        if source.name in {"raw", "processed"}:
            return source.parent
        processed = find_local_files(source, ("data_v3.pt", "data.pt"))
        for path in processed:
            if path.parent.name == "processed":
                return path.parent.parent
        raw_candidates = find_local_files(source, ("gdb9.sdf",))
        if raw_candidates:
            raw_source = raw_candidates[0].parent
            required = ("gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt")
            missing = [name for name in required if not (raw_source / name).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"incomplete local QM9 raw directory {raw_source}; missing {missing}"
                )
            root = cache_root / "qm9-local-pyg"
            link_or_copy_directory(raw_source, root / "raw")
            return root
    if source.is_file() and source.suffix.lower() in {".pt", ".pth"}:
        root = cache_root / "qm9-local-pyg"
        processed = root / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        target = processed / "data_v3.pt"
        if not target.exists():
            target.write_bytes(source.read_bytes())
        return root
    raise FileNotFoundError(
        "QM9 local input must be a normalized NPZ, a PyG root with raw/processed, "
        "or an extracted directory containing gdb9.sdf, gdb9.sdf.csv, and "
        "uncharacterized.txt"
    )


def _npz_value(payload: Any, *names: str, required: bool = True):
    for name in names:
        if name in payload.files:
            return payload[name]
    if required:
        raise ValueError(f"QM9 NPZ is missing one of the required keys: {names}")
    return None
