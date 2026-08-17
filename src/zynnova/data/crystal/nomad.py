from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..local_input import iter_records
from ..record import MaterialSample
from ..registry import DATASETS
from .base import CrystalDatasetSource


@DATASETS.register("nomad_archive", aliases=("nomad",))
class NomadArchiveDatasetSource(CrystalDatasetSource):
    """NOMAD v1 archive-query adapter with configurable required fields."""

    name = "nomad_archive"
    homepage = "https://nomad-lab.eu/"

    def __init__(
        self,
        root: str | Path,
        *,
        query: Mapping[str, Any] | None = None,
        required: Mapping[str, Any] | str = "*",
        base_url: str = "https://nomad-lab.eu/prod/v1/api/v1",
        page_size: int = 100,
        max_entries: int | None = None,
        token: str | None = None,
        local_file: str | Path | None = None,
        local_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.query = dict(query or {})
        self.required = required
        self.base_url = base_url.rstrip("/")
        self.page_size = page_size
        self.max_entries = max_entries
        self.token = token
        self._local_root: Path | None = None
        super().__init__(
            root,
            local_file=local_file,
            local_dir=local_dir,
            **kwargs,
        )

    @property
    def raw_file(self) -> Path:
        return self.raw_dir / "nomad_archives.jsonl"

    def download(self, *, force: bool = False) -> None:
        local = self.materialize_local_input(
            force=force,
            extract_subdir="nomad-local",
        )
        if local is not None:
            self._local_root = local
            return
        if self.raw_file.exists() and not force:
            return
        endpoint = f"{self.base_url}/entries/archive/query"
        page_after: str | None = None
        count = 0
        with self.raw_file.open("w", encoding="utf-8") as handle:
            while True:
                body: dict[str, Any] = {
                    "query": self.query,
                    "required": self.required,
                    "pagination": {"page_size": self.page_size},
                }
                if page_after:
                    body["pagination"]["page_after_value"] = page_after
                request = urllib.request.Request(
                    endpoint,
                    data=json.dumps(body).encode(),
                    headers={
                        "Content-Type": "application/json",
                        **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.load(response)
                entries = payload.get("data", [])
                for entry in entries:
                    handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                    count += 1
                    if self.max_entries is not None and count >= self.max_entries:
                        return
                pagination = payload.get("pagination", {})
                page_after = pagination.get("next_page_after_value")
                if not entries or not page_after:
                    return

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        from ...structure import StructureData

        source = self._local_root or self.raw_file
        for index, payload in enumerate(iter_records(source)):
                archive = payload.get("archive", payload)
                system = _last_system(archive)
                atoms = system.get("atoms", {})
                labels = atoms.get("labels") or atoms.get("species")
                positions = _quantity(atoms.get("positions"), target="angstrom")
                lattice = _quantity(atoms.get("lattice_vectors"), target="angstrom")
                if labels is None or positions is None:
                    continue
                atomic_numbers = [_atomic_number(label) for label in labels]
                periodic = atoms.get("periodic", [True, True, True])
                results = archive.get("results", payload.get("results", {}))
                properties = _flatten_scalars(results)
                entry_id = str(payload.get("entry_id", archive.get("entry_id", index)))
                yield MaterialSample(
                    id=entry_id,
                    material_type=self.material_type,
                    structure=StructureData(
                        atomic_numbers=atomic_numbers,
                        positions=positions,
                        cell=lattice if lattice is not None else np.zeros((3, 3)),
                        pbc=periodic,
                    ),
                    labels=properties,
                    metadata={"mainfile": payload.get("mainfile")},
                    provenance={"dataset": "NOMAD", "entry_id": entry_id},
                )


def _last_system(archive: dict[str, Any]) -> dict[str, Any]:
    runs = archive.get("run", [])
    if not runs:
        return archive.get("system", {})
    systems = runs[-1].get("system", [])
    return systems[-1] if systems else {}


def _quantity(value: Any, *, target: str) -> np.ndarray | None:
    if value is None:
        return None
    unit = None
    if isinstance(value, dict) and "magnitude" in value:
        unit = value.get("unit") or value.get("units")
        value = value["magnitude"]
    array = np.asarray(value, dtype=np.float64)
    if target == "angstrom" and unit in {"meter", "m"}:
        array = array * 1.0e10
    return array


def _atomic_number(symbol: Any) -> int:
    if isinstance(symbol, (int, np.integer)):
        return int(symbol)
    try:
        from ase.data import atomic_numbers
    except ImportError as exc:
        raise ImportError("ASE is required to parse NOMAD element labels") from exc
    return int(atomic_numbers[symbol])


def _flatten_scalars(mapping: Any, prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    if not isinstance(mapping, dict):
        return output
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            output.update(_flatten_scalars(value, path))
        elif isinstance(value, (int, float, bool)):
            output[path] = value
    return output
