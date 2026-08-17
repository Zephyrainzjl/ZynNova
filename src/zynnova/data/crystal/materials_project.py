from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..adapters import as_structure_data
from ..local_input import iter_records
from ..record import MaterialSample
from ..registry import DATASETS
from .base import CrystalDatasetSource


@DATASETS.register("materials_project", aliases=("mp", "materialsproject"))
class MaterialsProjectDatasetSource(CrystalDatasetSource):
    """Materials Project summary adapter with API and offline export modes."""

    name = "materials_project"
    homepage = "https://materialsproject.org/"

    DEFAULT_FIELDS = (
        "material_id",
        "formula_pretty",
        "structure",
        "band_gap",
        "energy_above_hull",
        "formation_energy_per_atom",
        "is_stable",
        "volume",
        "density",
        "symmetry",
    )

    def __init__(
        self,
        root: str | Path,
        *,
        query: Mapping[str, Any] | None = None,
        fields: Sequence[str] | None = None,
        api_key: str | None = None,
        limit: int | None = None,
        local_file: str | Path | None = None,
        local_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.query = dict(query or {})
        self.fields = tuple(fields or self.DEFAULT_FIELDS)
        self.api_key = api_key or os.getenv("MP_API_KEY")
        self.limit = limit
        self._local_root: Path | None = None
        super().__init__(
            root,
            local_file=local_file,
            local_dir=local_dir,
            **kwargs,
        )

    @property
    def raw_file(self) -> Path:
        return self.raw_dir / "materials_project.jsonl"

    def download(self, *, force: bool = False) -> None:
        local = self.materialize_local_input(
            force=force,
            extract_subdir="materials-project-local",
        )
        if local is not None:
            self._local_root = local
            return
        if self.raw_file.exists() and not force:
            return
        if not self.api_key:
            raise ValueError(
                "Materials Project requires api_key/MP_API_KEY for online mode, "
                "or local_file/local_dir for offline mode"
            )
        try:
            from mp_api.client import MPRester
        except ImportError as exc:
            raise ImportError("mp-api is required; install zynnova[data-crystal]") from exc
        self.raw_file.parent.mkdir(parents=True, exist_ok=True)
        with self.raw_file.open("w", encoding="utf-8") as handle, MPRester(
            self.api_key
        ) as mpr:
            documents = mpr.materials.summary.search(fields=list(self.fields), **self.query)
            for index, document in enumerate(documents):
                if self.limit is not None and index >= self.limit:
                    break
                payload = (
                    document.model_dump(mode="json")
                    if hasattr(document, "model_dump")
                    else document.dict()
                )
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _payloads(self) -> Iterator[dict[str, Any]]:
        source = self._local_root or self.raw_file
        yield from iter_records(source)

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        del split
        for index, record in enumerate(self._payloads()):
            if self.limit is not None and index >= self.limit:
                break
            payload = dict(record)
            structure_payload = payload.pop("structure", None)
            if structure_payload is None:
                continue
            structure = _decode_structure(structure_payload)
            material_id = str(payload.pop("material_id", payload.pop("id", index)))
            formula = payload.pop("formula_pretty", payload.pop("formula", None))
            labels = {
                key: value
                for key, value in payload.items()
                if isinstance(value, (int, float, bool)) or value is None
            }
            metadata = {
                "formula": formula,
                **{key: value for key, value in payload.items() if key not in labels},
            }
            yield MaterialSample(
                id=material_id,
                material_type=self.material_type,
                structure=structure,
                labels=labels,
                metadata=metadata,
                provenance={"dataset": "Materials Project", "material_id": material_id},
            )


def _decode_structure(value: Any):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{"):
            value = json.loads(stripped)
    if isinstance(value, dict):
        try:
            return as_structure_data(value)
        except (TypeError, ValueError, FileNotFoundError):
            try:
                from pymatgen.core import Structure
            except ImportError as exc:
                raise ImportError(
                    "pymatgen is required to parse a Materials Project structure dictionary"
                ) from exc
            return as_structure_data(Structure.from_dict(value))
    return as_structure_data(value, kind="crystal")
