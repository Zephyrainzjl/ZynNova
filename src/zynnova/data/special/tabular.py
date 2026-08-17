from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..adapters import as_structure_data, smiles_to_structure
from ..record import MaterialSample, MaterialType
from ..source import DatasetSource
from ..registry import DATASETS


@dataclass(slots=True)
class ColumnMapping:
    id: str | None = None
    structure: str | None = None
    smiles: str | None = None
    split: str | None = None
    features: Mapping[str, str] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)
    conditions: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)


@DATASETS.register("tabular", aliases=("csv", "jsonl", "parquet"))
class TabularDatasetSource(DatasetSource):
    """Highly configurable CSV/JSONL/Parquet adapter for special datasets."""

    name = "tabular"

    def __init__(
        self,
        path: str | Path,
        *,
        material_type: MaterialType | str = MaterialType.SPECIAL,
        columns: ColumnMapping | None = None,
        row_converter: Callable[[dict[str, Any], int], MaterialSample] | None = None,
        root: str | Path | None = None,
        structure_kind: str | None = None,
        rdkit_embed: bool = True,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.material_type = MaterialType(material_type)
        self.columns = columns or ColumnMapping()
        self.row_converter = row_converter
        self.structure_kind = structure_kind
        self.rdkit_embed = rdkit_embed
        super().__init__(root or self.path.parent, download=False, prepare=False)

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        for index, row in enumerate(_iter_rows(self.path)):
            sample = (
                self.row_converter(row, index)
                if self.row_converter is not None
                else self._convert_row(row, index)
            )
            if split is None or sample.split == split:
                yield sample

    def _convert_row(self, row: dict[str, Any], index: int) -> MaterialSample:
        structure = None
        if self.columns.structure and row.get(self.columns.structure) not in {None, ""}:
            value = row[self.columns.structure]
            candidate = Path(str(value))
            if not candidate.is_absolute():
                candidate = self.path.parent / candidate
            structure = as_structure_data(candidate, kind=self.structure_kind)
        elif self.columns.smiles and row.get(self.columns.smiles):
            structure = smiles_to_structure(
                str(row[self.columns.smiles]),
                embed_3d=self.rdkit_embed,
                seed=index,
            )
        sample_id = str(row.get(self.columns.id, index)) if self.columns.id else str(index)
        return MaterialSample(
            id=sample_id,
            material_type=self.material_type,
            structure=structure,
            features=_map_columns(row, self.columns.features),
            labels=_map_columns(row, self.columns.labels),
            conditions=_map_columns(row, self.columns.conditions),
            metadata=_map_columns(row, self.columns.metadata),
            split=(
                str(row[self.columns.split])
                if self.columns.split and row.get(self.columns.split)
                else None
            ),
            provenance={"dataset": self.path.name, "row": index, "source": str(self.path)},
        )


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            yield from csv.DictReader(handle)
        return
    if suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = (
            payload
            if isinstance(payload, list)
            else payload.get("data", payload.get("records"))
        )
        if not isinstance(records, list):
            raise ValueError("JSON table must contain a list or a data/records list")
        yield from records
        return
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas and pyarrow are required for Parquet input") from exc
        yield from pd.read_parquet(path).to_dict(orient="records")
        return
    raise ValueError(f"unsupported table format: {path.suffix}")


def _map_columns(row: Mapping[str, Any], mapping: Mapping[str, str]) -> dict[str, Any]:
    return {name: _parse(row.get(column)) for name, column in mapping.items()}


def _parse(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped == "":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return float(stripped)
        except ValueError:
            return stripped
