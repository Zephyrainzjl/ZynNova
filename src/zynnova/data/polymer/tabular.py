from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ..record import MaterialSample
from ..registry import DATASETS
from .base import PolymerDatasetSource, polymer_record_from_psmiles


@DATASETS.register("polymer_table", aliases=("polymer_csv",))
class PolymerTableDatasetSource(PolymerDatasetSource):
    """Configurable PSMILES/SMILES property table.

    This source is also the recommended base class for a new public polymer
    dataset plugin: specialize the download step and provide the table path and
    source-column mapping.
    """

    name = "polymer_table"

    def __init__(
        self,
        path: str | Path,
        *,
        psmiles_column: str = "smiles",
        id_column: str | None = None,
        target_columns: Mapping[str, str] | None = None,
        feature_columns: Mapping[str, str] | None = None,
        condition_columns: Mapping[str, str] | None = None,
        metadata_columns: Mapping[str, str] | None = None,
        split_column: str | None = None,
        root: str | Path | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.psmiles_column = psmiles_column
        self.id_column = id_column
        self.target_columns = dict(target_columns or {})
        self.feature_columns = dict(feature_columns or {})
        self.condition_columns = dict(condition_columns or {})
        self.metadata_columns = dict(metadata_columns or {})
        self.split_column = split_column
        super().__init__(root or self.path.parent, download=False, prepare=False)

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        for index, row in enumerate(_rows(self.path)):
            psmiles = str(row[self.psmiles_column]).strip()
            identifier = str(row.get(self.id_column, index)) if self.id_column else str(index)
            labels = _mapping(row, self.target_columns)
            metadata = _mapping(row, self.metadata_columns)
            sample_split = (
                str(row[self.split_column])
                if self.split_column and row.get(self.split_column) not in {None, ""}
                else None
            )
            sample = MaterialSample(
                id=identifier,
                material_type=self.material_type,
                structure=polymer_record_from_psmiles(
                    psmiles,
                    record_id=identifier,
                    properties=labels,
                    metadata=metadata,
                ),
                features=_mapping(row, self.feature_columns),
                labels=labels,
                conditions=_mapping(row, self.condition_columns),
                metadata={"psmiles": psmiles, **metadata},
                provenance={"dataset": self.path.name, "source": str(self.path), "row": index},
                split=sample_split,
            )
            if split is None or sample.split == split:
                yield sample


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            yield from csv.DictReader(handle)
        return
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = (
            payload
            if isinstance(payload, list)
            else payload.get("data", payload.get("records"))
        )
        if not isinstance(records, list):
            raise ValueError("polymer JSON table must contain a record list")
        yield from records
        return
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas and pyarrow are required for Parquet") from exc
        yield from pd.read_parquet(path).to_dict(orient="records")
        return
    raise ValueError(f"unsupported polymer table format: {path.suffix}")


def _mapping(row: Mapping[str, Any], mapping: Mapping[str, str]) -> dict[str, Any]:
    return {name: _parse(row.get(column)) for name, column in mapping.items()}


def _parse(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return float(stripped)
        except ValueError:
            return stripped
