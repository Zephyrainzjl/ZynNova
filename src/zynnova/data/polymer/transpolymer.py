from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..download import DownloadSpec
from ..record import MaterialSample
from ..registry import DATASETS
from .base import PolymerDatasetSource, polymer_record_from_psmiles


@DATASETS.register("transpolymer", aliases=("trans_polymer",))
class TransPolymerDatasetSource(PolymerDatasetSource):
    """Adapter for CSV property datasets distributed with TransPolymer.

    The upstream repository contains multiple processed/original dataset files,
    so the plugin deliberately exposes ``file_pattern`` and column mapping rather
    than hard-coding one benchmark task.
    """

    name = "transpolymer"
    homepage = "https://github.com/ChangwenXu98/TransPolymer"
    license = "MIT"
    ARCHIVE_URL = "https://github.com/ChangwenXu98/TransPolymer/archive/refs/heads/main.zip"

    def __init__(
        self,
        root: str | Path,
        *,
        file_pattern: str = "**/*.csv",
        file_name_contains: str | None = None,
        psmiles_column: str | None = None,
        id_column: str | None = None,
        target_columns: Sequence[str] | Mapping[str, str] | None = None,
        condition_columns: Sequence[str] | Mapping[str, str] | None = None,
        split_column: str | None = None,
        limit: int | None = None,
        local_file: str | Path | None = None,
        local_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.file_pattern = file_pattern
        self.file_name_contains = file_name_contains
        self.psmiles_column = psmiles_column
        self.id_column = id_column
        self.target_columns = target_columns
        self.condition_columns = condition_columns
        self.split_column = split_column
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
            extract_subdir="transpolymer-local",
        )
        if local is not None:
            self._local_root = local
            return
        self.download_manager.fetch(
            DownloadSpec(
                urls=(self.ARCHIVE_URL,),
                filename="transpolymer-main.zip",
                extract=True,
            ),
            force=force,
        )

    def _tables(self) -> list[Path]:
        root = self._local_root or self.raw_dir
        if root.is_file():
            candidates = [root] if root.suffix.lower() == ".csv" else []
        else:
            candidates = sorted(root.glob(self.file_pattern))
        if self.file_name_contains:
            needle = self.file_name_contains.lower()
            candidates = [path for path in candidates if needle in path.name.lower()]
        if not candidates:
            raise FileNotFoundError(
                f"no TransPolymer CSV matched {self.file_pattern!r} in {root}"
            )
        return candidates

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        produced = 0
        for table in self._tables():
            with table.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                smiles_column = self.psmiles_column or _first_column(
                    reader.fieldnames,
                    ("psmiles", "polymer_smiles", "smiles", "SMILES", "Polymer"),
                )
                if smiles_column is None:
                    continue
                targets = _column_mapping(
                    self.target_columns,
                    reader.fieldnames,
                    excluded={smiles_column, self.id_column, self.split_column},
                )
                conditions = _column_mapping(self.condition_columns, reader.fieldnames)
                for row_index, row in enumerate(reader):
                    psmiles = (row.get(smiles_column) or "").strip()
                    if not psmiles:
                        continue
                    identifier = (
                        str(row.get(self.id_column))
                        if self.id_column and row.get(self.id_column) not in {None, ""}
                        else f"{table.stem}:{row_index}"
                    )
                    labels = {name: _parse(row.get(column)) for name, column in targets.items()}
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
                            metadata={"source_file": str(table)},
                        ),
                        labels=labels,
                        conditions={
                            name: _parse(row.get(column))
                            for name, column in conditions.items()
                        },
                        metadata={"psmiles": psmiles, "source_file": str(table)},
                        provenance={
                            "dataset": "TransPolymer",
                            "source": str(table),
                            "row": row_index,
                        },
                        split=sample_split,
                    )
                    if split is None or sample.split == split:
                        yield sample
                        produced += 1
                        if self.limit is not None and produced >= self.limit:
                            return


def _first_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _column_mapping(
    requested: Sequence[str] | Mapping[str, str] | None,
    columns: Sequence[str],
    *,
    excluded: set[str | None] | None = None,
) -> dict[str, str]:
    if isinstance(requested, Mapping):
        return dict(requested)
    if requested is not None:
        return {str(name): str(name) for name in requested}
    blocked = {value for value in (excluded or set()) if value is not None}
    return {column: column for column in columns if column not in blocked}


def _parse(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return stripped
