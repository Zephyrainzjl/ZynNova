from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..adapters import as_structure_data
from ..local_input import iter_records
from ..record import MaterialSample
from ..registry import DATASETS
from .base import CrystalDatasetSource


@DATASETS.register("matbench")
class MatbenchDatasetSource(CrystalDatasetSource):
    """One Matbench task exposed as normalized crystal/composition samples."""

    name = "matbench"
    homepage = "https://matbench.materialsproject.org/"
    license = "MIT"

    def __init__(
        self,
        root: str | Path,
        *,
        task: str,
        structure_column: str | None = None,
        target_columns: tuple[str, ...] | None = None,
        local_file: str | Path | None = None,
        local_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.structure_column = structure_column
        self.target_columns = target_columns
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
            extract_subdir="matbench-local",
        )
        if local is not None:
            self._local_root = local

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        del split
        if self._local_root is not None:
            yield from self._iter_local()
            return
        try:
            from matminer.datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "matminer is required for automatic Matbench download; "
                "install zynnova[data-crystal] or pass local_file/local_dir"
            ) from exc
        frame = load_dataset(self.task)
        structure_column = self.structure_column or next(
            (name for name in ("structure", "initial_structure") if name in frame.columns),
            None,
        )
        targets = self.target_columns or tuple(
            name
            for name in frame.columns
            if name != structure_column and name not in {"composition", "formula"}
        )
        for index, row in frame.iterrows():
            yield self._sample_from_mapping(
                dict(row),
                index=index,
                structure_column=structure_column,
                targets=targets,
            )

    def _iter_local(self) -> Iterator[MaterialSample]:
        for index, row in enumerate(iter_records(self._local_root)):
            structure_column = self.structure_column or next(
                (
                    name
                    for name in ("structure", "initial_structure")
                    if name in row
                ),
                None,
            )
            targets = self.target_columns or tuple(
                name
                for name in row
                if name != structure_column
                and name not in {"composition", "formula", "id", "index", "split"}
            )
            yield self._sample_from_mapping(
                row,
                index=row.get("index", row.get("id", index)),
                structure_column=structure_column,
                targets=targets,
            )

    def _sample_from_mapping(
        self,
        row: dict[str, Any],
        *,
        index: Any,
        structure_column: str | None,
        targets: tuple[str, ...],
    ) -> MaterialSample:
        structure_value = row.get(structure_column) if structure_column else None
        structure = (
            _decode_structure(structure_value)
            if structure_value is not None and structure_value != ""
            else None
        )
        labels = {name: _clean(row.get(name)) for name in targets}
        metadata = {
            name: str(row[name])
            for name in ("composition", "formula")
            if name in row and row[name] is not None
        }
        return MaterialSample(
            id=f"{self.task}:{index}",
            material_type=self.material_type,
            structure=structure,
            labels=labels,
            metadata=metadata,
            provenance={"dataset": "Matbench", "task": self.task, "row": str(index)},
            split=(str(row["split"]) if row.get("split") not in {None, ""} else None),
        )


def _decode_structure(value: Any):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            value = json.loads(stripped)
    if isinstance(value, dict):
        try:
            return as_structure_data(value)
        except (TypeError, ValueError, FileNotFoundError):
            try:
                from pymatgen.core import Structure
            except ImportError as exc:
                raise ImportError(
                    "pymatgen is required to parse a Matbench structure dictionary"
                ) from exc
            return as_structure_data(Structure.from_dict(value))
    return as_structure_data(value)


def _clean(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, str):
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
    return value
