from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from ..record import MaterialSample, MaterialType
from ..source import DatasetSource
from ..registry import DATASETS


@DATASETS.register("generic")
class GenericDatasetSource(DatasetSource):
    """Adapter for data that does not fit crystal/molecule/polymer conventions."""

    name = "generic"
    material_type = MaterialType.SPECIAL

    def __init__(
        self,
        records: Iterable[Any] | Callable[[], Iterable[Any]],
        converter: Callable[[Any, int], MaterialSample],
        *,
        root: str | Path = ".zynnova-data",
    ) -> None:
        self.records = records
        self.converter = converter
        super().__init__(root, download=False, prepare=False)

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        records = self.records() if callable(self.records) else self.records
        for index, record in enumerate(records):
            sample = self.converter(record, index)
            if split is None or sample.split == split:
                yield sample
