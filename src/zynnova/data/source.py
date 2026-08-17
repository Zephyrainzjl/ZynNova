from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from .download import DownloadManager
from .local_input import LocalDatasetInput
from .record import MaterialSample, MaterialType


class DatasetSource(ABC):
    """Base class for every dataset-specific plugin.

    Plugins may expose samples lazily, wrap another library's native dataset, or
    materialize a normalized cache.  ``iter_samples`` is the only method required
    by downstream pipelines.
    """

    name = "dataset"
    material_type = MaterialType.SPECIAL
    citation: str | None = None
    homepage: str | None = None
    license: str | None = None

    def __init__(
        self,
        root: str | Path,
        *,
        download: bool = True,
        prepare: bool = True,
        force: bool = False,
        local_file: str | Path | None = None,
        local_dir: str | Path | None = None,
        **options: Any,
    ) -> None:
        self.root = Path(root).expanduser().resolve() / self.name
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.local_input = LocalDatasetInput.create(
            local_file=local_file,
            local_dir=local_dir,
        )
        self.options = dict(options)
        self.download_manager = DownloadManager(self.raw_dir)
        if download:
            self.download(force=force)
        if prepare:
            self.prepare(force=force)

    def materialize_local_input(
        self,
        *,
        force: bool = False,
        extract_subdir: str = "local-input",
    ) -> Path | None:
        if self.local_input is None:
            return None
        return self.local_input.materialize(
            self.download_manager,
            force=force,
            extract_subdir=extract_subdir,
        )

    def download(self, *, force: bool = False) -> None:
        """Download raw files. API-backed or already-local plugins may override."""

    def prepare(self, *, force: bool = False) -> None:
        """Build optional normalized caches. Lazy plugins may leave this empty."""

    @abstractmethod
    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        """Yield normalized samples."""

    def __iter__(self) -> Iterator[MaterialSample]:
        return self.iter_samples()

    def take(self, count: int, *, split: str | None = None) -> list[MaterialSample]:
        if count < 0:
            raise ValueError("count must be non-negative")
        result: list[MaterialSample] = []
        for sample in self.iter_samples(split):
            result.append(sample)
            if len(result) == count:
                break
        return result

    def materialize(self, *, split: str | None = None) -> list[MaterialSample]:
        return list(self.iter_samples(split))

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "material_type": str(self.material_type),
            "root": str(self.root),
            "citation": self.citation,
            "homepage": self.homepage,
            "license": self.license,
            "options": dict(self.options),
            "local_input": (
                None if self.local_input is None else str(self.local_input.path)
            ),
        }


class SequenceSource(DatasetSource):
    """Wrap an in-memory sequence as a dataset source."""

    name = "sequence"

    def __init__(
        self,
        samples: Sequence[MaterialSample] | Iterable[MaterialSample],
        *,
        root: str | Path = ".zynnova-data",
    ) -> None:
        self._samples = samples
        super().__init__(root, download=False, prepare=False)

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        for sample in self._samples:
            if split is None or sample.split == split:
                yield sample
