from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from functools import lru_cache
from typing import Any

from ..encoding import CompiledSample, TaskCompiler
from ..record import MaterialSample
from ..source import DatasetSource
from ..transforms import Compose


def _require_torch_data():
    try:
        from torch.utils.data import Dataset, IterableDataset
    except ImportError as exc:
        raise ImportError("PyTorch is required; install zynnova[data]") from exc
    return Dataset, IterableDataset


DatasetBase, IterableDatasetBase = _require_torch_data()


class MaterialDataset(DatasetBase):
    """Map-style Torch dataset for random-access material records."""

    def __init__(
        self,
        samples: Sequence[MaterialSample] | DatasetSource,
        compiler: TaskCompiler,
        *,
        transforms: Compose | None = None,
        split: str | None = None,
        cache_size: int = 0,
    ) -> None:
        if isinstance(samples, DatasetSource):
            self.samples = list(samples.iter_samples(split))
        else:
            self.samples = [sample for sample in samples if split is None or sample.split == split]
        self.compiler = compiler
        self.transforms = transforms
        self._compile_cached = (
            lru_cache(maxsize=cache_size)(self._compile_index)
            if cache_size > 0
            else self._compile_index
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _compile_index(self, index: int) -> CompiledSample:
        sample = self.samples[index]
        if self.transforms is not None:
            sample = self.transforms(sample)
            if sample is None:
                raise IndexError(f"sample at index {index} was filtered by transforms")
        compiled = self.compiler(sample)
        if compiled is None:
            raise IndexError(f"sample at index {index} was rejected by task")
        return compiled

    def __getitem__(self, index: int) -> CompiledSample:
        return self._compile_cached(index)


class StreamingMaterialDataset(IterableDatasetBase):
    """Worker-aware streaming dataset for APIs, shards and very large files."""

    def __init__(
        self,
        source: DatasetSource | Iterable[MaterialSample],
        compiler: TaskCompiler,
        *,
        transforms: Compose | None = None,
        split: str | None = None,
    ) -> None:
        self.source = source
        self.compiler = compiler
        self.transforms = transforms
        self.split = split

    def _samples(self) -> Iterator[MaterialSample]:
        if isinstance(self.source, DatasetSource):
            yield from self.source.iter_samples(self.split)
        else:
            for sample in self.source:
                if self.split is None or sample.split == self.split:
                    yield sample

    def __iter__(self) -> Iterator[CompiledSample]:
        try:
            from torch.utils.data import get_worker_info
        except ImportError:
            get_worker_info = lambda: None  # type: ignore[assignment]
        worker = get_worker_info()
        for index, sample in enumerate(self._samples()):
            if worker is not None and index % worker.num_workers != worker.id:
                continue
            if self.transforms is not None:
                sample = self.transforms(sample)
                if sample is None:
                    continue
            compiled = self.compiler(sample)
            if compiled is not None:
                yield compiled
