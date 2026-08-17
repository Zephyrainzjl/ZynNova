from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from ..record import MaterialSample


class SampleTransform(ABC):
    @abstractmethod
    def __call__(self, sample: MaterialSample) -> MaterialSample | None:
        """Return a transformed sample, or ``None`` to filter it out."""


class Compose(SampleTransform):
    def __init__(self, transforms: Sequence[SampleTransform | Any]) -> None:
        self.transforms = tuple(transforms)

    def __call__(self, sample: MaterialSample) -> MaterialSample | None:
        current: MaterialSample | None = sample
        for transform in self.transforms:
            if current is None:
                break
            current = transform(current)
        return current

    def apply(self, samples: Iterable[MaterialSample]) -> Iterator[MaterialSample]:
        for sample in samples:
            transformed = self(sample)
            if transformed is not None:
                yield transformed
