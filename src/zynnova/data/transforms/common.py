from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..record import MaterialSample
from .base import SampleTransform


@dataclass(slots=True)
class Filter(SampleTransform):
    predicate: Callable[[MaterialSample], bool]

    def __call__(self, sample: MaterialSample) -> MaterialSample | None:
        return sample if self.predicate(sample) else None


@dataclass(slots=True)
class MapField(SampleTransform):
    source: str
    target: str
    function: Callable[[Any], Any]
    drop_source: bool = False

    def __call__(self, sample: MaterialSample) -> MaterialSample:
        output = sample.copy()
        value = output.require(self.source)
        output.set(self.target, self.function(value))
        if self.drop_source:
            root, key = self.source.split(".", 1)
            mapping = getattr(output, root)
            if "." not in key:
                mapping.pop(key, None)
        return output


@dataclass(slots=True)
class RenameFields(SampleTransform):
    mapping: Mapping[str, str]

    def __call__(self, sample: MaterialSample) -> MaterialSample:
        output = sample.copy()
        for source, target in self.mapping.items():
            value = output.require(source)
            output.set(target, value)
        return output


@dataclass(slots=True)
class SelectFields(SampleTransform):
    features: Sequence[str] = ()
    labels: Sequence[str] = ()
    conditions: Sequence[str] = ()
    keep_metadata: bool = True

    def __call__(self, sample: MaterialSample) -> MaterialSample:
        return sample.copy(
            features={name: sample.get(f"features.{name}") for name in self.features},
            labels={name: sample.get(f"labels.{name}") for name in self.labels},
            conditions={name: sample.get(f"conditions.{name}") for name in self.conditions},
            metadata=dict(sample.metadata) if self.keep_metadata else {},
        )


@dataclass(slots=True)
class DropMissing(SampleTransform):
    fields: Sequence[str]

    def __call__(self, sample: MaterialSample) -> MaterialSample | None:
        sentinel = object()
        for path in self.fields:
            value = sample.get(path, sentinel)
            if value is sentinel or value is None:
                return None
            if isinstance(value, float) and np.isnan(value):
                return None
        return sample


@dataclass(slots=True)
class ClipField(SampleTransform):
    path: str
    minimum: float | None = None
    maximum: float | None = None

    def __call__(self, sample: MaterialSample) -> MaterialSample:
        output = sample.copy()
        value = np.asarray(output.require(self.path))
        output.set(self.path, np.clip(value, self.minimum, self.maximum))
        return output


@dataclass(slots=True)
class StandardizeField(SampleTransform):
    path: str
    mean: float
    std: float
    epsilon: float = 1.0e-12

    def __call__(self, sample: MaterialSample) -> MaterialSample:
        output = sample.copy()
        value = np.asarray(output.require(self.path), dtype=np.float64)
        output.set(self.path, (value - self.mean) / max(self.std, self.epsilon))
        return output


@dataclass(slots=True)
class AddDerivedFields(SampleTransform):
    functions: Mapping[str, Callable[[MaterialSample], Any]] = field(default_factory=dict)

    def __call__(self, sample: MaterialSample) -> MaterialSample:
        output = sample.copy()
        for path, function in self.functions.items():
            output.set(path, function(output))
        return output
