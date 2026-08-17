from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from ..core.polymer import PolymerRecord

RawRecord = TypeVar("RawRecord")


class DatasetAdapter(ABC, Generic[RawRecord]):
    """Plugin interface for normalizing any external dataset into PolymerRecord."""

    name: str

    @abstractmethod
    def can_handle(self, source: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def iter_records(self, source: Any) -> Iterable[PolymerRecord]:
        raise NotImplementedError


@dataclass
class FunctionalDatasetAdapter(DatasetAdapter[RawRecord]):
    name: str
    predicate: Callable[[Any], bool]
    reader: Callable[[Any], Iterable[RawRecord]]
    converter: Callable[[RawRecord], PolymerRecord]

    def can_handle(self, source: Any) -> bool:
        return self.predicate(source)

    def iter_records(self, source: Any) -> Iterable[PolymerRecord]:
        for raw_record in self.reader(source):
            record = self.converter(raw_record)
            record.validate()
            yield record


@dataclass
class AdapterRegistry:
    adapters: dict[str, DatasetAdapter[Any]] = field(default_factory=dict)

    def register(self, adapter: DatasetAdapter[Any], *, replace: bool = False) -> None:
        if adapter.name in self.adapters and not replace:
            raise KeyError(f"dataset adapter already registered: {adapter.name}")
        self.adapters[adapter.name] = adapter

    def get(self, name: str) -> DatasetAdapter[Any]:
        try:
            return self.adapters[name]
        except KeyError as exc:
            raise KeyError(f"unknown dataset adapter: {name}") from exc

    def resolve(self, source: Any) -> DatasetAdapter[Any]:
        matches = [adapter for adapter in self.adapters.values() if adapter.can_handle(source)]
        if not matches:
            raise ValueError("no registered dataset adapter can handle the source")
        if len(matches) > 1:
            names = ", ".join(adapter.name for adapter in matches)
            raise ValueError(f"multiple adapters match the source: {names}")
        return matches[0]

    def load(self, source: Any, *, adapter: str | None = None) -> list[PolymerRecord]:
        selected = self.get(adapter) if adapter else self.resolve(source)
        return list(selected.iter_records(source))


DEFAULT_ADAPTER_REGISTRY = AdapterRegistry()
