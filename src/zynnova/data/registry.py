from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from .exceptions import DatasetNotFoundError

T = TypeVar("T")


class Registry:
    """Case-insensitive plugin registry with aliases."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._items: dict[str, Any] = {}

    @staticmethod
    def _key(name: str) -> str:
        return name.strip().lower().replace("-", "_")

    def register(
        self,
        name: str,
        value: T | None = None,
        *,
        aliases: tuple[str, ...] = (),
        replace: bool = False,
    ) -> T | Callable[[T], T]:
        def decorator(item: T) -> T:
            for candidate in (name, *aliases):
                key = self._key(candidate)
                if key in self._items and not replace:
                    raise KeyError(f"{self.label} {candidate!r} is already registered")
                self._items[key] = item
            return item

        return decorator(value) if value is not None else decorator

    def get(self, name: str) -> Any:
        key = self._key(name)
        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise DatasetNotFoundError(
                f"unknown {self.label} {name!r}; available: {available}"
            ) from exc

    def create(self, name: str, /, **kwargs: Any) -> Any:
        item = self.get(name)
        return item(**kwargs) if callable(item) else item

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def items(self) -> tuple[tuple[str, Any], ...]:
        return tuple(sorted(self._items.items()))


DATASETS = Registry("dataset")
TRANSFORMS = Registry("transform")
ENCODERS = Registry("encoder")
STORAGE_FORMATS = Registry("storage format")
