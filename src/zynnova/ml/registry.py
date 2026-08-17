from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class ModelEntry:
    category: str
    name: str
    factory: Callable[..., Any]
    description: str = ""


class ModelRegistry:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], ModelEntry] = {}

    def register(
        self,
        category: str,
        name: str,
        *,
        description: str = "",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        key = (category.lower(), name.lower())

        def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
            if key in self._entries:
                raise KeyError(f"model already registered: {key[0]}/{key[1]}")
            self._entries[key] = ModelEntry(key[0], key[1], factory, description)
            return factory

        return decorator

    def create(self, category: str, name: str, **kwargs: Any) -> Any:
        key = (category.lower(), name.lower())
        try:
            entry = self._entries[key]
        except KeyError as exc:
            available = self.names(category)
            raise KeyError(
                f"unknown model {category}/{name}; available={available}"
            ) from exc
        return entry.factory(**kwargs)

    def names(self, category: str | None = None) -> tuple[str, ...]:
        entries = self._entries.values()
        if category is not None:
            entries = (entry for entry in entries if entry.category == category.lower())
        return tuple(sorted(f"{entry.category}/{entry.name}" for entry in entries))

    def get(self, category: str, name: str) -> ModelEntry:
        return self._entries[(category.lower(), name.lower())]


MODELS = ModelRegistry()


__all__ = ["MODELS", "ModelEntry", "ModelRegistry"]
