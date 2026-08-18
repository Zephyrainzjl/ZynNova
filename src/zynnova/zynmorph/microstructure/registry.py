"""Thread-safe plugin registries for microstructure characterization/reconstruction."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from threading import RLock
from typing import Any


class PluginRegistry:
    """Small explicit registry with normalized names and duplicate protection."""

    def __init__(self, kind: str) -> None:
        self.kind = str(kind)
        self._items: dict[str, Any] = {}
        self._display_names: dict[str, str] = {}
        self._lock = RLock()

    @staticmethod
    def _key(name: str) -> str:
        value = "".join(ch for ch in str(name).strip().lower() if ch.isalnum())
        if not value:
            raise ValueError("plugin name cannot be empty")
        return value

    def register(
        self,
        name: str,
        value: Any | None = None,
        *,
        aliases: tuple[str, ...] = (),
        replace: bool = False,
    ) -> Any:
        def apply(plugin: Any) -> Any:
            all_names = (name, *aliases)
            with self._lock:
                for raw in all_names:
                    key = self._key(raw)
                    if not replace and key in self._items and self._items[key] is not plugin:
                        raise KeyError(f"{self.kind} plugin {raw!r} is already registered")
                for raw in all_names:
                    key = self._key(raw)
                    self._items[key] = plugin
                    self._display_names.setdefault(key, str(name))
            return plugin

        return apply(value) if value is not None else apply

    def get(self, name: str) -> Any:
        key = self._key(name)
        with self._lock:
            try:
                return self._items[key]
            except KeyError as exc:
                raise KeyError(
                    f"unknown {self.kind} plugin {name!r}; available={self.names()}"
                ) from exc

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(set(self._display_names.values())))

    def contains(self, name: str) -> bool:
        with self._lock:
            return self._key(name) in self._items

    def items(self) -> Iterator[tuple[str, Any]]:
        for name in self.names():
            yield name, self.get(name)


DESCRIPTORS = PluginRegistry("descriptor")
LOSSES = PluginRegistry("loss")
OPTIMIZERS = PluginRegistry("optimizer")


def register_descriptor(name: str, *, aliases: tuple[str, ...] = (), replace: bool = False):
    return lambda plugin: DESCRIPTORS.register(name, plugin, aliases=aliases, replace=replace)


def register_loss(name: str, *, aliases: tuple[str, ...] = (), replace: bool = False):
    return lambda plugin: LOSSES.register(name, plugin, aliases=aliases, replace=replace)


def register_optimizer(name: str, *, aliases: tuple[str, ...] = (), replace: bool = False):
    return lambda plugin: OPTIMIZERS.register(name, plugin, aliases=aliases, replace=replace)


__all__ = [
    "DESCRIPTORS",
    "LOSSES",
    "OPTIMIZERS",
    "PluginRegistry",
    "register_descriptor",
    "register_loss",
    "register_optimizer",
]
