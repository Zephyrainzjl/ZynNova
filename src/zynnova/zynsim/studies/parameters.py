"""Immutable-safe nested parameter access for simulation studies."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import is_dataclass, replace
from typing import Any, Mapping


def get_parameter(root: Any, path: str) -> Any:
    value = root
    for component in _components(path):
        if isinstance(value, Mapping):
            value = value[component]
        elif isinstance(value, (list, tuple)):
            value = value[int(component)]
        else:
            value = getattr(value, component)
    return value


def replace_parameter(root: Any, path: str, value: Any) -> Any:
    """Return a deep-updated root without mutating the supplied object."""

    return _replace(root, _components(path), value)


def replace_parameters(root: Any, values: Mapping[str, Any]) -> Any:
    updated = root
    for path, value in values.items():
        updated = replace_parameter(updated, path, value)
    return updated


def _replace(owner: Any, components: tuple[str, ...], value: Any) -> Any:
    if not components:
        return value
    head, *tail = components
    remaining = tuple(tail)
    if isinstance(owner, Mapping):
        updated = dict(owner)
        if head not in updated:
            raise AttributeError(f"unknown mapping parameter {head!r}")
        updated[head] = _replace(updated[head], remaining, value)
        return updated
    if isinstance(owner, list):
        updated = list(owner)
        index = int(head)
        updated[index] = _replace(updated[index], remaining, value)
        return updated
    if isinstance(owner, tuple) and not is_dataclass(owner):
        updated = list(owner)
        index = int(head)
        updated[index] = _replace(updated[index], remaining, value)
        return type(owner)(updated)
    if not hasattr(owner, head):
        raise AttributeError(f"unknown parameter path component {head!r}")
    child = _replace(getattr(owner, head), remaining, value)
    if is_dataclass(owner):
        return replace(owner, **{head: child})
    updated = deepcopy(owner)
    setattr(updated, head, child)
    return updated


def _components(path: str) -> tuple[str, ...]:
    components = tuple(path.split("."))
    if not path or any(not component for component in components):
        raise ValueError("parameter path must contain non-empty dot-separated components")
    return components


__all__ = ["get_parameter", "replace_parameter", "replace_parameters"]
