from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Iterator, Mapping


class MaterialType(StrEnum):
    CRYSTAL = "crystal"
    MOLECULAR = "molecular"
    POLYMER = "polymer"
    SPECIAL = "special"


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {} if value is None else dict(value)


@dataclass(slots=True)
class MaterialSample:
    """Unified sample exchanged between dataset plugins and model encoders.

    ``structure`` intentionally accepts any object supported by
    :mod:`zynnova.structure` (``StructureData``, ASE ``Atoms``, ``PolymerRecord``
    or a dataset-native object).  Dataset-specific names are normalized into
    ``features`` and ``labels`` while source details remain in ``metadata`` and
    ``provenance``.
    """

    id: str
    material_type: MaterialType | str
    structure: Any | None = None
    features: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    conditions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    split: str | None = None

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.material_type = MaterialType(self.material_type)
        self.features = _copy_mapping(self.features)
        self.labels = _copy_mapping(self.labels)
        self.conditions = _copy_mapping(self.conditions)
        self.metadata = _copy_mapping(self.metadata)
        self.provenance = _copy_mapping(self.provenance)
        if self.split is not None:
            self.split = str(self.split)

    def copy(self, **updates: Any) -> "MaterialSample":
        payload = {
            "features": dict(self.features),
            "labels": dict(self.labels),
            "conditions": dict(self.conditions),
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }
        payload.update(updates)
        return replace(self, **payload)

    def get(self, path: str, default: Any = None) -> Any:
        """Resolve dotted paths such as ``labels.energy`` or ``structure.cell``."""
        if not path:
            return self
        current: Any = self
        for part in path.split("."):
            if isinstance(current, Mapping):
                if part not in current:
                    return default
                current = current[part]
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return default
        return current

    def require(self, path: str) -> Any:
        sentinel = object()
        value = self.get(path, sentinel)
        if value is sentinel:
            from .exceptions import SchemaError

            raise SchemaError(f"sample {self.id!r} has no field {path!r}")
        return value

    def set(self, path: str, value: Any) -> None:
        """Set a dotted field under features, labels, conditions or metadata."""
        root, *tail = path.split(".")
        if root not in {"features", "labels", "conditions", "metadata", "provenance"}:
            raise ValueError(f"mutable field root is not supported: {root!r}")
        mapping = getattr(self, root)
        for part in tail[:-1]:
            next_mapping = mapping.setdefault(part, {})
            if not isinstance(next_mapping, dict):
                raise TypeError(f"cannot descend through non-mapping field {part!r}")
            mapping = next_mapping
        if not tail:
            raise ValueError("set() requires a nested path such as features.foo")
        mapping[tail[-1]] = value

    def iter_scalar_fields(self) -> Iterator[tuple[str, Any]]:
        for prefix in ("features", "labels", "conditions", "metadata"):
            yield from _walk_scalars(prefix, getattr(self, prefix))


def _walk_scalars(prefix: str, mapping: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
    for key, value in mapping.items():
        path = f"{prefix}.{key}"
        if isinstance(value, Mapping):
            yield from _walk_scalars(path, value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            yield path, value
