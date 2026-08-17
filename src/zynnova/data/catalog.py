from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .api import dataset_class, list_datasets, load_builtin_plugins


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    name: str
    material_type: str
    homepage: str | None
    license: str | None
    citation: str | None
    plugin: type[Any]


def dataset_catalog() -> tuple[DatasetInfo, ...]:
    load_builtin_plugins()
    entries: list[DatasetInfo] = []
    seen: set[type[Any]] = set()
    for name in list_datasets():
        plugin = dataset_class(name)
        if plugin in seen:
            continue
        seen.add(plugin)
        entries.append(
            DatasetInfo(
                name=getattr(plugin, "name", name),
                material_type=str(getattr(plugin, "material_type", "special")),
                homepage=getattr(plugin, "homepage", None),
                license=getattr(plugin, "license", None),
                citation=getattr(plugin, "citation", None),
                plugin=plugin,
            )
        )
    return tuple(sorted(entries, key=lambda entry: (entry.material_type, entry.name)))
