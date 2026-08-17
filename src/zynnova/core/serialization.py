"""Stable JSON conversion helpers for manifests and backend contracts."""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convert common scientific Python values into deterministic JSON values."""

    if dataclasses.is_dataclass(value):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [to_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def dump_json(path: str | Path, value: Any) -> Path:
    """Write UTF-8 JSON atomically and return its resolved path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(to_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def load_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON document."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = ["dump_json", "load_json", "to_jsonable"]
