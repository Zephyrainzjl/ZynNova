from __future__ import annotations

from typing import Any

from .io.json_codec import record_to_dict
from .record_conversion import stru2record


def stru2simple(structure: Any, *, as_dict: bool = False, **kwargs: Any):
    record = stru2record(structure, **kwargs)
    return record_to_dict(record) if as_dict else record

from .simple2stru import simple2stru  # noqa: E402

__all__ = ["stru2simple", "simple2stru"]
