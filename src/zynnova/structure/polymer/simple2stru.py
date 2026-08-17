from __future__ import annotations

from typing import Any

from .core import PolymerRecord
from .io.json_codec import record_from_dict
from .record_conversion import record2stru


def simple2stru(simple: PolymerRecord | dict[str, Any], **kwargs: Any):
    record = simple if isinstance(simple, PolymerRecord) else record_from_dict(simple)
    return record2stru(record, **kwargs)
