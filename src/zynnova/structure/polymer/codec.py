from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..common.types import StructureData
from .core import MolecularGraph, PolymerRecord, PolymerUnit
from .decoder import view2record
from .factory import ViewKind, make_view
from .record_conversion import record2stru, stru2record
from .schema import RepresentationSchema


@dataclass(slots=True)
class PolymerCodec:
    """Reusable high-throughput encoder/decoder.

    Vocabulary maps and unit templates are held once on the codec, avoiding
    repeated schema construction in Dataset/DataLoader loops.
    """

    schema: RepresentationSchema | None = None
    unit_library: Mapping[str, PolymerUnit | MolecularGraph | StructureData] = field(
        default_factory=dict
    )
    include_reconstruction: bool = False

    def encode_record(
        self,
        record: PolymerRecord,
        kind: ViewKind | str,
        **kwargs: Any,
    ) -> Any:
        if self.schema is not None:
            kwargs.setdefault("schema", self.schema)
        kwargs.setdefault("include_reconstruction", self.include_reconstruction)
        return make_view(record, kind, **kwargs)

    def encode_structure(
        self,
        structure: Any,
        kind: ViewKind | str,
        *,
        record_kwargs: dict[str, Any] | None = None,
        **view_kwargs: Any,
    ) -> Any:
        record = stru2record(structure, **dict(record_kwargs or {}))
        return self.encode_record(record, kind, **view_kwargs)

    def decode_record(self, view: Any, **kwargs: Any) -> PolymerRecord:
        kwargs.setdefault("schema", self.schema)
        kwargs.setdefault("unit_library", self.unit_library)
        return view2record(view, **kwargs)

    def decode_structure(self, view: Any, **kwargs: Any):
        record_keys = {"record_id", "prefer_payload"}
        decode_kwargs = {
            key: kwargs.pop(key) for key in list(kwargs) if key in record_keys
        }
        record = self.decode_record(view, **decode_kwargs)
        return record2stru(record, **kwargs)

    def encode_many(
        self,
        records: Iterable[PolymerRecord],
        kind: ViewKind | str,
        **kwargs: Any,
    ) -> list[Any]:
        return [self.encode_record(record, kind, **kwargs) for record in records]
