from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..record import MaterialSample
from .base import SampleTransform


@dataclass(slots=True)
class ConvertStructure(SampleTransform):
    """Normalize any ASE-compatible object into ZynNova ``StructureData``."""

    kind: str | None = None
    format: str | None = None
    index: int | str = -1

    def __call__(self, sample: MaterialSample) -> MaterialSample:
        if sample.structure is None:
            return sample
        if sample.material_type.value == "polymer":
            try:
                from ...structure.polymer import PolymerRecord
            except ImportError:
                PolymerRecord = ()  # type: ignore[assignment]
            if isinstance(sample.structure, PolymerRecord):
                return sample
        from ...structure.common.io import load_structure

        structure = load_structure(
            sample.structure,
            kind=self.kind or sample.material_type.value,
            format=self.format,
            index=self.index,
        )
        return sample.copy(structure=structure)


@dataclass(slots=True)
class CenterStructure(SampleTransform):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __call__(self, sample: MaterialSample) -> MaterialSample:
        if sample.structure is None or not hasattr(sample.structure, "positions"):
            return sample
        structure = sample.structure.copy()
        positions = structure.positions
        shift = positions.mean(axis=0) - self.center
        structure.positions = positions - shift
        return sample.copy(structure=structure)
