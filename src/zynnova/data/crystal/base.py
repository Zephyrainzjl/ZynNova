from __future__ import annotations

from ..record import MaterialType
from ..source import DatasetSource


class CrystalDatasetSource(DatasetSource):
    material_type = MaterialType.CRYSTAL
