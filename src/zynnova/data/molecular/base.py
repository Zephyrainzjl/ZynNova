from __future__ import annotations

from ..record import MaterialType
from ..source import DatasetSource


class MolecularDatasetSource(DatasetSource):
    material_type = MaterialType.MOLECULAR
