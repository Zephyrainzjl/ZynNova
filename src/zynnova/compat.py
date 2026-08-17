"""Compatibility exports for the original structure conversion entry point.

New code should import converters from :mod:`zynnova.structure.crystal` or
:mod:`zynnova.structure.molecular`.
"""

from .structure import GraphData, StructureData

__all__ = ["GraphData", "StructureData"]
