"""Physical constants used by :mod:`zynnova.zynsim`.

All continuum and electrochemical solvers use SI units internally.
"""

from __future__ import annotations

FARADAY = 96485.33212  # C mol-1
GAS_CONSTANT = 8.31446261815324  # J mol-1 K-1
VACUUM_PERMITTIVITY = 8.8541878128e-12  # F m-1
BOLTZMANN = 1.380649e-23  # J K-1
ELEMENTARY_CHARGE = 1.602176634e-19  # C

__all__ = [
    "BOLTZMANN",
    "ELEMENTARY_CHARGE",
    "FARADAY",
    "GAS_CONSTANT",
    "VACUUM_PERMITTIVITY",
]
