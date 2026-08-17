"""Element metadata used by graph feature construction.

ASE is used when available for authoritative symbols, masses, and radii.  The
small built-in fallback keeps direct ``StructureData`` conversions usable even
without optional I/O dependencies.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

_FALLBACK_SYMBOLS = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

# Pauling electronegativities. Missing/undefined values are represented as 0 and
# accompanied by a presence mask in node features.
_PAULING = {
    1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55,
    26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01,
    33: 2.18, 34: 2.55, 35: 2.96, 37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33,
    41: 1.60, 42: 2.16, 43: 1.90, 44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93,
    48: 1.69, 49: 1.78, 50: 1.96, 51: 2.05, 52: 2.10, 53: 2.66, 55: 0.79,
    56: 0.89, 57: 1.10, 58: 1.12, 59: 1.13, 60: 1.14, 62: 1.17, 63: 1.20,
    64: 1.20, 65: 1.10, 66: 1.22, 67: 1.23, 68: 1.24, 69: 1.25, 70: 1.10,
    71: 1.27, 72: 1.30, 73: 1.50, 74: 2.36, 75: 1.90, 76: 2.20, 77: 2.20,
    78: 2.28, 79: 2.54, 80: 2.00, 81: 1.62, 82: 2.33, 83: 2.02, 84: 2.00,
    85: 2.20, 87: 0.70, 88: 0.90, 89: 1.10, 90: 1.30, 91: 1.50, 92: 1.38,
    93: 1.36, 94: 1.28, 95: 1.13, 96: 1.28,
}

_PERIOD_ROWS = {
    1: {1: 1, 2: 18},
    2: {3: 1, 4: 2, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18},
    3: {11: 1, 12: 2, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18},
    4: {z: g for z, g in zip(range(19, 37), range(1, 19), strict=True)},
    5: {z: g for z, g in zip(range(37, 55), range(1, 19), strict=True)},
    6: {
        55: 1, 56: 2, **{z: 3 for z in range(57, 72)},
        **{z: g for z, g in zip(range(72, 87), range(4, 19), strict=True)},
    },
    7: {
        87: 1, 88: 2, **{z: 3 for z in range(89, 104)},
        **{z: g for z, g in zip(range(104, 119), range(4, 19), strict=True)},
    },
}

@lru_cache(maxsize=1)
def _ase_tables():
    try:
        from ase.data import atomic_masses, chemical_symbols, covalent_radii, vdw_radii
    except ImportError:
        return None
    return (
        list(chemical_symbols),
        np.asarray(atomic_masses, dtype=np.float64),
        np.asarray(covalent_radii, dtype=np.float64),
        np.asarray(vdw_radii, dtype=np.float64),
    )


def symbols_from_numbers(numbers: np.ndarray) -> list[str]:
    tables = _ase_tables()
    symbols = tables[0] if tables is not None else _FALLBACK_SYMBOLS
    return [symbols[int(z)] if 0 <= int(z) < len(symbols) else "X" for z in numbers]


def masses(numbers: np.ndarray) -> np.ndarray:
    tables = _ase_tables()
    if tables is not None:
        return np.nan_to_num(tables[1][numbers], nan=0.0)
    return numbers.astype(np.float64)


def covalent_radii(numbers: np.ndarray) -> np.ndarray:
    tables = _ase_tables()
    if tables is not None:
        values = tables[2][numbers]
        return np.where(np.isfinite(values) & (values > 0), values, 1.0)
    # Conservative fallback for graph construction, in Å.
    return np.where(numbers == 1, 0.31, np.where(numbers <= 10, 0.75, 1.1)).astype(float)


def van_der_waals_radii(numbers: np.ndarray) -> np.ndarray:
    tables = _ase_tables()
    cov = covalent_radii(numbers)
    if tables is None:
        return 1.7 * cov
    values = tables[3][numbers]
    return np.where(np.isfinite(values) & (values > 0), values, 1.7 * cov)


def periods_and_groups(numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    periods = np.zeros(len(numbers), dtype=np.float64)
    groups = np.zeros(len(numbers), dtype=np.float64)
    for i, raw_z in enumerate(numbers):
        z = int(raw_z)
        for period, row in _PERIOD_ROWS.items():
            if z in row:
                periods[i] = period
                groups[i] = row[z]
                break
    return periods, groups


def electronegativity(numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.array([_PAULING.get(int(z), 0.0) for z in numbers], dtype=np.float64)
    mask = (values > 0).astype(np.float64)
    return values, mask
