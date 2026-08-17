from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
from typing import Mapping, Sequence

import numpy as np

from .schema import DimensionlessGroup


SI_BASES = (
    "length",
    "mass",
    "time",
    "electric_current",
    "temperature",
    "amount",
    "luminous_intensity",
)


@dataclass(frozen=True, slots=True)
class PhysicalDimension:
    """Exponents of the seven SI base dimensions in a fixed order."""

    exponents: tuple[float, ...] = (0.0,) * 7

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.exponents)
        if len(values) != len(SI_BASES):
            raise ValueError("a physical dimension must contain seven SI exponents")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("dimension exponents must be finite")
        object.__setattr__(self, "exponents", values)

    @classmethod
    def dimensionless(cls) -> PhysicalDimension:
        return cls()

    @property
    def is_dimensionless(self) -> bool:
        return all(abs(value) < 1.0e-12 for value in self.exponents)

    def __mul__(self, other: PhysicalDimension) -> PhysicalDimension:
        return PhysicalDimension(
            tuple(
                left + right
                for left, right in zip(
                    self.exponents,
                    other.exponents,
                    strict=True,
                )
            )
        )

    def __truediv__(self, other: PhysicalDimension) -> PhysicalDimension:
        return PhysicalDimension(
            tuple(
                left - right
                for left, right in zip(
                    self.exponents,
                    other.exponents,
                    strict=True,
                )
            )
        )

    def __pow__(self, exponent: float) -> PhysicalDimension:
        return PhysicalDimension(
            tuple(float(exponent) * value for value in self.exponents)
        )

    def close_to(
        self,
        other: PhysicalDimension,
        *,
        tolerance: float = 1.0e-8,
    ) -> bool:
        return bool(
            np.allclose(
                self.exponents,
                other.exponents,
                atol=tolerance,
                rtol=0.0,
            )
        )

    def physo_vector(self) -> list[float]:
        return list(self.exponents)

    def pysr_unit(self) -> str:
        symbols = ("m", "kg", "s", "A", "K", "mol", "cd")
        pieces = []
        for symbol, exponent in zip(symbols, self.exponents, strict=True):
            if abs(exponent) < 1.0e-12:
                continue
            formatted = _format_exponent(exponent)
            pieces.append(symbol if formatted == "1" else f"{symbol}^{formatted}")
        return "1" if not pieces else " * ".join(pieces)

    def phye2e_unit(self) -> str | None:
        """Convert SI dimensions to PhyE2E's kg/m/s/T/V basis.

        PhyE2E does not encode amount or luminous intensity. ``None`` therefore
        means the unit hint must be omitted for that variable.
        """

        length, mass, time, current, temperature, amount, luminous = self.exponents
        if abs(amount) > 1.0e-12 or abs(luminous) > 1.0e-12:
            return None
        voltage = -current
        kg = mass - voltage
        metre = length - 2.0 * voltage
        second = time + 3.0 * voltage
        values = (kg, metre, second, temperature, voltage)
        rounded = []
        for value in values:
            nearest = round(value)
            if abs(value - nearest) > 1.0e-8:
                return None
            rounded.append(int(nearest))
        return (
            f"kg{rounded[0]}m{rounded[1]}s{rounded[2]}"
            f"T{rounded[3]}V{rounded[4]}"
        )


@dataclass(frozen=True, slots=True)
class VariableSpec:
    """Name, display unit, SI dimension, and domain constraints."""

    name: str
    unit: str = "1"
    dimension: PhysicalDimension = PhysicalDimension()
    positive: bool = False
    reference_scale: float | None = None
    dimension_known: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "unit", str(self.unit or "1"))
        object.__setattr__(
            self,
            "dimension_known",
            bool(self.dimension_known)
            and str(self.unit).strip().lower() not in {"?", "unknown"},
        )
        if self.reference_scale is not None:
            scale = float(self.reference_scale)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("reference_scale must be finite and positive")
            object.__setattr__(self, "reference_scale", scale)


DIMENSIONLESS = PhysicalDimension()
LENGTH = PhysicalDimension((1, 0, 0, 0, 0, 0, 0))
MASS = PhysicalDimension((0, 1, 0, 0, 0, 0, 0))
TIME = PhysicalDimension((0, 0, 1, 0, 0, 0, 0))
CURRENT = PhysicalDimension((0, 0, 0, 1, 0, 0, 0))
TEMPERATURE = PhysicalDimension((0, 0, 0, 0, 1, 0, 0))
ENERGY = PhysicalDimension((2, 1, -2, 0, 0, 0, 0))
PRESSURE = PhysicalDimension((-1, 1, -2, 0, 0, 0, 0))
ELECTRIC_FIELD = PhysicalDimension((1, 1, -3, -1, 0, 0, 0))
POLARIZATION = PhysicalDimension((-2, 0, 1, 1, 0, 0, 0))
DIPOLE = PhysicalDimension((1, 0, 1, 1, 0, 0, 0))
DIFFUSIVITY = PhysicalDimension((2, 0, -1, 0, 0, 0, 0))
FREQUENCY = PhysicalDimension((0, 0, -1, 0, 0, 0, 0))
CONDUCTIVITY = PhysicalDimension((-3, -1, 3, 2, 0, 0, 0))
PIEZOELECTRIC_D = PhysicalDimension((-1, -1, 3, 1, 0, 0, 0))


_EXACT_SPECS: dict[str, tuple[str, PhysicalDimension, bool]] = {
    "temperature_K": ("K", TEMPERATURE, True),
    "electric_field_MV_m": ("MV/m", ELECTRIC_FIELD, False),
    "breakdown_strength_MV_m": ("MV/m", ELECTRIC_FIELD, True),
    "maximum_polarization_C_m2": ("C/m^2", POLARIZATION, False),
    "remanent_polarization_C_m2": ("C/m^2", POLARIZATION, False),
    "recoverable_energy_density_J_cm3": ("J/cm^3", PRESSURE, True),
    "linear_energy_density_J_cm3": ("J/cm^3", PRESSURE, True),
    "cohesive_energy_density_J_cm3": ("J/cm^3", PRESSURE, True),
    "barrier_standard_deviation_eV": ("eV", ENERGY, True),
    "phase_energy_gap_eV": ("eV", ENERGY, False),
    "diffusion_coefficient_A2_fs": ("A^2/fs", DIFFUSIVITY, True),
    "piezoelectric_d33_pC_N": ("pC/N", PIEZOELECTRIC_D, False),
}


_DIMENSIONLESS_TOKENS = (
    "fraction",
    "ratio",
    "entropy",
    "dielectric_constant",
    "relative_permittivity",
    "efficiency",
    "crystallinity",
    "dispersity",
    "component_count",
    "coordination_number",
    "order_parameter",
    "novelty",
)


def infer_variable_spec(name: str) -> VariableSpec:
    """Infer common polymer-discovery units without guessing unknown fields."""

    name = str(name)
    if name in _EXACT_SPECS:
        unit, dimension, positive = _EXACT_SPECS[name]
        return VariableSpec(name, unit, dimension, positive)
    lowered = name.lower()
    if any(token in lowered for token in _DIMENSIONLESS_TOKENS):
        return VariableSpec(name, "1", DIMENSIONLESS)
    if lowered.endswith("_ev") or any(
        token in lowered
        for token in ("barrier_ev", "bandgap", "band_gap", "homo_ev", "lumo_ev")
    ):
        return VariableSpec(name, "eV", ENERGY)
    if lowered.endswith(("_j_cm3", "_mj_m3", "_pa", "_mpa", "_gpa")):
        return VariableSpec(name, "J/m^3", PRESSURE)
    if lowered.endswith(("_mv_m", "_v_m")):
        return VariableSpec(name, "V/m", ELECTRIC_FIELD)
    if lowered.endswith(("_c_m2", "_uc_cm2")):
        return VariableSpec(name, "C/m^2", POLARIZATION)
    if lowered.endswith(("_a", "_angstrom", "_nm", "_um", "_mm", "_m")):
        return VariableSpec(name, "m", LENGTH)
    if lowered.endswith(("_fs", "_ps", "_ns", "_us", "_ms", "_s")):
        return VariableSpec(name, "s", TIME)
    if lowered.endswith(("_hz", "_khz", "_mhz", "_ghz")):
        return VariableSpec(name, "Hz", FREQUENCY)
    if "diffusion" in lowered or lowered.endswith(("_m2_s", "_cm2_s")):
        return VariableSpec(name, "m^2/s", DIFFUSIVITY)
    if "conductivity" in lowered:
        return VariableSpec(name, "S/m", CONDUCTIVITY)
    if "dipole" in lowered and lowered.endswith(("_ea", "_debye")):
        return VariableSpec(name, "C*m", DIPOLE)
    if lowered.endswith("_k"):
        return VariableSpec(name, "K", TEMPERATURE)
    return VariableSpec(name, "unknown", dimension_known=False)


def resolve_variable_specs(
    names: Sequence[str],
    overrides: Mapping[str, VariableSpec] | None = None,
) -> tuple[VariableSpec, ...]:
    overrides = {} if overrides is None else dict(overrides)
    result = []
    for name in names:
        candidate = overrides.get(str(name))
        if candidate is None:
            candidate = infer_variable_spec(str(name))
        elif candidate.name != str(name):
            candidate = VariableSpec(
                name=str(name),
                unit=candidate.unit,
                dimension=candidate.dimension,
                positive=candidate.positive,
                reference_scale=candidate.reference_scale,
                dimension_known=candidate.dimension_known,
            )
        result.append(candidate)
    return tuple(result)


def buckingham_pi_groups(
    variables: Sequence[VariableSpec],
    *,
    max_denominator: int = 8,
    tolerance: float = 1.0e-8,
) -> tuple[DimensionlessGroup, ...]:
    """Compute an auditable integer basis for the dimensional null space."""

    variables = tuple(variables)
    if len(variables) < 2:
        return ()
    matrix = np.asarray(
        [variable.dimension.exponents for variable in variables],
        dtype=float,
    ).T
    groups = []
    seen: set[tuple[int, ...]] = set()
    for vector in _fraction_null_space(
        matrix,
        max_denominator=max_denominator,
    ):
        exponents = _integerize_fraction_vector(
            vector,
            max_denominator=max_denominator,
        )
        if not any(exponents):
            continue
        residual = float(np.linalg.norm(matrix @ np.asarray(exponents, dtype=float)))
        if residual > max(tolerance, 1.0e-6):
            continue
        canonical = _canonical_sign(exponents)
        if canonical in seen:
            continue
        seen.add(canonical)
        expression = _group_expression(variables, canonical)
        groups.append(
            DimensionlessGroup(
                expression=expression,
                variables=tuple(
                    variable.name
                    for variable, exponent in zip(
                        variables,
                        canonical,
                        strict=True,
                    )
                    if exponent
                ),
                exponents=canonical,
                residual=residual,
            )
        )
    return tuple(groups)


def _fraction_null_space(
    matrix: np.ndarray,
    *,
    max_denominator: int,
) -> tuple[tuple[Fraction, ...], ...]:
    """Return a deterministic rational null-space basis by exact RREF."""

    rows = [
        [
            Fraction(float(value)).limit_denominator(max_denominator)
            for value in row
        ]
        for row in np.asarray(matrix, dtype=float)
    ]
    if not rows:
        return ()
    row_count = len(rows)
    column_count = len(rows[0])
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(column_count):
        selected = next(
            (
                row
                for row in range(pivot_row, row_count)
                if rows[row][column] != 0
            ),
            None,
        )
        if selected is None:
            continue
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        pivot = rows[pivot_row][column]
        rows[pivot_row] = [value / pivot for value in rows[pivot_row]]
        for row in range(row_count):
            if row == pivot_row:
                continue
            factor = rows[row][column]
            if factor == 0:
                continue
            rows[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    rows[row],
                    rows[pivot_row],
                    strict=True,
                )
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == row_count:
            break
    free_columns = [
        column
        for column in range(column_count)
        if column not in pivot_columns
    ]
    basis = []
    for free in free_columns:
        vector = [Fraction(0, 1) for _ in range(column_count)]
        vector[free] = Fraction(1, 1)
        for row, pivot in enumerate(pivot_columns):
            vector[pivot] = -rows[row][free]
        basis.append(tuple(vector))
    return tuple(basis)


def _integerize_fraction_vector(
    vector: Sequence[Fraction],
    *,
    max_denominator: int,
) -> tuple[int, ...]:
    fractions = [
        Fraction(value).limit_denominator(max_denominator)
        for value in vector
    ]
    denominator = _lcm_many([fraction.denominator for fraction in fractions])
    integers = [int(fraction * denominator) for fraction in fractions]
    divisor = reduce(gcd, (abs(value) for value in integers if value), 0) or 1
    return tuple(value // divisor for value in integers)


def _canonical_sign(exponents: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in exponents)
    first = next((value for value in values if value), 1)
    return values if first > 0 else tuple(-value for value in values)


def _group_expression(
    variables: Sequence[VariableSpec],
    exponents: Sequence[int],
) -> str:
    numerator = []
    denominator = []
    for variable, exponent in zip(variables, exponents, strict=True):
        if exponent == 0:
            continue
        token = variable.name if abs(exponent) == 1 else f"{variable.name}^{abs(exponent)}"
        (numerator if exponent > 0 else denominator).append(token)
    top = "*".join(numerator) or "1"
    bottom = "*".join(denominator)
    return top if not bottom else f"({top})/({bottom})"


def _lcm_many(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result = abs(result * value) // gcd(result, value)
    return result


def _format_exponent(value: float) -> str:
    nearest = round(value)
    if abs(value - nearest) < 1.0e-10:
        return str(int(nearest))
    return re.sub(r"\.?0+$", "", f"{value:.8f}")


__all__ = [
    "CONDUCTIVITY",
    "CURRENT",
    "DIFFUSIVITY",
    "DIMENSIONLESS",
    "DIPOLE",
    "ELECTRIC_FIELD",
    "ENERGY",
    "FREQUENCY",
    "LENGTH",
    "MASS",
    "PIEZOELECTRIC_D",
    "POLARIZATION",
    "PRESSURE",
    "PhysicalDimension",
    "SI_BASES",
    "TEMPERATURE",
    "TIME",
    "VariableSpec",
    "buckingham_pi_groups",
    "infer_variable_spec",
    "resolve_variable_specs",
]
