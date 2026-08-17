from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

DipoleUnit = Literal["eA", "debye", "C_m"]

_ELEMENTARY_CHARGE_C = 1.602176634e-19
_ANGSTROM_M = 1.0e-10
_DEBYE_C_M = 3.33564e-30
_BOLTZMANN_J_K = 1.380649e-23
_VACUUM_PERMITTIVITY_F_M = 8.8541878128e-12
_EV_J = 1.602176634e-19
_A3_M3 = 1.0e-30


@dataclass(frozen=True, slots=True)
class DielectricEstimate:
    relative_permittivity: float
    dipole_variance_C2_m2: float
    temperature_K: float
    mean_volume_A3: float
    sample_count: int
    block_standard_error: float | None
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class DiffusionEstimate:
    diffusion_coefficient_A2_fs: float
    diffusion_coefficient_cm2_s: float
    slope_A2_fs: float
    fit_r2: float
    fit_start_index: int
    fit_stop_index: int


@dataclass(frozen=True, slots=True)
class BarrierStatistics:
    mean_eV: float
    standard_deviation_eV: float
    minimum_eV: float
    maximum_eV: float
    coefficient_of_variation: float
    effective_rate_s_inv: float | None = None


def recoverable_energy_density(
    electric_field_MV_m: Sequence[float],
    discharge_polarization_C_m2: Sequence[float],
) -> float:
    """Integrate the discharge branch and return ``J cm^-3``.

    ``1 MV m^-1 * 1 C m^-2`` equals ``1 J cm^-3``, so the numeric trapezoid
    integral already has the requested unit.
    """

    field = np.asarray(electric_field_MV_m, dtype=float).reshape(-1)
    polarization = np.asarray(discharge_polarization_C_m2, dtype=float).reshape(-1)
    if field.shape != polarization.shape or field.size < 2:
        raise ValueError("field and polarization must have equal length >= 2")
    if np.any(~np.isfinite(field)) or np.any(~np.isfinite(polarization)):
        raise ValueError("P-E data must be finite")
    order = np.argsort(polarization)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(abs(trapezoid(field[order], polarization[order])))


def charge_discharge_efficiency(
    stored_energy_density_J_cm3: float,
    recoverable_energy_density_J_cm3: float,
) -> float:
    stored = float(stored_energy_density_J_cm3)
    recovered = float(recoverable_energy_density_J_cm3)
    if stored <= 0 or recovered < 0:
        raise ValueError("stored energy must be positive and recovered energy non-negative")
    return recovered / stored


def linear_dielectric_energy_density(
    dielectric_constant: float,
    electric_field_MV_m: float,
) -> float:
    """Linear dielectric estimate in ``J cm^-3``."""

    epsilon_r = float(dielectric_constant)
    field_v_m = float(electric_field_MV_m) * 1.0e6
    if epsilon_r <= 0 or field_v_m < 0:
        raise ValueError("dielectric constant must be positive and field non-negative")
    joule_m3 = 0.5 * _VACUUM_PERMITTIVITY_F_M * epsilon_r * field_v_m**2
    return joule_m3 / 1.0e6


def dielectric_from_dipole_fluctuations(
    dipoles: Sequence[Sequence[float]],
    *,
    temperature_K: float,
    volume_A3: float | Sequence[float],
    unit: DipoleUnit = "eA",
    block_count: int = 5,
) -> DielectricEstimate:
    """Estimate the isotropic static dielectric constant from an equilibrium MD run."""

    vectors = np.asarray(dipoles, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or vectors.shape[0] < 2:
        raise ValueError("dipoles must have shape [frames, 3] with at least two frames")
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    volumes = np.asarray(volume_A3, dtype=float)
    mean_volume = float(np.mean(volumes))
    if not np.isfinite(mean_volume) or mean_volume <= 0:
        raise ValueError("volume_A3 must be finite and positive")
    factors = {
        "eA": _ELEMENTARY_CHARGE_C * _ANGSTROM_M,
        "debye": _DEBYE_C_M,
        "C_m": 1.0,
    }
    try:
        dipole_c_m = vectors * factors[unit]
    except KeyError as exc:
        raise ValueError(f"unsupported dipole unit: {unit}") from exc
    centered = dipole_c_m - dipole_c_m.mean(axis=0, keepdims=True)
    variance = float(np.mean(np.sum(centered * centered, axis=1)))
    denominator = (
        3.0
        * _VACUUM_PERMITTIVITY_F_M
        * _BOLTZMANN_J_K
        * float(temperature_K)
        * mean_volume
        * _A3_M3
    )
    epsilon = 1.0 + variance / denominator
    block_error = _block_dielectric_error(
        dipole_c_m,
        temperature_K=float(temperature_K),
        volume_A3=mean_volume,
        block_count=block_count,
    )
    warning = None
    if vectors.shape[0] < 100:
        warning = "Fewer than 100 dipole samples; convergence and autocorrelation need checking."
    return DielectricEstimate(
        relative_permittivity=float(epsilon),
        dipole_variance_C2_m2=variance,
        temperature_K=float(temperature_K),
        mean_volume_A3=mean_volume,
        sample_count=int(vectors.shape[0]),
        block_standard_error=block_error,
        warning=warning,
    )


def _block_dielectric_error(
    dipoles_c_m: np.ndarray,
    *,
    temperature_K: float,
    volume_A3: float,
    block_count: int,
) -> float | None:
    block_count = min(max(int(block_count), 2), dipoles_c_m.shape[0] // 2)
    if block_count < 2:
        return None
    blocks = np.array_split(dipoles_c_m, block_count)
    estimates = []
    denominator = (
        3.0
        * _VACUUM_PERMITTIVITY_F_M
        * _BOLTZMANN_J_K
        * temperature_K
        * volume_A3
        * _A3_M3
    )
    for block in blocks:
        centered = block - block.mean(axis=0, keepdims=True)
        variance = float(np.mean(np.sum(centered * centered, axis=1)))
        estimates.append(1.0 + variance / denominator)
    return float(np.std(estimates, ddof=1) / math.sqrt(len(estimates)))


def mean_squared_displacement(
    positions_A: np.ndarray,
    *,
    atom_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Return MSD for already-unwrapped positions with shape ``[frames, atoms, 3]``."""

    positions = np.asarray(positions_A, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions_A must have shape [frames, atoms, 3]")
    if atom_indices is not None:
        positions = positions[:, np.asarray(atom_indices, dtype=int)]
    displacement = positions - positions[0:1]
    return np.mean(np.sum(displacement * displacement, axis=-1), axis=1)


def diffusion_from_msd(
    time_fs: Sequence[float],
    msd_A2: Sequence[float],
    *,
    fit_fraction: tuple[float, float] = (0.5, 1.0),
    dimensions: int = 3,
) -> DiffusionEstimate:
    time = np.asarray(time_fs, dtype=float).reshape(-1)
    msd = np.asarray(msd_A2, dtype=float).reshape(-1)
    if time.shape != msd.shape or time.size < 4:
        raise ValueError("time and MSD must have equal length >= 4")
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    start = int(math.floor(fit_fraction[0] * (time.size - 1)))
    stop = int(math.ceil(fit_fraction[1] * time.size))
    start = min(max(start, 0), time.size - 2)
    stop = min(max(stop, start + 2), time.size)
    design = np.column_stack((np.ones(stop - start), time[start:stop]))
    coefficients, *_ = np.linalg.lstsq(design, msd[start:stop], rcond=None)
    fitted = design @ coefficients
    residual = msd[start:stop] - fitted
    total = msd[start:stop] - np.mean(msd[start:stop])
    r2 = 1.0 - float(residual @ residual) / max(float(total @ total), 1.0e-15)
    slope = max(float(coefficients[1]), 0.0)
    diffusion_a2_fs = slope / (2.0 * dimensions)
    # 1 A^2/fs = 0.1 cm^2/s.
    return DiffusionEstimate(
        diffusion_coefficient_A2_fs=diffusion_a2_fs,
        diffusion_coefficient_cm2_s=diffusion_a2_fs * 0.1,
        slope_A2_fs=slope,
        fit_r2=r2,
        fit_start_index=start,
        fit_stop_index=stop,
    )


def cohesive_energy_density(
    bulk_energy_eV: float,
    isolated_chain_energies_eV: Sequence[float],
    volume_A3: float,
) -> tuple[float, float]:
    """Return cohesive energy density in ``J cm^-3`` and solubility parameter."""

    if volume_A3 <= 0:
        raise ValueError("volume_A3 must be positive")
    cohesive_eV = float(sum(isolated_chain_energies_eV) - bulk_energy_eV)
    ced_eV_A3 = max(cohesive_eV / float(volume_A3), 0.0)
    ced_j_cm3 = ced_eV_A3 * _EV_J / _A3_M3 / 1.0e6
    return float(ced_j_cm3), float(math.sqrt(ced_j_cm3))


def summarize_barriers(
    barriers_eV: Sequence[float],
    *,
    temperature_K: float | None = None,
    attempt_frequency_s_inv: float = 1.0e13,
) -> BarrierStatistics:
    values = np.asarray(barriers_eV, dtype=float).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("barriers must be a non-empty finite non-negative sequence")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    rate = None
    if temperature_K is not None:
        if temperature_K <= 0 or attempt_frequency_s_inv <= 0:
            raise ValueError("temperature and attempt frequency must be positive")
        boltzmann_eV_K = _BOLTZMANN_J_K / _EV_J
        rates = attempt_frequency_s_inv * np.exp(
            -values / (boltzmann_eV_K * temperature_K)
        )
        rate = float(np.mean(rates))
    return BarrierStatistics(
        mean_eV=mean,
        standard_deviation_eV=std,
        minimum_eV=float(np.min(values)),
        maximum_eV=float(np.max(values)),
        coefficient_of_variation=std / mean if mean > 0 else 0.0,
        effective_rate_s_inv=rate,
    )


__all__ = [
    "BarrierStatistics",
    "DielectricEstimate",
    "DiffusionEstimate",
    "charge_discharge_efficiency",
    "cohesive_energy_density",
    "dielectric_from_dipole_fluctuations",
    "diffusion_from_msd",
    "linear_dielectric_energy_density",
    "mean_squared_displacement",
    "recoverable_energy_density",
    "summarize_barriers",
]
