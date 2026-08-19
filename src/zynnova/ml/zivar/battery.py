"""Periodic lithium transport analysis and reproducible diffusion runs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .materials import MolecularDynamicsResult, run_molecular_dynamics


@dataclass(frozen=True, slots=True)
class DiffusionResult:
    time_ps: np.ndarray
    msd_A2: np.ndarray
    diffusion_cm2_s: float
    conductivity_S_cm: float
    fit_window_ps: tuple[float, float]
    lithium_count: int
    temperature_K: float


def _unwrap_scaled(scaled: np.ndarray) -> np.ndarray:
    delta = np.diff(scaled, axis=0)
    delta -= np.round(delta)
    return np.concatenate((scaled[:1], scaled[:1] + np.cumsum(delta, axis=0)), axis=0)


def _multi_origin_msd(cartesian: np.ndarray) -> np.ndarray:
    """FFT tracer MSD averaged over every valid time origin and mobile ion."""

    if cartesian.ndim != 3 or cartesian.shape[-1] != 3:
        raise ValueError("cartesian trajectory must have shape [T,N,3]")
    steps, particles, _ = cartesian.shape
    flattened = np.asarray(cartesian, dtype=np.float64).reshape(steps, -1)
    transform = np.fft.rfft(flattened, n=2 * steps, axis=0)
    autocorrelation = np.fft.irfft(
        transform * transform.conj(), n=2 * steps, axis=0
    )[:steps].sum(axis=1)
    norm2 = np.sum(flattened * flattened, axis=1)
    prefix = np.concatenate(([0.0], np.cumsum(norm2)))
    lag = np.arange(steps)
    samples = steps - lag
    first = prefix[steps - lag]
    second = prefix[steps] - prefix[lag]
    msd = (first + second - 2.0 * autocorrelation) / (samples * particles)
    msd[0] = 0.0
    return np.maximum(msd, 0.0)


def analyze_li_diffusion(
    frames: Iterable[Any],
    *,
    timestep_fs: float,
    frame_interval: int = 1,
    temperature_K: float,
    fit_fraction: tuple[float, float] = (0.2, 0.8),
    effective_charge_e: float = 1.0,
    remove_framework_drift: bool = True,
) -> DiffusionResult:
    """Fit 3-D tracer MSD using continuously unwrapped fractional coordinates."""

    structures = list(frames)
    if len(structures) < 5 or timestep_fs <= 0 or frame_interval < 1 or temperature_K <= 0:
        raise ValueError("diffusion analysis requires >=5 frames and positive scales")
    symbols = np.asarray(structures[0].get_chemical_symbols())
    reference_symbols = structures[0].get_chemical_symbols()
    if any(frame.get_chemical_symbols() != reference_symbols for frame in structures):
        raise ValueError("all trajectory frames must preserve atom ordering and species")
    lithium = np.flatnonzero(symbols == "Li")
    if lithium.size == 0:
        raise ValueError("trajectory contains no lithium")
    scaled = np.stack([frame.get_scaled_positions(wrap=True) for frame in structures])
    unwrapped = _unwrap_scaled(scaled)
    cells = np.stack([np.asarray(frame.cell.array) for frame in structures])
    cartesian_all = np.einsum("tni,tij->tnj", unwrapped, cells)
    cartesian = cartesian_all[:, lithium]
    framework = np.flatnonzero(symbols != "Li")
    if remove_framework_drift and framework.size:
        drift = (
            cartesian_all[:, framework] - cartesian_all[:1, framework]
        ).mean(axis=1)
        cartesian = cartesian - drift[:, None, :]
    msd = _multi_origin_msd(cartesian)
    time_ps = np.arange(len(structures), dtype=float) * timestep_fs * frame_interval / 1000.0
    start = int(np.floor(fit_fraction[0] * (len(structures) - 1)))
    stop = int(np.ceil(fit_fraction[1] * (len(structures) - 1))) + 1
    if not 0 <= start < stop - 2 <= len(structures):
        raise ValueError("fit_fraction leaves too few samples")
    slope = float(np.polyfit(time_ps[start:stop], msd[start:stop], 1)[0])
    diffusion_A2_ps = max(slope, 0.0) / 6.0
    diffusion_cm2_s = diffusion_A2_ps * 1.0e-4
    volume_m3 = float(np.mean([frame.get_volume() for frame in structures])) * 1.0e-30
    number_density = lithium.size / volume_m3
    elementary_charge, boltzmann = 1.602176634e-19, 1.380649e-23
    conductivity_S_m = (
        number_density
        * (effective_charge_e * elementary_charge) ** 2
        * (diffusion_cm2_s * 1.0e-4)
        / (boltzmann * temperature_K)
    )
    return DiffusionResult(
        time_ps, msd, diffusion_cm2_s, conductivity_S_m / 100.0,
        (float(time_ps[start]), float(time_ps[stop - 1])), int(lithium.size),
        float(temperature_K)
    )


def run_li_diffusion(
    structure: Any,
    potential: Any,
    *,
    temperature_K: float = 800.0,
    timestep_fs: float = 1.0,
    steps: int = 100_000,
    trajectory_interval: int = 10,
    output_directory: str | Path = "zivar-li-diffusion",
    device: str = "cpu",
    dtype: str | None = None,
    seed: int = 42,
) -> tuple[MolecularDynamicsResult, DiffusionResult]:
    from ase.io import Trajectory

    md = run_molecular_dynamics(
        structure, potential, ensemble="nvt", temperature_K=temperature_K,
        timestep_fs=timestep_fs, steps=steps, trajectory_interval=trajectory_interval,
        output_directory=output_directory, device=device, dtype=dtype, seed=seed
    )
    trajectory = Trajectory(str(md.trajectory), "r")
    try:
        analysis = analyze_li_diffusion(
            trajectory, timestep_fs=timestep_fs, frame_interval=trajectory_interval,
            temperature_K=temperature_K
        )
    finally:
        trajectory.close()
    return md, analysis


__all__ = ["DiffusionResult", "analyze_li_diffusion", "run_li_diffusion"]
