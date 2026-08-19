"""Geometric Landau--Lifshitz--Gilbert and coupled spin-lattice dynamics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._deps import require_torch
from .config import SpinConfig

torch = require_torch()

EV_PER_ANGSTROM_PER_AMU_TO_ANGSTROM_PER_FS2 = 0.009648533215665327


def _normalise_to_magnitude(vectors: Any, magnitude: Any, floor: float) -> Any:
    norm = torch.linalg.vector_norm(vectors, dim=-1).clamp_min(float(floor))
    scaled = vectors * (magnitude / norm)[:, None]
    return torch.where((magnitude > 0)[:, None], scaled, torch.zeros_like(scaled))


def llg_rhs(
    spin_vectors: Any,
    effective_field_T: Any,
    *,
    gyromagnetic_ratio: float,
    damping: float,
    floor: float = 1.0e-12,
) -> Any:
    """Gilbert-form deterministic LLG right-hand side in moment-vector form."""

    if spin_vectors.shape != effective_field_T.shape or spin_vectors.shape[-1] != 3:
        raise ValueError("spins and fields must have shape [N,3]")
    magnitude = torch.sqrt(spin_vectors.square().sum(-1) + float(floor) ** 2)
    precession = torch.cross(spin_vectors, effective_field_T, dim=-1)
    damping_term = torch.cross(spin_vectors, precession, dim=-1) / magnitude[:, None]
    prefactor = -float(gyromagnetic_ratio) / (1.0 + float(damping) ** 2)
    return prefactor * (precession + float(damping) * damping_term)


def llg_midpoint_step(
    spin_vectors: Any,
    field_function: Callable[[Any], Any],
    dt_s: float,
    config: SpinConfig,
    *,
    iterations: int = 4,
    fixed_mask: Any | None = None,
    damping: float | None = None,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-11,
) -> Any:
    """Implicit midpoint LLG step preserving each spin magnitude.

    A small fixed-point solve is used because it is time symmetric at zero
    damping and substantially more stable than explicit Euler/Heun for stiff
    exchange fields.
    """

    if dt_s <= 0 or iterations < 1 or rtol <= 0 or atol <= 0:
        raise ValueError("dt_s, midpoint iterations and tolerances must be positive")
    magnitude = torch.linalg.vector_norm(spin_vectors, dim=-1)
    candidate = spin_vectors
    converged = False
    for _ in range(iterations):
        previous = candidate
        midpoint = 0.5 * (spin_vectors + candidate)
        midpoint = _normalise_to_magnitude(
            midpoint, magnitude, config.minimum_moment
        )
        field = field_function(midpoint)
        derivative = llg_rhs(
            midpoint,
            field,
            gyromagnetic_ratio=config.llg_gyromagnetic_ratio,
            damping=config.llg_damping if damping is None else float(damping),
            floor=config.minimum_moment,
        )
        candidate = _normalise_to_magnitude(
            spin_vectors + float(dt_s) * derivative,
            magnitude,
            config.minimum_moment,
        )
        if fixed_mask is not None:
            mask = fixed_mask.to(device=spin_vectors.device, dtype=torch.bool)
            if mask.shape != (spin_vectors.shape[0],):
                raise ValueError("fixed_mask must have shape [N]")
            candidate = torch.where(mask[:, None], spin_vectors, candidate)
        if torch.allclose(candidate, previous, rtol=float(rtol), atol=float(atol)):
            converged = True
            break
    if not converged:
        absolute = torch.max(torch.abs(candidate - previous)).detach().cpu().item()
        raise RuntimeError(
            "implicit midpoint spin solve did not converge within "
            f"{iterations} iterations (max update={absolute:.3e}); reduce dt or "
            "increase midpoint_iterations"
        )
    return candidate


@dataclass(slots=True)
class SpinLatticeTrajectory:
    positions_A: np.ndarray
    velocities_A_per_fs: np.ndarray
    spin_vectors_muB: np.ndarray
    energies_eV: np.ndarray
    potential_energies_eV: np.ndarray
    kinetic_energies_eV: np.ndarray
    temperatures_K: np.ndarray
    times_fs: np.ndarray


def run_spin_lattice_dynamics(
    atoms: Any,
    model: Any,
    spin_vectors: Any,
    *,
    steps: int,
    lattice_dt_fs: float = 0.5,
    spin_substeps: int = 5,
    midpoint_iterations: int = 12,
    midpoint_rtol: float = 1.0e-8,
    midpoint_atol: float = 1.0e-11,
    fixed_spin_mask: Any | None = None,
    ensemble: str = "nve",
) -> SpinLatticeTrajectory:
    """Coupled velocity-Verlet/implicit-midpoint spin-lattice NVE dynamics.

    ``ensemble='nve'`` uses zero spin damping and records the full lattice
    kinetic plus potential energy. ``ensemble='damped'`` uses the configured
    deterministic Gilbert damping and is therefore explicitly non-NVE. A
    stochastic spin thermostat is deliberately not fabricated: lattice and
    spin baths require separately validated fluctuation-dissipation noise.
    """

    if model.config.spin.mode != "spin_lattice":
        raise ValueError("coupled dynamics requires spin_lattice mode")
    if steps < 1 or lattice_dt_fs <= 0 or spin_substeps < 1:
        raise ValueError("invalid dynamics integration settings")
    if ensemble not in {"nve", "damped"}:
        raise ValueError("ensemble must be 'nve' or 'damped'")
    from .data import atoms_to_batch

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    spin = torch.as_tensor(spin_vectors, device=device, dtype=dtype)
    if spin.shape != (len(atoms), 3):
        raise ValueError("spin_vectors must have shape [N,3]")
    masses = np.asarray(atoms.get_masses(), dtype=float)
    velocities = np.asarray(atoms.get_velocities(), dtype=float)
    if velocities.shape != (len(atoms), 3):
        velocities = np.zeros((len(atoms), 3), dtype=float)

    def evaluate(current_spin: Any) -> tuple[np.ndarray, float, Any]:
        batch, conditions = atoms_to_batch(atoms, model)
        conditions["spin_vectors"] = current_spin
        with torch.enable_grad():
            output = model.energy_forces_stress(
                batch,
                conditions=conditions,
                create_graph=False,
                compute_stress=False,
                compute_spin_fields=True,
            )
        return (
            output["forces"].detach().cpu().numpy(),
            float(output["energy"].sum().detach().cpu()),
            output["effective_field_T"].detach(),
        )

    positions_history = []
    velocities_history = []
    spins_history = []
    energies = []
    potential_energies = []
    kinetic_energies = []
    temperatures = []
    times = []
    _, energy, _ = evaluate(spin)
    acceleration_scale = EV_PER_ANGSTROM_PER_AMU_TO_ANGSTROM_PER_FS2
    spin_half_dt_s = lattice_dt_fs * 1.0e-15 / (2 * spin_substeps)
    for step in range(steps + 1):
        positions_history.append(np.asarray(atoms.positions, dtype=float).copy())
        velocities_history.append(velocities.copy())
        spins_history.append(spin.detach().cpu().numpy().copy())
        kinetic = 0.5 * np.sum(masses[:, None] * velocities**2)
        # ASE kinetic conversion: amu*(A/fs)^2 to eV.
        kinetic_eV = kinetic / EV_PER_ANGSTROM_PER_AMU_TO_ANGSTROM_PER_FS2
        potential_energies.append(energy)
        kinetic_energies.append(float(kinetic_eV))
        energies.append(energy + float(kinetic_eV))
        temperature = 2.0 * kinetic_eV / max(1, 3 * len(atoms) - 3) / 8.617333262145e-5
        temperatures.append(float(temperature))
        times.append(step * lattice_dt_fs)
        if step == steps:
            break
        # Symmetric Strang split: half spin flow, full velocity-Verlet lattice
        # flow at fixed midpoint spin, then the second half spin flow.
        for _ in range(spin_substeps):
            def field_function(candidate: Any) -> Any:
                return evaluate(candidate)[2]

            spin = llg_midpoint_step(
                spin,
                field_function,
                spin_half_dt_s,
                model.config.spin,
                iterations=midpoint_iterations,
                fixed_mask=fixed_spin_mask,
                damping=(0.0 if ensemble == "nve" else model.config.spin.llg_damping),
                rtol=midpoint_rtol,
                atol=midpoint_atol,
            )
        forces, _, _ = evaluate(spin)
        velocities += (
            0.5 * lattice_dt_fs * acceleration_scale * forces / masses[:, None]
        )
        atoms.positions[:] = np.asarray(atoms.positions) + lattice_dt_fs * velocities
        forces, _, _ = evaluate(spin)
        velocities += (
            0.5 * lattice_dt_fs * acceleration_scale * forces / masses[:, None]
        )
        for _ in range(spin_substeps):
            def field_function(candidate: Any) -> Any:
                return evaluate(candidate)[2]

            spin = llg_midpoint_step(
                spin,
                field_function,
                spin_half_dt_s,
                model.config.spin,
                iterations=midpoint_iterations,
                fixed_mask=fixed_spin_mask,
                damping=(0.0 if ensemble == "nve" else model.config.spin.llg_damping),
                rtol=midpoint_rtol,
                atol=midpoint_atol,
            )
        _, energy, _ = evaluate(spin)
    atoms.set_velocities(velocities)
    return SpinLatticeTrajectory(
        positions_A=np.asarray(positions_history),
        velocities_A_per_fs=np.asarray(velocities_history),
        spin_vectors_muB=np.asarray(spins_history),
        energies_eV=np.asarray(energies),
        potential_energies_eV=np.asarray(potential_energies),
        kinetic_energies_eV=np.asarray(kinetic_energies),
        temperatures_K=np.asarray(temperatures),
        times_fs=np.asarray(times),
    )


__all__ = [
    "SpinLatticeTrajectory", "llg_midpoint_step", "llg_rhs",
    "run_spin_lattice_dynamics",
]
