from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np

from .constants import (
    ANGSTROM_TO_BOHR,
    ATOMIC_TIME_TO_FS,
    EV_TO_HARTREE,
    HARTREE_TO_EV,
)
from .native import NativeBackend, native_module, resolve_native_backend
from .results import StationaryStates, WavefunctionTrajectory

QuantumUnits = Literal["atomic", "ev_angstrom"]
PotentialInput = Sequence[float] | np.ndarray | Callable[[np.ndarray], np.ndarray]


def _coerce_inputs(
    grid: Sequence[float] | np.ndarray,
    potential: PotentialInput,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.ascontiguousarray(grid, dtype=np.float64)
    if x.ndim != 1 or len(x) < 4:
        raise ValueError("grid must be one-dimensional with at least four points")
    values = potential(x) if callable(potential) else potential
    v = np.ascontiguousarray(values, dtype=np.float64)
    if v.shape != x.shape:
        raise ValueError("potential must return/have the same shape as grid")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(v)):
        raise ValueError("grid and potential must contain only finite values")
    spacing = np.diff(x)
    if not np.all(spacing > 0):
        raise ValueError("grid must be strictly increasing")
    if not np.allclose(spacing, spacing[0], rtol=1.0e-10, atol=1.0e-13):
        raise ValueError("grid must be uniformly spaced")
    return x, v


def _validate_units(units: str) -> QuantumUnits:
    normalized = units.strip().lower().replace("-", "_")
    aliases = {
        "atomic": "atomic",
        "au": "atomic",
        "hartree_bohr": "atomic",
        "ev_angstrom": "ev_angstrom",
        "ev_a": "ev_angstrom",
        "ev_å": "ev_angstrom",
    }
    try:
        return aliases[normalized]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError("units must be 'atomic' or 'ev_angstrom'") from exc


def _to_atomic(
    grid: np.ndarray,
    potential: np.ndarray,
    units: QuantumUnits,
) -> tuple[np.ndarray, np.ndarray]:
    if units == "atomic":
        return grid, potential
    return (
        np.ascontiguousarray(grid * ANGSTROM_TO_BOHR),
        np.ascontiguousarray(potential * EV_TO_HARTREE),
    )


def _python_stationary_atomic(
    grid: np.ndarray,
    potential: np.ndarray,
    mass: float,
    num_states: int,
    tolerance: float,
) -> dict[str, np.ndarray]:
    spacing = float(grid[1] - grid[0])
    interior = len(grid) - 2
    kinetic = 1.0 / (2.0 * mass * spacing**2)
    diagonal = potential[1:-1] + 2.0 * kinetic
    off_diagonal = np.full(interior - 1, -kinetic, dtype=np.float64)
    try:
        from scipy.linalg import eigh_tridiagonal

        energies, vectors = eigh_tridiagonal(
            diagonal,
            off_diagonal,
            select="i",
            select_range=(0, num_states - 1),
            tol=tolerance,
            check_finite=False,
        )
    except ImportError:
        hamiltonian = np.diag(diagonal)
        hamiltonian += np.diag(off_diagonal, 1)
        hamiltonian += np.diag(off_diagonal, -1)
        energies, vectors = np.linalg.eigh(hamiltonian)
        energies = energies[:num_states]
        vectors = vectors[:, :num_states]

    wavefunctions = np.zeros((num_states, len(grid)), dtype=np.float64)
    wavefunctions[:, 1:-1] = vectors.T / np.sqrt(spacing)
    residuals = np.empty(num_states, dtype=np.float64)
    for state in range(num_states):
        vector = vectors[:, state]
        applied = diagonal * vector
        applied[:-1] += off_diagonal * vector[1:]
        applied[1:] += off_diagonal * vector[:-1]
        residuals[state] = np.linalg.norm(applied - energies[state] * vector)
        largest = int(np.argmax(np.abs(wavefunctions[state])))
        if wavefunctions[state, largest] < 0:
            wavefunctions[state] *= -1
    return {
        "energies": energies,
        "wavefunctions": wavefunctions,
        "residual_norms": residuals,
    }


def solve_schrodinger_1d(
    grid: Sequence[float] | np.ndarray,
    potential: PotentialInput,
    *,
    mass: float = 1.0,
    num_states: int = 6,
    units: str = "atomic",
    backend: NativeBackend = "auto",
    tolerance: float = 1.0e-12,
    max_iterations: int = 80,
) -> StationaryStates:
    """Solve a one-dimensional, nonrelativistic bound-state problem.

    Dirichlet boundary conditions are imposed at the first and last grid point.
    ``mass`` is measured in electron masses. In ``atomic`` mode, coordinates and
    energies are Bohr and Hartree. In ``ev_angstrom`` mode they are Å and eV.
    """
    x, v = _coerce_inputs(grid, potential)
    normalized_units = _validate_units(units)
    if mass <= 0 or not np.isfinite(mass):
        raise ValueError("mass must be finite and positive")
    if int(num_states) != num_states or not 1 <= int(num_states) <= len(x) - 2:
        raise ValueError("num_states must lie in [1, len(grid) - 2]")
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if max_iterations < 8:
        raise ValueError("max_iterations must be at least 8")

    x_atomic, v_atomic = _to_atomic(x, v, normalized_units)
    tolerance_atomic = (
        float(tolerance) if normalized_units == "atomic" else float(tolerance) * EV_TO_HARTREE
    )
    selected = resolve_native_backend(backend)
    if selected == "cpp":
        native = native_module()
        assert native is not None
        raw = native.solve_schrodinger_1d(
            x_atomic,
            v_atomic,
            float(mass),
            int(num_states),
            tolerance_atomic,
            int(max_iterations),
        )
        raw = {key: np.asarray(value) for key, value in raw.items()}
    else:
        raw = _python_stationary_atomic(
            x_atomic, v_atomic, float(mass), int(num_states), tolerance_atomic
        )

    energies = np.asarray(raw["energies"], dtype=np.float64)
    wavefunctions = np.asarray(raw["wavefunctions"], dtype=np.float64)
    residuals = np.asarray(raw["residual_norms"], dtype=np.float64)
    if normalized_units == "ev_angstrom":
        energies = energies * HARTREE_TO_EV
        residuals = residuals * HARTREE_TO_EV
        wavefunctions = wavefunctions * np.sqrt(ANGSTROM_TO_BOHR)
    return StationaryStates(
        grid=x,
        potential=v,
        energies=energies,
        wavefunctions=wavefunctions,
        residual_norms=residuals,
        units=normalized_units,
        backend=selected,
        mass=float(mass),
        metadata={
            "boundary_condition": "dirichlet",
            "discretization": "second_order_centered_finite_difference",
            "tolerance": float(tolerance),
        },
    )


def solve_radial_schrodinger(
    radial_grid: Sequence[float] | np.ndarray,
    central_potential: PotentialInput,
    *,
    angular_momentum: int = 0,
    mass: float = 1.0,
    num_states: int = 6,
    units: str = "atomic",
    backend: NativeBackend = "auto",
    tolerance: float = 1.0e-12,
    max_iterations: int = 80,
) -> StationaryStates:
    """Solve the reduced radial equation for a central potential.

    Returned wavefunctions are reduced radial functions ``u(r) = r R(r)``,
    normalized as ``integral |u(r)|² dr = 1``. The reported potential is the
    effective potential including the centrifugal term.
    """
    radius, potential = _coerce_inputs(radial_grid, central_potential)
    if np.any(radius < 0):
        raise ValueError("radial_grid cannot contain negative radii")
    if mass <= 0 or not np.isfinite(mass):
        raise ValueError("mass must be finite and positive")
    if int(angular_momentum) != angular_momentum or angular_momentum < 0:
        raise ValueError("angular_momentum must be a non-negative integer")
    normalized_units = _validate_units(units)
    radial_atomic = radius if normalized_units == "atomic" else radius * ANGSTROM_TO_BOHR
    centrifugal_atomic = np.zeros_like(radius)
    positive = radial_atomic > 0
    angular_factor = float(angular_momentum * (angular_momentum + 1))
    centrifugal_atomic[positive] = angular_factor / (
        2.0 * float(mass) * radial_atomic[positive] ** 2
    )
    centrifugal = (
        centrifugal_atomic if normalized_units == "atomic" else centrifugal_atomic * HARTREE_TO_EV
    )
    result = solve_schrodinger_1d(
        radius,
        potential + centrifugal,
        mass=mass,
        num_states=num_states,
        units=normalized_units,
        backend=backend,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    result.metadata.update(
        {
            "coordinate": "radial",
            "angular_momentum": int(angular_momentum),
            "wavefunction_convention": "reduced_radial_u_equals_r_times_R",
            "potential_kind": "effective_central_plus_centrifugal",
        }
    )
    return result


def radial_wavefunction(states: StationaryStates, index: int) -> np.ndarray:
    """Convert a reduced radial state ``u(r)`` to ``R(r)``."""
    radius = states.grid
    reduced = states.wavefunctions[index]
    radial = np.zeros_like(reduced)
    positive = radius > 0
    radial[positive] = reduced[positive] / radius[positive]
    if not np.any(positive):
        raise ValueError("radial grid must contain positive radii")
    if not positive[0]:
        radial[0] = radial[np.flatnonzero(positive)[0]]
    return radial


def _solve_complex_tridiagonal(
    diagonal: np.ndarray,
    off_diagonal: np.ndarray,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    size = len(diagonal)
    upper = np.empty(size - 1, dtype=np.complex128)
    rhs = np.empty(size, dtype=np.complex128)
    pivot = diagonal[0]
    upper[0] = off_diagonal[0] / pivot
    rhs[0] = right_hand_side[0] / pivot
    for index in range(1, size):
        pivot = diagonal[index] - off_diagonal[index - 1] * upper[index - 1]
        if index + 1 < size:
            upper[index] = off_diagonal[index] / pivot
        rhs[index] = (right_hand_side[index] - off_diagonal[index - 1] * rhs[index - 1]) / pivot
    solution = np.empty(size, dtype=np.complex128)
    solution[-1] = rhs[-1]
    for index in range(size - 2, -1, -1):
        solution[index] = rhs[index] - upper[index] * solution[index + 1]
    return solution


def _python_propagate_atomic(
    grid: np.ndarray,
    potential: np.ndarray,
    initial_wavefunction: np.ndarray,
    mass: float,
    timestep: float,
    steps: int,
    save_every: int,
) -> dict[str, np.ndarray]:
    spacing = float(grid[1] - grid[0])
    interior = len(grid) - 2
    kinetic = 1.0 / (2.0 * mass * spacing**2)
    h_diagonal = potential[1:-1] + 2.0 * kinetic
    h_off = np.full(interior - 1, -kinetic, dtype=np.float64)
    wavefunction = np.asarray(initial_wavefunction[1:-1], dtype=np.complex128).copy()
    norm = spacing * float(np.vdot(wavefunction, wavefunction).real)
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("initial_wavefunction has zero or invalid norm")
    wavefunction /= np.sqrt(norm)

    left_diagonal = 1.0 + 0.5j * timestep * h_diagonal
    left_off = 0.5j * timestep * h_off
    frames: list[np.ndarray] = []
    times: list[float] = []
    norms: list[float] = []

    def save(step: int) -> None:
        frame = np.zeros(len(grid), dtype=np.complex128)
        frame[1:-1] = wavefunction
        frames.append(frame)
        times.append(step * timestep)
        norms.append(spacing * float(np.vdot(wavefunction, wavefunction).real))

    save(0)
    for step in range(1, steps + 1):
        rhs = (1.0 - 0.5j * timestep * h_diagonal) * wavefunction
        rhs[:-1] -= 0.5j * timestep * h_off * wavefunction[1:]
        rhs[1:] -= 0.5j * timestep * h_off * wavefunction[:-1]
        wavefunction = _solve_complex_tridiagonal(left_diagonal, left_off, rhs)
        if step % save_every == 0 or step == steps:
            save(step)
    return {
        "times": np.asarray(times),
        "wavefunctions": np.asarray(frames),
        "norms": np.asarray(norms),
    }


def propagate_wavefunction_1d(
    grid: Sequence[float] | np.ndarray,
    potential: PotentialInput,
    initial_wavefunction: Sequence[complex] | np.ndarray,
    *,
    mass: float = 1.0,
    timestep: float = 0.01,
    steps: int = 100,
    save_every: int = 1,
    units: str = "atomic",
    backend: NativeBackend = "auto",
) -> WavefunctionTrajectory:
    """Propagate a wavefunction with unitary Crank–Nicolson time stepping.

    ``timestep`` is in atomic time for ``units='atomic'`` and femtoseconds for
    ``units='ev_angstrom'``.
    """
    x, v = _coerce_inputs(grid, potential)
    psi = np.ascontiguousarray(initial_wavefunction, dtype=np.complex128)
    if psi.shape != x.shape:
        raise ValueError("initial_wavefunction must have the same shape as grid")
    if not np.all(np.isfinite(psi.real)) or not np.all(np.isfinite(psi.imag)):
        raise ValueError("initial_wavefunction must contain only finite values")
    if int(steps) != steps or int(save_every) != save_every:
        raise ValueError("steps and save_every must be integers")
    if mass <= 0 or timestep <= 0 or steps < 0 or save_every <= 0:
        raise ValueError(
            "mass and timestep must be positive; steps non-negative; save_every positive"
        )
    normalized_units = _validate_units(units)
    x_atomic, v_atomic = _to_atomic(x, v, normalized_units)
    timestep_atomic = (
        float(timestep) if normalized_units == "atomic" else float(timestep) / ATOMIC_TIME_TO_FS
    )
    psi_atomic = psi if normalized_units == "atomic" else psi / np.sqrt(ANGSTROM_TO_BOHR)
    selected = resolve_native_backend(backend)
    if selected == "cpp":
        native = native_module()
        assert native is not None
        raw = native.propagate_schrodinger_1d(
            x_atomic,
            v_atomic,
            np.ascontiguousarray(psi_atomic),
            float(mass),
            timestep_atomic,
            int(steps),
            int(save_every),
        )
        raw = {key: np.asarray(value) for key, value in raw.items()}
    else:
        raw = _python_propagate_atomic(
            x_atomic,
            v_atomic,
            psi_atomic,
            float(mass),
            timestep_atomic,
            int(steps),
            int(save_every),
        )
    times = np.asarray(raw["times"], dtype=np.float64)
    wavefunctions = np.asarray(raw["wavefunctions"], dtype=np.complex128)
    if normalized_units == "ev_angstrom":
        times = times * ATOMIC_TIME_TO_FS
        wavefunctions = wavefunctions * np.sqrt(ANGSTROM_TO_BOHR)
    return WavefunctionTrajectory(
        grid=x,
        potential=v,
        times=times,
        wavefunctions=wavefunctions,
        norms=np.asarray(raw["norms"], dtype=np.float64),
        units=normalized_units,
        backend=selected,
        mass=float(mass),
        metadata={
            "boundary_condition": "dirichlet",
            "integrator": "crank_nicolson",
            "timestep": float(timestep),
        },
    )


def particle_in_box_energies(
    length: float,
    num_states: int,
    *,
    mass: float = 1.0,
    units: str = "atomic",
) -> np.ndarray:
    """Return exact infinite-square-well energies for states ``n=1..N``."""
    normalized_units = _validate_units(units)
    if length <= 0 or mass <= 0 or num_states <= 0:
        raise ValueError("length, mass, and num_states must be positive")
    length_atomic = (
        float(length) if normalized_units == "atomic" else float(length) * ANGSTROM_TO_BOHR
    )
    quantum_number = np.arange(1, num_states + 1, dtype=np.float64)
    energies = np.pi**2 * quantum_number**2 / (2.0 * mass * length_atomic**2)
    return energies if normalized_units == "atomic" else energies * HARTREE_TO_EV


def harmonic_oscillator_energies(
    energy_quantum: float,
    num_states: int,
) -> np.ndarray:
    """Return ``(n + 1/2) ħω`` in the same energy unit as ``energy_quantum``."""
    if energy_quantum <= 0 or num_states <= 0:
        raise ValueError("energy_quantum and num_states must be positive")
    return (np.arange(num_states, dtype=np.float64) + 0.5) * energy_quantum


def hydrogen_energies(
    num_states: int,
    *,
    nuclear_charge: float = 1.0,
    units: str = "atomic",
) -> np.ndarray:
    """Return nonrelativistic hydrogenic energies for principal levels ``1..N``."""
    normalized_units = _validate_units(units)
    if num_states <= 0 or nuclear_charge <= 0:
        raise ValueError("num_states and nuclear_charge must be positive")
    principal = np.arange(1, num_states + 1, dtype=np.float64)
    energies = -(float(nuclear_charge) ** 2) / (2.0 * principal**2)
    return energies if normalized_units == "atomic" else energies * HARTREE_TO_EV
