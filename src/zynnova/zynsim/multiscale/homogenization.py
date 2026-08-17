"""Analytic and finite-element homogenization operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import DirichletBC, Mesh
from ..fem import ScalarFEMProblem


@dataclass(frozen=True, slots=True)
class VoigtReussHillResult:
    voigt: np.ndarray
    reuss: np.ndarray
    hill: np.ndarray


def _fractions(values: np.ndarray, count: int) -> np.ndarray:
    fractions = np.asarray(values, dtype=np.float64)
    if fractions.shape != (count,) or np.any(fractions < 0.0):
        raise ValueError("volume fractions must be non-negative and match phase count")
    total = float(np.sum(fractions))
    if total <= 0.0:
        raise ValueError("volume fractions must have a positive sum")
    return fractions / total


def voigt_reuss_hill(
    stiffness_tensors: np.ndarray,
    volume_fractions: np.ndarray,
) -> VoigtReussHillResult:
    """Return Voigt, Reuss, and Hill averages of symmetric 6x6 stiffnesses."""

    stiffness = np.asarray(stiffness_tensors, dtype=np.float64)
    if stiffness.ndim != 3 or stiffness.shape[1:] != (6, 6):
        raise ValueError("stiffness_tensors must have shape (n_phases, 6, 6)")
    fractions = _fractions(volume_fractions, len(stiffness))
    for tensor in stiffness:
        if not np.allclose(tensor, tensor.T, rtol=1.0e-10, atol=1.0e-8):
            raise ValueError("phase stiffness must be symmetric")
        if np.linalg.eigvalsh(tensor).min() <= 0.0:
            raise ValueError("phase stiffness must be positive definite")
    voigt = np.einsum("p,pij->ij", fractions, stiffness)
    compliance = np.linalg.inv(stiffness)
    reuss = np.linalg.inv(np.einsum("p,pij->ij", fractions, compliance))
    return VoigtReussHillResult(voigt=voigt, reuss=reuss, hill=0.5 * (voigt + reuss))


def bruggeman_effective(
    bulk: float | np.ndarray,
    volume_fraction: float,
    exponent: float = 1.5,
) -> float | np.ndarray:
    if not 0.0 <= volume_fraction <= 1.0 or exponent <= 0.0:
        raise ValueError("Bruggeman volume fraction/exponent is invalid")
    result = np.asarray(bulk, dtype=np.float64) * volume_fraction**exponent
    return float(result) if result.ndim == 0 else result


def maxwell_garnett(
    matrix_conductivity: float,
    inclusion_conductivity: float,
    inclusion_fraction: float,
) -> float:
    """Isotropic spherical-inclusion Maxwell–Garnett conductivity."""

    if matrix_conductivity <= 0.0 or inclusion_conductivity < 0.0:
        raise ValueError("conductivities must be non-negative with a positive matrix")
    if not 0.0 <= inclusion_fraction < 1.0:
        raise ValueError("inclusion fraction must lie in [0, 1)")
    numerator = inclusion_conductivity + 2.0 * matrix_conductivity + 2.0 * inclusion_fraction * (
        inclusion_conductivity - matrix_conductivity
    )
    denominator = inclusion_conductivity + 2.0 * matrix_conductivity - inclusion_fraction * (
        inclusion_conductivity - matrix_conductivity
    )
    return float(matrix_conductivity * numerator / denominator)


def homogenize_conductivity_dirichlet(
    mesh: Mesh,
    cell_conductivity: np.ndarray,
    *,
    backend: str = "auto",
) -> np.ndarray:
    """Compute an apparent 3x3 tensor from three affine-boundary RVE solves.

    This is the uniform-kinematic (Dirichlet) apparent response. It is a rigorous
    upper-type boundary estimate for a finite RVE, not a periodic-boundary claim.
    """

    conductivity = np.asarray(cell_conductivity, dtype=np.float64)
    if conductivity.shape == (mesh.n_cells,):
        conductivity = conductivity[:, None, None] * np.eye(3)[None, :, :]
    if conductivity.shape != (mesh.n_cells, 3, 3):
        raise ValueError("cell_conductivity must have shape (n_cells,) or (n_cells,3,3)")
    exterior_nodes = np.unique(mesh.exterior_faces().reshape(-1))
    volumes = mesh.cell_volumes()
    effective = np.empty((3, 3), dtype=np.float64)
    for loading_axis in range(3):
        prescribed = mesh.nodes[exterior_nodes, loading_axis]
        solution = ScalarFEMProblem(
            mesh,
            conductivity,
            dirichlet=(DirichletBC.scalar(exterior_nodes, prescribed),),
            backend=backend,
        ).solve()
        average_flux = np.einsum("e,ei->i", volumes, solution.flux) / np.sum(volumes)
        effective[:, loading_axis] = -average_flux
    return 0.5 * (effective + effective.T)


__all__ = [
    "VoigtReussHillResult",
    "bruggeman_effective",
    "homogenize_conductivity_dirichlet",
    "maxwell_garnett",
    "voigt_reuss_hill",
]
