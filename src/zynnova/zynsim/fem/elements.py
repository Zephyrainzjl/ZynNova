"""Reference implementations of first-order tetrahedral element kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class NeoHookeanElementResult:
    energy: float
    residual: np.ndarray
    tangent: np.ndarray
    jacobian: float


def tet4_geometry(coordinates: np.ndarray) -> tuple[float, np.ndarray]:
    """Return volume and physical shape-function gradients for a Tet4."""

    x = np.asarray(coordinates, dtype=np.float64)
    if x.shape != (4, 3) or not np.all(np.isfinite(x)):
        raise ValueError("Tet4 coordinates must be finite with shape (4, 3)")
    jacobian = np.column_stack((x[1] - x[0], x[2] - x[0], x[3] - x[0]))
    determinant = float(np.linalg.det(jacobian))
    scale = float(np.max(np.abs(jacobian)))
    threshold = 64.0 * np.finfo(float).eps * max(
        scale**3, np.finfo(float).tiny
    )
    if abs(determinant) <= threshold:
        raise ValueError("degenerate tetrahedral element")
    reference = np.asarray(
        ((-1.0, -1.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    gradients = reference @ np.linalg.inv(jacobian)
    return abs(determinant) / 6.0, gradients


def normalize_tensor(value: float | np.ndarray) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 0:
        tensor = np.eye(3) * float(raw)
    elif raw.shape == (3, 3):
        tensor = raw
    else:
        raise ValueError("coefficient must be a scalar or a (3, 3) tensor")
    if not np.all(np.isfinite(tensor)):
        raise ValueError("coefficient tensor contains non-finite values")
    if not np.allclose(tensor, tensor.T, rtol=1.0e-12, atol=1.0e-14):
        raise ValueError("coefficient tensor must be symmetric")
    if np.linalg.eigvalsh(tensor).min(initial=0.0) < -1.0e-14 * max(
        np.linalg.norm(tensor), 1.0
    ):
        raise ValueError("coefficient tensor must be positive semidefinite")
    return np.ascontiguousarray(tensor)


def scalar_matrices(
    coordinates: np.ndarray,
    conductivity: float | np.ndarray = 1.0,
    density: float = 1.0,
    *,
    lumped: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    volume, gradients = tet4_geometry(coordinates)
    if not np.isfinite(density) or density < 0.0:
        raise ValueError("density must be finite and non-negative")
    if lumped:
        mass = np.eye(4) * density * volume / 4.0
    else:
        mass = np.full((4, 4), density * volume / 20.0)
        np.fill_diagonal(mass, density * volume / 10.0)
    stiffness = volume * gradients @ normalize_tensor(conductivity) @ gradients.T
    return mass, stiffness


def isotropic_constitutive(young_modulus: float, poisson_ratio: float) -> np.ndarray:
    if not np.isfinite(young_modulus) or young_modulus <= 0.0:
        raise ValueError("Young's modulus must be finite and positive")
    if not np.isfinite(poisson_ratio) or not -1.0 < poisson_ratio < 0.5:
        raise ValueError("Poisson ratio must lie strictly between -1 and 0.5")
    shear = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lame = young_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )
    constitutive = np.zeros((6, 6), dtype=np.float64)
    constitutive[:3, :3] = lame
    constitutive[np.arange(3), np.arange(3)] += 2.0 * shear
    constitutive[3:, 3:] = np.eye(3) * shear
    return constitutive


def strain_displacement(gradients: np.ndarray) -> np.ndarray:
    gradients = np.asarray(gradients, dtype=np.float64)
    if gradients.shape != (4, 3):
        raise ValueError("Tet4 gradients must have shape (4, 3)")
    matrix = np.zeros((6, 12), dtype=np.float64)
    for node, (dx, dy, dz) in enumerate(gradients):
        column = 3 * node
        matrix[0, column] = dx
        matrix[1, column + 1] = dy
        matrix[2, column + 2] = dz
        matrix[3, column : column + 2] = (dy, dx)
        matrix[4, column + 1 : column + 3] = (dz, dy)
        matrix[5, (column, column + 2)] = (dz, dx)
    return matrix


def elastic_stiffness(
    coordinates: np.ndarray,
    young_modulus: float,
    poisson_ratio: float,
) -> np.ndarray:
    volume, gradients = tet4_geometry(coordinates)
    b_matrix = strain_displacement(gradients)
    return volume * b_matrix.T @ isotropic_constitutive(young_modulus, poisson_ratio) @ b_matrix


def compressible_neo_hookean(
    coordinates: np.ndarray,
    displacement: np.ndarray,
    shear_modulus: float,
    lame_lambda: float,
) -> NeoHookeanElementResult:
    """Exact residual and consistent material tangent for a total-Lagrangian Tet4."""

    if not np.isfinite(shear_modulus) or shear_modulus <= 0.0:
        raise ValueError("shear_modulus must be finite and positive")
    if not np.isfinite(lame_lambda) or lame_lambda < 0.0:
        raise ValueError("lame_lambda must be finite and non-negative")
    displacement = np.asarray(displacement, dtype=np.float64)
    if displacement.shape != (4, 3):
        raise ValueError("displacement must have shape (4, 3)")
    volume, gradients = tet4_geometry(coordinates)
    deformation = np.eye(3) + displacement.T @ gradients
    jacobian = float(np.linalg.det(deformation))
    if not np.isfinite(jacobian) or jacobian <= 0.0:
        raise ValueError("Neo-Hookean deformation Jacobian is non-positive")
    inverse_transpose = np.linalg.inv(deformation).T
    log_j = np.log(jacobian)
    first_piola = (
        shear_modulus * (deformation - inverse_transpose)
        + lame_lambda * log_j * inverse_transpose
    )
    energy_density = (
        0.5 * shear_modulus * (np.sum(deformation * deformation) - 3.0)
        - shear_modulus * log_j
        + 0.5 * lame_lambda * log_j**2
    )
    residual = volume * (first_piola @ gradients.T).T

    tangent = np.zeros((12, 12), dtype=np.float64)
    for node_a in range(4):
        for i in range(3):
            row = 3 * node_a + i
            for node_b in range(4):
                for k in range(3):
                    column = 3 * node_b + k
                    value = 0.0
                    for j in range(3):
                        for ell in range(3):
                            material = (
                                shear_modulus * float(i == k and j == ell)
                                + (shear_modulus - lame_lambda * log_j)
                                * inverse_transpose[i, ell]
                                * inverse_transpose[k, j]
                                + lame_lambda
                                * inverse_transpose[k, ell]
                                * inverse_transpose[i, j]
                            )
                            value += gradients[node_a, j] * material * gradients[node_b, ell]
                    tangent[row, column] = volume * value
    return NeoHookeanElementResult(
        energy=float(volume * energy_density),
        residual=residual,
        tangent=tangent,
        jacobian=jacobian,
    )


__all__ = [
    "NeoHookeanElementResult",
    "compressible_neo_hookean",
    "elastic_stiffness",
    "isotropic_constitutive",
    "normalize_tensor",
    "scalar_matrices",
    "strain_displacement",
    "tet4_geometry",
]
