"""Cartesian symmetric-traceless multipoles through angular rank four.

ZIVAR stores irreducible coefficients with exactly ``2*l+1`` components per
rank.  A deterministic orthonormal Cartesian STF basis converts them to
physical tensors and provides rotation-covariant Fourier form factors without
introducing a second learned equivariant network.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ._deps import require_torch

torch = require_torch()


def multipole_slice(ell: int) -> slice:
    if ell < 0:
        raise ValueError("ell must be nonnegative")
    return slice(ell * ell, (ell + 1) * (ell + 1))


def multipole_dim(lmax: int) -> int:
    if not 0 <= lmax <= 4:
        raise ValueError("lmax must lie in [0, 4]")
    return (lmax + 1) ** 2


@lru_cache(maxsize=9)
def symmetric_multiindices(rank: int) -> tuple[tuple[int, int, int], ...]:
    """Return the unique Cartesian component counts for a symmetric rank."""

    if not 0 <= rank <= 8:
        raise ValueError("symmetric rank must lie in [0, 8]")
    return tuple(
        (nx, ny, rank - nx - ny)
        for nx in range(rank + 1)
        for ny in range(rank - nx + 1)
    )


@lru_cache(maxsize=9)
def symmetric_multiplicities(rank: int) -> tuple[int, ...]:
    """Number of full Cartesian entries represented by each unique component."""

    return tuple(
        math.factorial(rank)
        // (math.factorial(nx) * math.factorial(ny) * math.factorial(nz))
        for nx, ny, nz in symmetric_multiindices(rank)
    )


def _symmetric_basis(rank: int) -> np.ndarray:
    if rank == 0:
        return np.ones((1, 1), dtype=np.float64)
    rows: list[np.ndarray] = []
    for nx in range(rank + 1):
        for ny in range(rank - nx + 1):
            nz = rank - nx - ny
            axes = (0,) * nx + (1,) * ny + (2,) * nz
            permutations = sorted(set(itertools.permutations(axes)))
            row = np.zeros(3**rank, dtype=np.float64)
            scale = 1.0 / math.sqrt(len(permutations))
            for item in permutations:
                row[np.ravel_multi_index(item, (3,) * rank)] = scale
            rows.append(row)
    return np.stack(rows)


def _trace_matrix(rank: int) -> np.ndarray:
    if rank < 2:
        return np.zeros((0, 3**rank), dtype=np.float64)
    output_rank = rank - 2
    matrix = np.zeros((3**output_rank, 3**rank), dtype=np.float64)
    output_indices = [()] if output_rank == 0 else itertools.product(range(3), repeat=output_rank)
    for output_flat, tail in enumerate(output_indices):
        for axis in range(3):
            index = (axis, axis, *tail)
            matrix[output_flat, np.ravel_multi_index(index, (3,) * rank)] = 1.0
    return matrix


@lru_cache(maxsize=5)
def stf_basis_numpy(rank: int) -> np.ndarray:
    """Return an orthonormal ``[2*l+1, 3**l]`` STF basis."""

    if not 0 <= rank <= 4:
        raise ValueError("STF rank must lie in [0, 4]")
    symmetric = _symmetric_basis(rank)
    if rank < 2:
        result = symmetric
    else:
        restricted_trace = _trace_matrix(rank) @ symmetric.T
        _, singular_values, right = np.linalg.svd(restricted_trace, full_matrices=True)
        tolerance = max(restricted_trace.shape) * np.finfo(float).eps * (
            singular_values[0] if singular_values.size else 1.0
        )
        matrix_rank = int(np.sum(singular_values > tolerance))
        result = right[matrix_rank:] @ symmetric
    expected = 2 * rank + 1
    if result.shape != (expected, 3**rank):
        raise RuntimeError(f"invalid STF basis shape for l={rank}: {result.shape}")
    # Stabilise each row's otherwise arbitrary SVD sign.
    for row in result:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    result[np.abs(result) < 1.0e-14] = 0.0
    result.setflags(write=False)
    return result


def stf_bases(lmax: int, *, device: Any, dtype: Any) -> tuple[Any, ...]:
    return tuple(
        _stf_basis_torch(ell, str(torch.device(device)), dtype)
        for ell in range(lmax + 1)
    )


@lru_cache(maxsize=40)
def _stf_basis_torch(rank: int, device: str, dtype: Any) -> Any:
    """Cache the five tiny immutable STF transforms on their execution device."""

    return torch.as_tensor(
        stf_basis_numpy(rank).copy(), device=torch.device(device), dtype=dtype
    )


def coefficients_to_cartesian(coefficients: Any, ell: int) -> Any:
    """Convert ``[..., 2*l+1]`` coefficients to ``[..., 3, ..., 3]``."""

    if coefficients.shape[-1] != 2 * ell + 1:
        raise ValueError("coefficient width does not match ell")
    basis = _stf_basis_torch(ell, str(coefficients.device), coefficients.dtype)
    flat = coefficients @ basis
    return flat.reshape((*coefficients.shape[:-1], *((3,) * ell)))


@lru_cache(maxsize=40)
def _stf_to_symmetric_components(rank: int, device: str, dtype: Any) -> Any:
    basis = _stf_basis_torch(rank, device, dtype)
    if rank == 0:
        indices = (0,)
    else:
        indices = tuple(
            int(np.ravel_multi_index((0,) * nx + (1,) * ny + (2,) * nz, (3,) * rank))
            for nx, ny, nz in symmetric_multiindices(rank)
        )
    return basis[:, torch.as_tensor(indices, device=torch.device(device))]


def coefficients_to_symmetric_components(coefficients: Any, ell: int) -> Any:
    """Convert irreducible coefficients to unique raw Cartesian components."""

    if coefficients.shape[-1] != 2 * ell + 1:
        raise ValueError("coefficient width does not match ell")
    transform = _stf_to_symmetric_components(
        ell, str(coefficients.device), coefficients.dtype
    )
    return coefficients @ transform


def cartesian_to_coefficients(tensor: Any, ell: int) -> Any:
    if ell and tuple(tensor.shape[-ell:]) != (3,) * ell:
        raise ValueError("Cartesian tensor shape does not match ell")
    flat = tensor.reshape((*tensor.shape[:-ell], 3**ell)) if ell else tensor.unsqueeze(-1)
    basis = _stf_basis_torch(ell, str(tensor.device), tensor.dtype)
    return flat @ basis.T


def rotate_coefficients(coefficients: Any, rotation: Any, ell: int) -> Any:
    """Actively rotate irreducible coefficients by a proper or improper O(3) map."""

    tensor = coefficients_to_cartesian(coefficients, ell)
    if ell == 0:
        return coefficients
    letters = "abcdefghijklmnop"
    old = letters[:ell]
    new = letters[ell : 2 * ell]
    equation = ",".join(f"{new[i]}{old[i]}" for i in range(ell))
    equation += ",..." + "".join(old) + "->..." + "".join(new)
    rotated = torch.einsum(equation, *((rotation,) * ell), tensor)
    return cartesian_to_coefficients(rotated, ell)


def solid_harmonic_features(k_vectors: Any, lmax: int) -> tuple[Any, ...]:
    """Evaluate the STF contractions ``B_l : k^l`` for every reciprocal vector."""

    if k_vectors.ndim != 2 or k_vectors.shape[1] != 3:
        raise ValueError("k_vectors must have shape [K, 3]")
    features: list[Any] = []
    power = k_vectors.new_ones((k_vectors.shape[0], 1))
    for ell, basis in enumerate(
        stf_bases(lmax, device=k_vectors.device, dtype=k_vectors.dtype)
    ):
        if ell:
            power = (power.unsqueeze(-1) * k_vectors.unsqueeze(1)).reshape(
                k_vectors.shape[0], -1
            )
        features.append(power @ basis.T)
    return tuple(features)


def gaussian_multipole_form_factor(coefficients: Any, k_vectors: Any, width_A: float) -> Any:
    """Fourier amplitude of an atom-centred Gaussian STF multipole density."""

    lmax = int(round(math.sqrt(coefficients.shape[-1]) - 1))
    if multipole_dim(lmax) != coefficients.shape[-1]:
        raise ValueError("multipole coefficient width is not a complete l family")
    values = coefficients.new_zeros(
        (coefficients.shape[0], k_vectors.shape[0]), dtype=torch.complex64
        if coefficients.dtype in {torch.float16, torch.bfloat16, torch.float32}
        else torch.complex128
    )
    features = solid_harmonic_features(k_vectors, lmax)
    for ell, feature in enumerate(features):
        block = coefficients[..., multipole_slice(ell)] @ feature.T
        values = values + (1j**ell / math.factorial(ell)) * block
    damping = torch.exp(-0.5 * float(width_A) ** 2 * k_vectors.square().sum(-1))
    return values * damping


@dataclass(slots=True)
class SpinMultipoleState:
    """Charge density and non-collinear magnetisation-density multipoles.

    ``charge`` has shape ``[N, (lmax+1)^2]``. ``spin`` has shape
    ``[N, 3, (lmax+1)^2]``; its Cartesian axis is axial and time odd, while the
    last axis carries the spatial STF multipole representation.
    """

    charge: Any
    spin: Any

    def __post_init__(self) -> None:
        if self.charge.ndim != 2 or self.spin.ndim != 3:
            raise ValueError("invalid spin-multipole tensor ranks")
        if self.spin.shape[:2] != (self.charge.shape[0], 3):
            raise ValueError("spin must have shape [N, 3, D]")
        if self.spin.shape[-1] != self.charge.shape[-1]:
            raise ValueError("charge and spin multipole widths differ")
        lmax = int(round(math.sqrt(self.charge.shape[-1]) - 1))
        if multipole_dim(lmax) != self.charge.shape[-1]:
            raise ValueError("state does not contain a complete multipole family")

    @property
    def lmax(self) -> int:
        return int(round(math.sqrt(self.charge.shape[-1]) - 1))

    @property
    def charges(self) -> Any:
        return self.charge[:, 0]

    @property
    def magnetic_moments(self) -> Any:
        return self.spin[:, :, 0]

    @property
    def dipoles(self) -> Any:
        if self.lmax < 1:
            return self.charge.new_zeros((self.charge.shape[0], 3))
        return coefficients_to_cartesian(self.charge[:, multipole_slice(1)], 1)

    @property
    def quadrupoles(self) -> Any:
        if self.lmax < 2:
            return self.charge.new_zeros((self.charge.shape[0], 3, 3))
        return coefficients_to_cartesian(self.charge[:, multipole_slice(2)], 2)

    def detached(self, *, requires_grad: bool = False) -> SpinMultipoleState:
        return SpinMultipoleState(
            self.charge.detach().requires_grad_(requires_grad),
            self.spin.detach().requires_grad_(requires_grad),
        )

    def time_reversed(self) -> SpinMultipoleState:
        return SpinMultipoleState(self.charge, -self.spin)

    def spin_density_matrix(self) -> Any:
        """Return Hermitian Pauli coefficients of the signed charge response.

        The trace is the signed atomic charge-response multipole, not a positive
        electron-population density; positive semidefiniteness is therefore not
        implied.
        """

        complex_dtype = (
            torch.complex64
            if self.charge.dtype in {torch.float16, torch.bfloat16, torch.float32}
            else torch.complex128
        )
        charge = self.charge.to(complex_dtype)
        sx, sy, sz = (self.spin[:, axis].to(complex_dtype) for axis in range(3))
        matrix = torch.zeros(
            (*charge.shape, 2, 2), device=charge.device, dtype=complex_dtype
        )
        matrix[..., 0, 0] = 0.5 * (charge + sz)
        matrix[..., 1, 1] = 0.5 * (charge - sz)
        matrix[..., 0, 1] = 0.5 * (sx - 1j * sy)
        matrix[..., 1, 0] = 0.5 * (sx + 1j * sy)
        return matrix


__all__ = [
    "SpinMultipoleState",
    "cartesian_to_coefficients",
    "coefficients_to_cartesian",
    "coefficients_to_symmetric_components",
    "gaussian_multipole_form_factor",
    "multipole_dim",
    "multipole_slice",
    "rotate_coefficients",
    "solid_harmonic_features",
    "stf_basis_numpy",
    "stf_bases",
    "symmetric_multiindices",
    "symmetric_multiplicities",
]
