"""Differentiable long-range electrostatics for Gaussian STF multipoles.

Both boundary paths use the same atom-centred density whose Fourier damping is
``exp(-sigma**2 k**2 / 2)``.  The corresponding isolated pair Green function
is ``erf(r / (2 sigma)) / r``; it is *not* replaced by a Plummer soft core.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ._deps import require_torch
from .config import ElectronicConfig
from .multipoles import (
    SpinMultipoleState,
    cartesian_to_coefficients,
    coefficients_to_symmetric_components,
    gaussian_multipole_form_factor,
    multipole_slice,
    solid_harmonic_features,
    symmetric_multiindices,
    symmetric_multiplicities,
)

torch = require_torch()


@dataclass(slots=True)
class ElectrostaticResult:
    energy: Any
    atomic_energy: Any
    backend: tuple[str, ...]


@dataclass(slots=True)
class ElectrostaticFeatures:
    """Functional derivative of Coulomb energy with respect to multipoles.

    ``coefficients[i, l*l:(l+1)**2]`` is an O(3)-covariant rank-``l``
    projection of the electrostatic potential at atom ``i``.  It is computed
    analytically rather than by invoking autograd inside the model forward.
    """

    coefficients: Any
    backend: tuple[str, ...]


def _num_graphs(batch: Any, cell: Any) -> int:
    if cell is not None and cell.ndim == 3:
        return int(cell.shape[0])
    return int(batch.max().item()) + 1 if batch.numel() else 0


def reciprocal_vectors(cell: Any, kmax: int) -> Any:
    """Return the complete inversion-symmetric nonzero reciprocal grid."""

    if cell.shape != (3, 3) or abs(float(torch.linalg.det(cell).detach())) < 1.0e-12:
        raise ValueError("periodic electrostatics requires a nonsingular 3x3 cell")
    n = _reciprocal_integer_grid(kmax, str(cell.device), cell.dtype)
    reciprocal = 2.0 * math.pi * torch.linalg.inv(cell).T
    return n @ reciprocal


@lru_cache(maxsize=32)
def _reciprocal_integer_grid(kmax: int, device: str, dtype: Any) -> Any:
    integers = tuple(
        value
        for value in itertools.product(range(-kmax, kmax + 1), repeat=3)
        if value != (0, 0, 0)
    )
    return torch.as_tensor(integers, device=torch.device(device), dtype=dtype)


@lru_cache(maxsize=32)
def _pair_indices(count: int, device: str) -> Any:
    return torch.triu_indices(
        count, count, offset=1, device=torch.device(device)
    )


@lru_cache(maxsize=9)
def _radial_derivative_spec(
    order: int,
) -> tuple[tuple[tuple[float, int, int, int, int], ...], ...]:
    """Cartesian chain-rule terms for a radial function ``f(x*x+y*y+z*z)``."""

    components: list[tuple[tuple[float, int, int, int, int], ...]] = []
    for alpha in symmetric_multiindices(order):
        terms: list[tuple[float, int, int, int, int]] = []
        ranges = tuple(range(value // 2 + 1) for value in alpha)
        for jx, jy, jz in itertools.product(*ranges):
            js = (jx, jy, jz)
            powers = tuple(alpha[axis] - 2 * js[axis] for axis in range(3))
            derivative_order = order - sum(js)
            coefficient = 1.0
            for axis in range(3):
                coefficient *= (
                    math.factorial(alpha[axis])
                    * 2.0 ** powers[axis]
                    / (
                        math.factorial(powers[axis])
                        * math.factorial(js[axis])
                    )
                )
            terms.append(
                (
                    coefficient,
                    powers[0],
                    powers[1],
                    powers[2],
                    derivative_order,
                )
            )
        components.append(tuple(terms))
    return tuple(components)


@lru_cache(maxsize=40)
def _legendre_rule(device: str, dtype: Any) -> tuple[Any, Any]:
    """A small differentiable quadrature used only in the recurrence's hard region."""

    nodes, weights = np.polynomial.legendre.leggauss(32)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    return (
        torch.as_tensor(nodes, device=torch.device(device), dtype=dtype),
        torch.as_tensor(weights, device=torch.device(device), dtype=dtype),
    )


def _gaussian_u_derivatives(vector: Any, max_order: int, width_A: float) -> tuple[Any, ...]:
    """Return ``d^n/du^n erf(sqrt(u)/(2 sigma))/sqrt(u)`` for ``n <= 8``.

    The integral representation

    ``f^(n)(u) = (-1)^n/(sqrt(pi)*sigma*(4*sigma^2)^n)
                  integral_0^1 t^(2n) exp(-u*t^2/(4*sigma^2)) dt``

    is regular at ``u=0``.  Gauss--Legendre quadrature is used for the compact
    interval where upward recurrence suffers cancellation; the exact recurrence
    is used in the asymptotic region.  Every operation remains autograd-visible.
    """

    sigma = float(width_A)
    u = vector.square().sum(-1)
    x = u / (4.0 * sigma * sigma)
    nodes, weights = _legendre_rule(str(vector.device), vector.dtype)
    exponential = torch.exp(-x[:, None] * nodes.square()[None, :])
    quadrature = []
    for order in range(max_order + 1):
        quadrature.append(
            (exponential * (weights * nodes.pow(2 * order))[None, :]).sum(-1)
        )

    safe_x = x.clamp_min(torch.finfo(vector.dtype).tiny)
    root = torch.sqrt(safe_x)
    recurrence = 0.5 * math.sqrt(math.pi) * torch.erf(root) / root
    integrals = [torch.where(x < 32.0, quadrature[0], recurrence)]
    decay = torch.exp(-x)
    for order in range(1, max_order + 1):
        recurrence = ((2 * order - 1) * recurrence - decay) / (2.0 * safe_x)
        integrals.append(torch.where(x < 32.0, quadrature[order], recurrence))

    prefactor = 1.0 / (math.sqrt(math.pi) * sigma)
    scale = 1.0
    derivatives = []
    for order, integral in enumerate(integrals):
        if order:
            scale *= -1.0 / (4.0 * sigma * sigma)
        derivatives.append(prefactor * scale * integral)
    return tuple(derivatives)


def gaussian_monopole_kernel_and_field(
    vector: Any, width_A: float
) -> tuple[Any, Any]:
    """Return the regular Gaussian Coulomb kernel and electric-field vector.

    ``vector`` is ``r_i - r_j`` with shape ``[..., 3]``.  The compact integral
    representation is analytic at coincident centres, unlike differentiating
    ``erf(r / (2 sigma)) / r`` through an explicit vector norm.  Consequently
    both the force and the force-training Hessian stay finite at ``r = 0``.
    """

    if vector.ndim < 1 or vector.shape[-1] != 3:
        raise ValueError("vector must have shape [..., 3]")
    sigma = float(width_A)
    if sigma <= 0.0:
        raise ValueError("width_A must be positive")
    squared_distance = vector.square().sum(-1)
    scaled = squared_distance / (4.0 * sigma * sigma)
    nodes, weights = _legendre_rule(str(vector.device), vector.dtype)
    exponential = torch.exp(-scaled[..., None] * nodes.square())
    kernel = (
        exponential * weights
    ).sum(-1) / (math.sqrt(math.pi) * sigma)
    field_coefficient = (
        exponential * (weights * nodes.square())
    ).sum(-1) / (2.0 * math.sqrt(math.pi) * sigma**3)
    return kernel, field_coefficient[..., None] * vector


def _gaussian_cartesian_derivatives(
    vector: Any, max_order: int, width_A: float
) -> tuple[Any, ...]:
    radial = _gaussian_u_derivatives(vector, max_order, width_A)
    output: list[Any] = []
    x, y, z = vector.unbind(-1)
    for order in range(max_order + 1):
        components: list[Any] = []
        for terms in _radial_derivative_spec(order):
            value = vector.new_zeros(vector.shape[0])
            for coefficient, px, py, pz, derivative_order in terms:
                value = value + (
                    coefficient
                    * x.pow(px)
                    * y.pow(py)
                    * z.pow(pz)
                    * radial[derivative_order]
                )
            components.append(value)
        output.append(torch.stack(components, dim=-1))
    return tuple(output)


@lru_cache(maxsize=25)
def _combined_component_map(left_rank: int, right_rank: int) -> tuple[tuple[int, ...], ...]:
    output = {item: index for index, item in enumerate(
        symmetric_multiindices(left_rank + right_rank)
    )}
    return tuple(
        tuple(
            output[tuple(left[axis] + right[axis] for axis in range(3))]
            for right in symmetric_multiindices(right_rank)
        )
        for left in symmetric_multiindices(left_rank)
    )


@lru_cache(maxsize=80)
def _component_map_torch(left_rank: int, right_rank: int, device: str) -> Any:
    return torch.as_tensor(
        _combined_component_map(left_rank, right_rank),
        device=torch.device(device),
        dtype=torch.long,
    )


@lru_cache(maxsize=40)
def _multiplicity_torch(rank: int, device: str, dtype: Any) -> Any:
    return torch.as_tensor(
        symmetric_multiplicities(rank), device=torch.device(device), dtype=dtype
    )


def _direct_pair_energy(
    components: tuple[Any, ...],
    positions: Any,
    i: Any,
    j: Any,
    config: ElectronicConfig,
) -> Any:
    vector = positions[j] - positions[i]
    lmax = len(components) - 1
    derivatives = _gaussian_cartesian_derivatives(
        vector, 2 * lmax, config.gaussian_width_A
    )
    pair_energy = positions.new_zeros(i.shape[0])
    for left_rank in range(lmax + 1):
        left_multiplicity = _multiplicity_torch(
            left_rank, str(positions.device), positions.dtype
        )
        left = components[left_rank][i] * left_multiplicity
        for right_rank in range(lmax + 1):
            right_multiplicity = _multiplicity_torch(
                right_rank, str(positions.device), positions.dtype
            )
            right = components[right_rank][j] * right_multiplicity
            mapping = _component_map_torch(
                left_rank, right_rank, str(positions.device)
            )
            derivative = derivatives[left_rank + right_rank][:, mapping]
            contraction = torch.einsum("pa,pb,pab->p", left, right, derivative)
            pair_energy = pair_energy + (
                (-1.0) ** left_rank
                * contraction
                / (
                    math.factorial(left_rank)
                    * math.factorial(right_rank)
                )
            )
    return config.coulomb_constant_eV_A * pair_energy


def _periodic_energy(
    coefficients: Any,
    positions: Any,
    cell: Any,
    config: ElectronicConfig,
) -> tuple[Any, Any]:
    k_vectors = reciprocal_vectors(cell, config.reciprocal_kmax)
    k2 = k_vectors.square().sum(-1)
    form = gaussian_multipole_form_factor(
        coefficients, k_vectors, config.gaussian_width_A
    )
    complex_dtype = form.dtype
    phase = torch.exp(-1j * (positions @ k_vectors.T).to(complex_dtype))
    atomic_amplitude = phase * form
    total_amplitude = atomic_amplitude.sum(0)
    volume = torch.abs(torch.linalg.det(cell))
    prefactor = 2.0 * math.pi * config.coulomb_constant_eV_A / volume
    spectral_weight = prefactor / k2
    reciprocal_energy = (spectral_weight * total_amplitude.abs().square()).sum().real
    atomic = (
        spectral_weight.unsqueeze(0)
        * (atomic_amplitude * total_amplitude.conj().unsqueeze(0)).real
    ).sum(-1)
    # The reciprocal density energy contains each atom's isolated Gaussian self
    # interaction.  Removing it makes the periodic and isolated pair-energy
    # conventions identical while retaining interactions with periodic images.
    self_atomic = _isolated_gaussian_self(coefficients, config)
    return reciprocal_energy - self_atomic.sum(), atomic - self_atomic


def _isolated_gaussian_self(coefficients: Any, config: ElectronicConfig) -> Any:
    """Analytic self energy of each isolated Gaussian STF density."""

    lmax = int(round(math.sqrt(coefficients.shape[-1]) - 1))
    sigma = float(config.gaussian_width_A)
    result = coefficients.new_zeros(coefficients.shape[0])
    for ell in range(lmax + 1):
        norm2 = coefficients[:, multipole_slice(ell)].square().sum(-1)
        denominator = (
            2.0 ** (ell + 1)
            * math.sqrt(math.pi)
            * math.factorial(ell)
            * (2 * ell + 1)
            * sigma ** (2 * ell + 1)
        )
        result = result + config.coulomb_constant_eV_A * norm2 / denominator
    return result


def _direct_energy(
    state: SpinMultipoleState,
    positions: Any,
    config: ElectronicConfig,
) -> tuple[Any, Any]:
    count = positions.shape[0]
    atomic = positions.new_zeros(count)
    if count < 2:
        return positions.new_zeros(()), atomic
    pair = _pair_indices(count, str(positions.device))
    i, j = pair[0], pair[1]
    components = tuple(
        coefficients_to_symmetric_components(
            state.charge[:, multipole_slice(ell)], ell
        )
        for ell in range(state.lmax + 1)
    )
    pieces = []
    for start in range(0, i.numel(), config.direct_pair_block):
        stop = min(start + config.direct_pair_block, i.numel())
        pieces.append(
            _direct_pair_energy(
                components, positions, i[start:stop], j[start:stop], config
            )
        )
    pair_energy = torch.cat(pieces)
    atomic.index_add_(0, i, 0.5 * pair_energy)
    atomic.index_add_(0, j, 0.5 * pair_energy)
    return pair_energy.sum(), atomic


def electrostatic_energy(
    state: SpinMultipoleState,
    positions: Any,
    batch: Any,
    cell: Any | None,
    pbc: Any | None,
    config: ElectronicConfig,
) -> ElectrostaticResult:
    """Evaluate periodic Gaussian-density Coulomb or isolated direct multipoles."""

    graph_count = _num_graphs(batch, cell)
    energies = positions.new_zeros(graph_count)
    atomic = positions.new_zeros(positions.shape[0])
    backends: list[str] = []
    if pbc is not None and pbc.ndim == 1:
        pbc = pbc.unsqueeze(0)
    for graph in range(graph_count):
        selection = torch.nonzero(batch == graph, as_tuple=False).flatten()
        if pbc is not None:
            flags = pbc[graph]
            if bool(torch.any(flags).detach()) and not bool(torch.all(flags).detach()):
                raise ValueError(
                    "ZIVAR electrostatics does not silently approximate partial PBC; "
                    "use a fully periodic vacuum cell or a nonperiodic structure"
                )
        periodic = bool(
            cell is not None
            and pbc is not None
            and bool(torch.all(pbc[graph]).detach())
        )
        local_state = SpinMultipoleState(
            state.charge[selection], state.spin[selection]
        )
        if periodic:
            if not config.periodic_background:
                net_charge = local_state.charges.sum()
                if abs(float(net_charge.detach())) > 1.0e-8:
                    raise ValueError(
                        "charged periodic cells require periodic_background=True"
                    )
            energy, per_atom = _periodic_energy(
                local_state.charge, positions[selection], cell[graph], config
            )
            backends.append("reciprocal_gaussian")
        else:
            energy, per_atom = _direct_energy(
                local_state, positions[selection], config
            )
            backends.append(f"direct_gaussian_multipole_l{local_state.lmax}")
        energies[graph] = energy
        atomic[selection] = per_atom
    return ElectrostaticResult(energies, atomic, tuple(backends))


def _coefficient_to_component_transform(ell: int, reference: Any) -> Any:
    identity = torch.eye(
        2 * ell + 1, device=reference.device, dtype=reference.dtype
    )
    return coefficients_to_symmetric_components(identity, ell)


def _isolated_potential_features(
    coefficients: Any,
    positions: Any,
    config: ElectronicConfig,
    potential_lmax: int,
) -> Any:
    """Exact open-boundary Gaussian-multipole potential projections."""

    density_lmax = int(round(math.sqrt(coefficients.shape[-1]) - 1))
    output = positions.new_zeros((positions.shape[0], (potential_lmax + 1) ** 2))
    count = int(positions.shape[0])
    if count < 2:
        return output
    all_atoms = torch.arange(count, device=positions.device)
    target = all_atoms.repeat_interleave(count)
    source = all_atoms.repeat(count)
    keep = target != source
    target, source = target[keep], source[keep]
    source_components = tuple(
        coefficients_to_symmetric_components(
            coefficients[:, multipole_slice(ell)], ell
        )
        for ell in range(density_lmax + 1)
    )
    for start in range(0, target.numel(), config.direct_pair_block):
        stop = min(start + config.direct_pair_block, target.numel())
        local_target = target[start:stop]
        local_source = source[start:stop]
        vector = positions[local_source] - positions[local_target]
        derivatives = _gaussian_cartesian_derivatives(
            vector, density_lmax + potential_lmax, config.gaussian_width_A
        )
        for left_rank in range(potential_lmax + 1):
            left_multiplicity = _multiplicity_torch(
                left_rank, str(positions.device), positions.dtype
            )
            transform = _coefficient_to_component_transform(left_rank, positions)
            block = positions.new_zeros((local_target.shape[0], 2 * left_rank + 1))
            for right_rank in range(density_lmax + 1):
                right_multiplicity = _multiplicity_torch(
                    right_rank, str(positions.device), positions.dtype
                )
                right = (
                    source_components[right_rank][local_source]
                    * right_multiplicity
                )
                mapping = _component_map_torch(
                    left_rank, right_rank, str(positions.device)
                )
                derivative = derivatives[left_rank + right_rank][:, mapping]
                component_gradient = torch.einsum(
                    "pb,pab->pa", right, derivative
                )
                coefficient_gradient = torch.einsum(
                    "ca,a,pa->pc", transform, left_multiplicity,
                    component_gradient,
                )
                block = block + (
                    config.coulomb_constant_eV_A
                    * (-1.0) ** left_rank
                    / (
                        math.factorial(left_rank)
                        * math.factorial(right_rank)
                    )
                    * coefficient_gradient
                )
            output[:, multipole_slice(left_rank)].index_add_(
                0, local_target, block
            )
    return output


def _periodic_potential_features(
    coefficients: Any,
    positions: Any,
    cell: Any,
    config: ElectronicConfig,
    potential_lmax: int,
) -> Any:
    """Exact reciprocal-space derivative of periodic multipole energy."""

    density_lmax = int(round(math.sqrt(coefficients.shape[-1]) - 1))
    k_vectors = reciprocal_vectors(cell, config.reciprocal_kmax)
    k2 = k_vectors.square().sum(-1)
    form = gaussian_multipole_form_factor(
        coefficients, k_vectors, config.gaussian_width_A
    )
    complex_dtype = form.dtype
    phase = torch.exp(-1j * (positions @ k_vectors.T).to(complex_dtype))
    total = (phase * form).sum(0)
    volume = torch.abs(torch.linalg.det(cell))
    spectral_weight = (
        2.0 * math.pi * config.coulomb_constant_eV_A / volume / k2
    )
    damping = torch.exp(
        -0.5 * float(config.gaussian_width_A) ** 2 * k2
    ).to(complex_dtype)
    harmonics = solid_harmonic_features(k_vectors, potential_lmax)
    output = positions.new_zeros((positions.shape[0], (potential_lmax + 1) ** 2))
    for ell in range(potential_lmax + 1):
        basis = (
            (1j ** ell / math.factorial(ell))
            * harmonics[ell].T.to(complex_dtype)
            * damping[None]
        )
        derivative = phase[:, None, :] * basis[None, :, :]
        block = 2.0 * (
            spectral_weight[None, None, :]
            * (derivative * total.conj()[None, None, :]).real
        ).sum(-1)
        if ell <= density_lmax:
            sigma = float(config.gaussian_width_A)
            denominator = (
                2.0 ** (ell + 1)
                * math.sqrt(math.pi)
                * math.factorial(ell)
                * (2 * ell + 1)
                * sigma ** (2 * ell + 1)
            )
            block = block - (
                2.0
                * config.coulomb_constant_eV_A
                / denominator
                * coefficients[:, multipole_slice(ell)]
            )
        output[:, multipole_slice(ell)] = block
    return output


def _add_applied_potential(
    coefficients: Any,
    positions: Any,
    batch: Any,
    conditions: dict[str, Any],
) -> Any:
    """Add external scalar/electric potentials in the same STF convention."""

    result = coefficients
    graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
    electrode = conditions.get("electrode_potential")
    if electrode is not None:
        electrode = electrode.to(device=positions.device, dtype=positions.dtype)
        if electrode.ndim == 0 or electrode.shape == (1,):
            local = electrode.reshape(1).expand(graph_count)[batch]
        elif electrode.shape == (graph_count,):
            local = electrode[batch]
        elif electrode.shape == (positions.shape[0],):
            local = electrode
        else:
            raise ValueError("electrode_potential must be scalar, per graph or per atom")
        result = result.clone()
        result[:, 0] = result[:, 0] - local
    electric = conditions.get("external_electric_field")
    if electric is not None:
        electric = electric.to(device=positions.device, dtype=positions.dtype)
        if electric.shape != (graph_count, 3):
            raise ValueError("external_electric_field must have shape [B,3]")
        origin = conditions.get("electric_field_origin")
        if origin is None:
            origin = positions.new_zeros((graph_count, 3))
        origin = origin.to(device=positions.device, dtype=positions.dtype)
        if result is coefficients:
            result = result.clone()
        result[:, 0] = result[:, 0] - (
            (positions - origin[batch]) * electric[batch]
        ).sum(-1)
        if result.shape[1] >= 4:
            result[:, multipole_slice(1)] = (
                result[:, multipole_slice(1)]
                - cartesian_to_coefficients(electric[batch], 1)
            )
    return result


def electrostatic_potential_features(
    state: SpinMultipoleState,
    positions: Any,
    batch: Any,
    cell: Any | None,
    pbc: Any | None,
    config: ElectronicConfig,
    *,
    conditions: dict[str, Any] | None = None,
) -> ElectrostaticFeatures:
    """Return analytic non-local equivariant potential features.

    Open systems use exact Gaussian translation tensors. Fully periodic
    systems use the analytic reciprocal derivative of the same density energy.
    Partial periodicity is rejected because silently mixing boundary
    conditions changes the Hamiltonian.
    """

    graph_count = _num_graphs(batch, cell)
    output = positions.new_zeros(
        (positions.shape[0], (config.potential_lmax + 1) ** 2)
    )
    backends: list[str] = []
    if pbc is not None and pbc.ndim == 1:
        pbc = pbc.unsqueeze(0)
    for graph in range(graph_count):
        selection = torch.nonzero(batch == graph, as_tuple=False).flatten()
        flags = None if pbc is None else pbc[graph]
        if (
            flags is not None
            and bool(torch.any(flags).detach())
            and not bool(torch.all(flags).detach())
        ):
            raise ValueError("electrostatic potential features require full or no PBC")
        periodic = bool(
            cell is not None and flags is not None and bool(torch.all(flags).detach())
        )
        if periodic:
            local = _periodic_potential_features(
                state.charge[selection], positions[selection], cell[graph],
                config, config.potential_lmax,
            )
            backends.append("reciprocal_gaussian_multipole_potential")
        else:
            local = _isolated_potential_features(
                state.charge[selection], positions[selection], config,
                config.potential_lmax,
            )
            backends.append("direct_gaussian_multipole_potential")
        output[selection] = local
    if config.include_external_field_in_updates:
        output = _add_applied_potential(
            output, positions, batch, dict(conditions or {})
        )
    return ElectrostaticFeatures(output, tuple(backends))


__all__ = [
    "ElectrostaticFeatures",
    "ElectrostaticResult",
    "electrostatic_energy",
    "electrostatic_potential_features",
    "gaussian_monopole_kernel_and_field",
    "reciprocal_vectors",
]
