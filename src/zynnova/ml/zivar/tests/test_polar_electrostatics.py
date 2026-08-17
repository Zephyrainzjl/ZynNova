from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from zynnova.ml.zivar.config import ElectronicConfig, OxidationConfig
from zynnova.ml.zivar.electrostatics import (
    electrostatic_energy,
    electrostatic_potential_features,
)
from zynnova.ml.zivar.multipoles import SpinMultipoleState


@pytest.mark.parametrize("periodic", [False, True])
def test_analytic_potential_is_energy_functional_derivative(periodic: bool) -> None:
    torch.manual_seed(11)
    config = ElectronicConfig(
        method="polar", energy_coupling="full", density_lmax=2,
        potential_lmax=2, hidden=(16,), polarization_updates=1,
        reciprocal_kmax=3, oxidation=OxidationConfig(enabled=False),
    )
    positions = torch.tensor(
        [[0.2, 0.3, 0.4], [1.6, 0.5, 0.7], [0.4, 1.8, 0.9]],
        dtype=torch.float64,
        requires_grad=True,
    )
    coefficients = (0.03 * torch.randn(3, 9, dtype=torch.float64)).requires_grad_(True)
    coefficients.data[:, 0] -= coefficients.data[:, 0].mean()
    state = SpinMultipoleState(
        coefficients,
        torch.zeros(3, 3, 9, dtype=torch.float64),
    )
    batch = torch.zeros(3, dtype=torch.long)
    cell = torch.eye(3, dtype=torch.float64).mul(8.0).unsqueeze(0) if periodic else None
    pbc = torch.ones(1, 3, dtype=torch.bool) if periodic else None
    energy = electrostatic_energy(state, positions, batch, cell, pbc, config).energy.sum()
    reference = torch.autograd.grad(energy, coefficients, create_graph=True)[0]
    analytic = electrostatic_potential_features(
        state, positions, batch, cell, pbc, config
    ).coefficients
    assert torch.allclose(reference, analytic, atol=2.0e-9, rtol=2.0e-8)
    second = torch.autograd.grad(analytic.square().sum(), positions, allow_unused=False)[0]
    assert torch.isfinite(second).all()


def test_uniform_external_field_enters_monopole_and_dipole_features() -> None:
    config = ElectronicConfig(
        method="polar", energy_coupling="full", density_lmax=1,
        potential_lmax=1, hidden=(16,), polarization_updates=1,
        oxidation=OxidationConfig(enabled=False),
    )
    positions = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    state = SpinMultipoleState(
        torch.zeros(1, 4, dtype=torch.float64),
        torch.zeros(1, 3, 4, dtype=torch.float64),
    )
    field = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
    result = electrostatic_potential_features(
        state, positions, torch.zeros(1, dtype=torch.long), None, None, config,
        conditions={"external_electric_field": field},
    ).coefficients
    assert torch.allclose(result[:, 0], torch.tensor([-2.0], dtype=torch.float64))
    assert torch.linalg.vector_norm(result[:, 1:]).item() > 0


@pytest.mark.parametrize("lmax", [0, 1, 3, 4])
def test_all_supported_multipole_ranks_have_exact_open_boundary_derivative(lmax: int) -> None:
    torch.manual_seed(31 + lmax)
    config = ElectronicConfig(
        method="polar", energy_coupling="full", density_lmax=lmax,
        potential_lmax=lmax, hidden=(8,), polarization_updates=1,
        reciprocal_kmax=1, oxidation=OxidationConfig(enabled=False),
    )
    positions = torch.tensor(
        [[0.2, 0.1, 0.3], [1.4, 0.7, 0.9]], dtype=torch.float64
    )
    dimension = (lmax + 1) ** 2
    coefficients = (0.01 * torch.randn(2, dimension, dtype=torch.float64)).requires_grad_(True)
    state = SpinMultipoleState(
        coefficients, torch.zeros(2, 3, dimension, dtype=torch.float64)
    )
    batch = torch.zeros(2, dtype=torch.long)
    energy = electrostatic_energy(state, positions, batch, None, None, config).energy.sum()
    reference = torch.autograd.grad(energy, coefficients)[0]
    analytic = electrostatic_potential_features(
        state, positions, batch, None, None, config
    ).coefficients
    assert torch.allclose(reference, analytic, atol=2.0e-8, rtol=2.0e-7)
