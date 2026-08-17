from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from zynnova.ml.zivar.config import ElectronicConfig
from zynnova.ml.zivar.qeq import gaussian_coulomb_matrix, solve_qeq


def test_gaussian_coulomb_plus_hardness_is_positive_definite() -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.3, 0.2, 0.0], [0.1, 2.0, -0.3]],
        dtype=torch.float64,
    )
    matrix = gaussian_coulomb_matrix(
        positions,
        width_A=0.75,
        coulomb_constant_eV_A=14.3996454784255,
    )
    eigenvalues = torch.linalg.eigvalsh(matrix + 0.5 * torch.eye(3))
    assert float(eigenvalues.min()) > 0.0


def test_qeq_is_exactly_charge_conserving_and_second_order_finite() -> None:
    config = replace(
        ElectronicConfig(method="qeq", polarization_updates=0),
        qeq_max_atoms=16,
    )
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.6, 0.1, 0.0], [0.2, 1.7, 0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    chi = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64, requires_grad=True)
    hardness = torch.full((3,), 1.2, dtype=torch.float64, requires_grad=True)
    result = solve_qeq(
        chi,
        hardness,
        positions,
        torch.zeros(3, dtype=torch.long),
        cell=None,
        pbc=None,
        conditions={"total_charge": torch.tensor([1.0], dtype=torch.float64)},
        config=config,
    )
    assert torch.allclose(result.charges.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert float(result.residual.max()) < 1.0e-9
    force = -torch.autograd.grad(result.graph_energy.sum(), positions, create_graph=True)[0]
    loss = force.square().mean() + result.charges.square().mean()
    gradients = torch.autograd.grad(loss, (chi, hardness), allow_unused=False)
    assert all(bool(torch.isfinite(value).all()) for value in gradients)


def test_qeq_supports_consistent_redundant_fragment_and_fixed_constraints() -> None:
    config = replace(
        ElectronicConfig(method="qeq", polarization_updates=0), qeq_max_atoms=8
    )
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.5, 0.0]],
        dtype=torch.float64,
    )
    result = solve_qeq(
        torch.zeros(3, dtype=torch.float64),
        torch.ones(3, dtype=torch.float64),
        positions,
        torch.zeros(3, dtype=torch.long),
        cell=None,
        pbc=None,
        conditions={
            "total_charge": torch.tensor([1.0], dtype=torch.float64),
            "fragment_membership": torch.tensor(
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
            ),
            "fragment_charge": torch.tensor([0.75, 0.25], dtype=torch.float64),
            "fixed_charge_mask": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            "fixed_charges": torch.tensor([0.0, 0.0, 0.25], dtype=torch.float64),
        },
        config=config,
    )
    assert torch.allclose(result.charges.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(result.charges[:2].sum(), torch.tensor(0.75, dtype=torch.float64))
    assert torch.allclose(result.charges[2], torch.tensor(0.25, dtype=torch.float64))
