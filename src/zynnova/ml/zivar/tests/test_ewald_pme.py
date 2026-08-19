from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from zynnova.ml.zivar.ewald_reference import (
    COULOMB_CONSTANT_EV_A,
    ewald_energy,
    isolated_coulomb_energy,
    plan_ewald,
)
from zynnova.ml.zivar.pme import assign_charges, plan_pme, pme_energy


def _triclinic_system(*, requires_grad: bool = False):
    positions = torch.tensor(
        [[0.2, 0.4, 0.7], [1.7, 1.1, 0.3], [2.3, 2.0, 1.4]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    charges = torch.tensor(
        [0.7, -1.1, 0.4], dtype=torch.float64, requires_grad=requires_grad
    )
    cell = torch.tensor(
        [[5.0, 0.0, 0.0], [0.6, 5.5, 0.0], [0.2, 0.4, 6.0]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, charges, cell, pbc


def test_isolated_direct_energy_and_force_are_exact() -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    charges = torch.tensor([1.0, -2.0], dtype=torch.float64)
    energy = isolated_coulomb_energy(positions, charges, pbc=(False, False, False))
    assert energy.item() == pytest.approx(-COULOMB_CONSTANT_EV_A, abs=1.0e-13)
    force = -torch.autograd.grad(energy, positions)[0]
    expected = torch.tensor(
        [[COULOMB_CONSTANT_EV_A / 2.0, 0.0, 0.0],
         [-COULOMB_CONSTANT_EV_A / 2.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    assert torch.allclose(force, expected, atol=1.0e-13, rtol=1.0e-13)


def test_direct_ewald_converges_in_a_triclinic_cell() -> None:
    positions, charges, cell, pbc = _triclinic_system()
    loose = ewald_energy(
        positions, charges, cell, pbc, plan_ewald(cell, 1.0e-5)
    ).energy
    tight = ewald_energy(
        positions, charges, cell, pbc, plan_ewald(cell, 1.0e-9)
    ).energy
    reference = ewald_energy(
        positions, charges, cell, pbc, plan_ewald(cell, 1.0e-11)
    ).energy
    loose_error = torch.abs(loose - reference)
    tight_error = torch.abs(tight - reference)
    assert tight_error.item() < 5.0e-9
    assert tight_error.item() < loose_error.item() / 1_000.0


def test_ewald_terms_include_self_and_tinfoil_background() -> None:
    positions, _, cell, pbc = _triclinic_system()
    charges = torch.tensor([0.7, -0.8, 0.4], dtype=torch.float64)
    parameters = plan_ewald(cell, 1.0e-8)
    result = ewald_energy(positions, charges, cell, pbc, parameters)
    expected_self = (
        -COULOMB_CONSTANT_EV_A
        * parameters.alpha_inv_A
        / math.sqrt(math.pi)
        * charges.square().sum()
    )
    expected_background = (
        -COULOMB_CONSTANT_EV_A
        * math.pi
        * charges.sum().square()
        / (
            2.0
            * parameters.alpha_inv_A**2
            * torch.linalg.det(cell).abs()
        )
    )
    assert torch.allclose(result.self_energy, expected_self)
    assert torch.allclose(result.background_energy, expected_background)
    assert torch.allclose(
        result.energy,
        result.real_energy
        + result.reciprocal_energy
        + result.self_energy
        + result.background_energy,
    )
    with pytest.raises(ValueError, match="charged periodic cell"):
        ewald_energy(
            positions,
            charges,
            cell,
            pbc,
            parameters,
            neutralizing_background=False,
        )


def test_direct_ewald_position_charge_and_cell_paths_are_differentiable() -> None:
    positions, charges, cell, pbc = _triclinic_system(requires_grad=True)
    parameters = plan_ewald(cell, 1.0e-7)
    energy = ewald_energy(positions, charges, cell, pbc, parameters).energy
    gradients = torch.autograd.grad(energy, (positions, charges, cell), create_graph=True)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.linalg.vector_norm(gradients[0].sum(dim=0)).item() < 1.0e-10
    curvature = torch.autograd.grad(gradients[0].square().sum(), positions)[0]
    assert torch.isfinite(curvature).all()


def test_pme_uses_a_charge_conserving_fft_mesh_and_converges() -> None:
    positions, charges, cell, pbc = _triclinic_system()
    reference = ewald_energy(
        positions, charges, cell, pbc, plan_ewald(cell, 1.0e-11)
    ).energy
    coarse_plan = plan_pme(
        cell, 1.0e-6, interpolation_order=4, mesh_shape=(32, 32, 32)
    )
    fine_plan = plan_pme(
        cell, 1.0e-6, interpolation_order=4, mesh_shape=(128, 128, 128)
    )
    assigned = assign_charges(positions, charges, cell, coarse_plan.mesh)
    assert torch.allclose(assigned.sum(), charges.sum(), atol=2.0e-14, rtol=0.0)

    coarse = pme_energy(positions, charges, cell, pbc, coarse_plan).energy
    fine = pme_energy(positions, charges, cell, pbc, fine_plan).energy
    coarse_error = torch.abs(coarse - reference)
    fine_error = torch.abs(fine - reference)
    assert fine_error.item() < 2.0e-6
    assert fine_error.item() < coarse_error.item() / 100.0


def test_error_target_refines_the_pme_plan_and_gradients_remain_finite() -> None:
    positions, charges, cell, pbc = _triclinic_system(requires_grad=True)
    loose = plan_pme(cell, 1.0e-3)
    tight = plan_pme(cell, 1.0e-7)
    assert math.prod(tight.mesh.shape) > math.prod(loose.mesh.shape)
    selected = plan_pme(
        cell, 1.0e-5, interpolation_order=4, mesh_shape=(24, 24, 24)
    )
    energy = pme_energy(positions, charges, cell, pbc, selected).energy
    gradients = torch.autograd.grad(energy, (positions, charges, cell))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("flags", [(True, False, True), (False, True, False)])
def test_partial_pbc_is_rejected_explicitly(flags: tuple[bool, bool, bool]) -> None:
    positions, charges, cell, _ = _triclinic_system()
    with pytest.raises(ValueError, match="partial PBC"):
        ewald_energy(positions, charges, cell, flags, plan_ewald(cell, 1.0e-5))
    with pytest.raises(ValueError, match="partial PBC"):
        pme_energy(positions, charges, cell, flags, plan_pme(cell, 1.0e-5))
    with pytest.raises(ValueError, match="partial PBC"):
        isolated_coulomb_energy(positions, charges, flags)
