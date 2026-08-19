from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from zynnova.ml.zivar.errors import NonPositiveDefiniteError
from zynnova.ml.zivar.functional import (
    ElectroSpinFunctional,
    ElectroSpinParameters,
    LinearConstraints,
    transform_conditions,
    transform_electronic_state,
    transform_parameters,
    transform_polar_vector,
)
from zynnova.ml.zivar.types import Conditions, ElectronicState, project_stf


class _SelfExcludedCoulombLike:
    """Symmetric negative K; positive onsite keeps the assembled Hessian SPD."""

    symmetric = True
    positive_semidefinite = False

    def __init__(self, strength: float) -> None:
        self.strength = float(strength)

    def matvec(
        self,
        state: ElectronicState,
        *,
        batch: object,
        conditions: Conditions,
    ) -> ElectronicState:
        del batch, conditions
        return ElectronicState(
            q=-self.strength * state.q,
            p=torch.zeros_like(state.p),
            Q=torch.zeros_like(state.Q),
            m=torch.zeros_like(state.m),
        )


def _state(dtype: object = torch.float64) -> ElectronicState:
    q = torch.tensor([0.4, -0.1, -0.3], dtype=dtype)
    p = torch.tensor(
        [[0.2, -0.3, 0.1], [0.1, 0.4, -0.2], [-0.3, 0.2, 0.5]],
        dtype=dtype,
    )
    raw_Q = torch.tensor(
        [
            [[0.4, 0.2, -0.1], [0.2, -0.3, 0.3], [-0.1, 0.3, 0.2]],
            [[-0.2, 0.4, 0.1], [0.4, 0.5, -0.2], [0.1, -0.2, 0.1]],
            [[0.1, -0.3, 0.2], [-0.3, 0.2, 0.4], [0.2, 0.4, -0.5]],
        ],
        dtype=dtype,
    )
    m = torch.tensor(
        [[0.3, 0.1, -0.2], [-0.4, 0.2, 0.5], [0.1, -0.3, 0.4]],
        dtype=dtype,
    )
    return ElectronicState(q=q, p=p, Q=project_stf(raw_Q), m=m)


def _positive_parameters(dtype: object = torch.float64) -> ElectroSpinParameters:
    zero_vector = torch.zeros(3, 3, dtype=dtype)
    return ElectroSpinParameters(
        electronegativity=torch.zeros(3, dtype=dtype),
        hardness=torch.tensor([1.1, 1.3, 1.7], dtype=dtype),
        inverse_polarizability=torch.tensor([0.8, 1.2, 1.5], dtype=dtype),
        inverse_quadrupole_polarizability=torch.tensor([0.6, 0.9, 1.4], dtype=dtype),
        inverse_magnetic_susceptibility=torch.tensor([0.7, 1.1, 1.6], dtype=dtype),
        dipole_drive=zero_vector,
        quadrupole_drive=torch.zeros(3, 3, 3, dtype=dtype),
        magnetic_drive=zero_vector,
        spin_coupling=torch.zeros(3, dtype=dtype),
    )


def test_state_pack_uses_only_five_stf_degrees_of_freedom() -> None:
    state = _state()
    packed = state.pack()
    restored = ElectronicState.from_packed(packed)
    assert packed.shape == (3, 12)
    assert torch.allclose(restored.q, state.q)
    assert torch.allclose(restored.p, state.p)
    assert torch.allclose(restored.Q, state.Q, atol=1.0e-14)
    assert torch.allclose(restored.m, state.m)
    flattened = state.flatten()
    from_flattened = ElectronicState.from_flattened(flattened, atom_count=3)
    assert flattened.shape == (36,)
    assert torch.allclose(from_flattened.pack(), packed, atol=1.0e-14)
    restored.assert_stf()


def test_scalar_energy_positive_onsite_and_charge_constraint() -> None:
    state = _state()
    parameters = _positive_parameters()
    batch = torch.zeros(3, dtype=torch.long)
    functional = ElectroSpinFunctional()
    result = functional(
        state,
        parameters,
        batch=batch,
        short_range_energy=torch.tensor(1.25, dtype=torch.float64),
        external_spin_energy=torch.tensor(0.15, dtype=torch.float64),
    )
    assert result.total.shape == (1,)
    assert result.scalar.ndim == 0
    assert torch.allclose(result.total, result.reconstructed_total(), atol=1.0e-14)
    assert result.charge.item() > 0.0
    assert result.polarization.item() > 0.0
    assert result.quadrupole.item() > 0.0
    assert result.magnetic.item() > 0.0

    constraints = LinearConstraints.total_charge(
        batch, torch.tensor([state.q.sum()], dtype=torch.float64)
    )
    residual = functional.constraint_residual(state, constraints)
    packed_residual = constraints.packed_matrix(state) @ state.pack().reshape(-1)
    packed_residual = packed_residual - constraints.target
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1.0e-14)
    assert torch.allclose(packed_residual, residual, atol=1.0e-14)


def test_stationarity_gradient_matches_the_single_scalar_energy() -> None:
    state = _state()
    packed = state.pack().detach().requires_grad_(True)
    variable = ElectronicState.from_packed(packed)
    parameters = _positive_parameters()
    batch = torch.zeros(3, dtype=torch.long)
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [0.7, -0.4, 0.2], [-0.3, 0.8, 0.5]],
        dtype=torch.float64,
    )
    conditions = Conditions(
        external_spins=torch.tensor(
            [[0.3, 0.1, -0.2], [0.2, -0.4, 0.5], [-0.1, 0.6, 0.2]],
            dtype=torch.float64,
        ),
        external_electric_field=torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64),
        external_magnetic_field=torch.tensor([[1.2, -0.4, 0.7]], dtype=torch.float64),
        electric_field_gradient=project_stf(
            torch.tensor(
                [[[0.3, 0.2, -0.1], [0.2, -0.4, 0.1], [-0.1, 0.1, 0.2]]],
                dtype=torch.float64,
            )
        ),
    )
    functional = ElectroSpinFunctional()
    energy = functional.energy(
        variable,
        parameters,
        batch=batch,
        positions=positions,
        conditions=conditions,
    )
    automatic = torch.autograd.grad(energy, packed)[0]
    analytic = functional.stationarity_gradient(
        variable,
        parameters,
        batch=batch,
        positions=positions,
        conditions=conditions,
    ).pack()
    assert torch.allclose(automatic, analytic, atol=1.0e-12, rtol=1.0e-12)


def test_nonpositive_onsite_response_is_rejected() -> None:
    parameters = _positive_parameters()
    invalid = ElectroSpinParameters(
        electronegativity=parameters.electronegativity,
        hardness=torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
        inverse_polarizability=parameters.inverse_polarizability,
        inverse_quadrupole_polarizability=parameters.inverse_quadrupole_polarizability,
        inverse_magnetic_susceptibility=parameters.inverse_magnetic_susceptibility,
    )
    with pytest.raises(NonPositiveDefiniteError, match="hardness"):
        ElectroSpinFunctional()(
            _state(), invalid, batch=torch.zeros(3, dtype=torch.long)
        )


def test_indefinite_interaction_is_allowed_when_total_curvature_is_positive() -> None:
    state = _state()
    parameters = _positive_parameters()
    batch = torch.zeros(3, dtype=torch.long)
    functional = ElectroSpinFunctional()
    operator = _SelfExcludedCoulombLike(strength=0.25)
    result = functional(
        state,
        parameters,
        batch=batch,
        quadratic_operator=operator,
    )
    hessian_direction = functional.quadratic_matvec(
        state,
        parameters,
        batch=batch,
        quadratic_operator=operator,
    )
    curvature = (state.pack() * hessian_direction.pack()).sum()
    assert result.scalar.ndim == 0
    assert curvature.item() > 0.0


def test_o3_and_time_reversal_invariance_with_external_fields() -> None:
    state = _state()
    dtype = torch.float64
    positions = torch.tensor(
        [[0.2, 0.1, -0.4], [1.1, -0.3, 0.6], [-0.5, 0.9, 0.7]],
        dtype=dtype,
    )
    quadrupole_drive = project_stf(
        torch.tensor(
            [
                [[0.2, -0.1, 0.3], [-0.1, 0.4, 0.2], [0.3, 0.2, -0.3]],
                [[-0.5, 0.2, 0.1], [0.2, 0.1, -0.4], [0.1, -0.4, 0.2]],
                [[0.3, 0.4, -0.2], [0.4, -0.1, 0.1], [-0.2, 0.1, 0.5]],
            ],
            dtype=dtype,
        )
    )
    parameters = ElectroSpinParameters(
        electronegativity=torch.tensor([0.2, -0.1, 0.3], dtype=dtype),
        hardness=torch.tensor([1.2, 1.4, 1.6], dtype=dtype),
        inverse_polarizability=torch.tensor([0.7, 0.9, 1.1], dtype=dtype),
        inverse_quadrupole_polarizability=torch.tensor([0.8, 1.0, 1.3], dtype=dtype),
        inverse_magnetic_susceptibility=torch.tensor([0.6, 1.2, 1.5], dtype=dtype),
        dipole_drive=torch.tensor(
            [[0.2, 0.1, -0.3], [-0.4, 0.2, 0.1], [0.3, -0.2, 0.5]],
            dtype=dtype,
        ),
        quadrupole_drive=quadrupole_drive,
        magnetic_drive=torch.tensor(
            [[0.1, -0.3, 0.2], [0.4, 0.1, -0.2], [-0.2, 0.5, 0.3]],
            dtype=dtype,
        ),
        spin_coupling=torch.tensor([0.4, -0.2, 0.3], dtype=dtype),
    )
    conditions = Conditions(
        external_spins=torch.tensor(
            [[0.8, 0.2, -0.1], [-0.3, 0.7, 0.4], [0.2, -0.5, 0.9]],
            dtype=dtype,
        ),
        external_electric_field=torch.tensor([[0.1, -0.2, 0.3]], dtype=dtype),
        external_magnetic_field=torch.tensor([[-0.2, 0.4, 0.1]], dtype=dtype),
        electric_field_gradient=project_stf(
            torch.tensor(
                [[[0.2, 0.1, -0.2], [0.1, -0.3, 0.4], [-0.2, 0.4, 0.5]]],
                dtype=dtype,
            )
        ),
        electric_field_origin=torch.tensor([[0.3, -0.1, 0.2]], dtype=dtype),
    )
    batch = torch.zeros(3, dtype=torch.long)
    functional = ElectroSpinFunctional()
    original = functional(
        state,
        parameters,
        batch=batch,
        positions=positions,
        conditions=conditions,
        short_range_energy=torch.tensor(0.7, dtype=dtype),
        external_spin_energy=torch.tensor(-0.15, dtype=dtype),
    ).scalar

    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=dtype))
    reflected = functional(
        transform_electronic_state(state, reflection),
        transform_parameters(parameters, reflection),
        batch=batch,
        positions=transform_polar_vector(positions, reflection),
        conditions=transform_conditions(conditions, reflection),
        short_range_energy=torch.tensor(0.7, dtype=dtype),
        external_spin_energy=torch.tensor(-0.15, dtype=dtype),
    ).scalar
    assert torch.allclose(original, reflected, atol=1.0e-12, rtol=1.0e-12)

    identity = torch.eye(3, dtype=dtype)
    time_reversed = functional(
        transform_electronic_state(state, identity, time_reversal=True),
        transform_parameters(parameters, identity, time_reversal=True),
        batch=batch,
        positions=positions,
        conditions=transform_conditions(conditions, identity, time_reversal=True),
        short_range_energy=torch.tensor(0.7, dtype=dtype),
        external_spin_energy=torch.tensor(-0.15, dtype=dtype),
    ).scalar
    assert torch.allclose(original, time_reversed, atol=1.0e-12, rtol=1.0e-12)
