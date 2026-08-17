from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from zynnova.ml.zivar.config import SpinConfig
from zynnova.ml.zivar.magnetism import SpinLatticeHamiltonian, magnetic_torque


def _randomize(model: object) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.08)


def _payload(dtype: object = torch.float64) -> dict[str, object]:
    positions = torch.tensor(
        [
            [0.2, 0.4, 0.1], [1.4, 0.3, 0.7],
            [0.5, 1.7, 0.6], [0.8, 0.6, 2.0],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    spins = torch.tensor(
        [
            [1.0, 0.2, 0.1], [-0.3, 0.8, 0.4],
            [0.1, -0.4, 1.1], [0.5, 0.6, -0.7],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    return {
        "features": torch.randn(4, 8, dtype=dtype),
        "positions": positions,
        "batch": torch.zeros(4, dtype=torch.long),
        "edge_index": torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
            ],
            dtype=torch.long,
        ),
        "shifts": torch.zeros(12, 3, dtype=dtype),
        "cutoff_A": 5.0,
        "conditions": {},
        "spin_vectors": spins,
    }


def test_spin_hamiltonian_is_time_reversal_invariant() -> None:
    torch.manual_seed(3)
    model = SpinLatticeHamiltonian(8, SpinConfig(hidden=(24,))).double()
    _randomize(model)
    payload = _payload()
    forward = model(**payload)
    assert float(forward.dmi_energy.abs().max()) > 1.0e-14
    payload["spin_vectors"] = -payload["spin_vectors"]
    reverse = model(**payload)
    assert torch.allclose(forward.energy, reverse.energy, atol=1.0e-11)
    assert torch.allclose(forward.dmi_energy, reverse.dmi_energy, atol=1.0e-11)


def test_joint_o3_covariance_including_reflection() -> None:
    torch.manual_seed(5)
    model = SpinLatticeHamiltonian(8, SpinConfig(hidden=(24,))).double()
    _randomize(model)
    payload = _payload()
    original = model(**payload)
    transform = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    assert torch.linalg.det(transform) < 0
    # Spins are axial: det(R) R S.
    rotated = dict(payload)
    rotated["positions"] = payload["positions"] @ transform.T
    rotated["shifts"] = payload["shifts"] @ transform.T
    rotated["spin_vectors"] = torch.linalg.det(transform) * payload["spin_vectors"] @ transform.T
    transformed = model(**rotated)
    assert torch.allclose(original.energy, transformed.energy, atol=1.0e-10)


def test_effective_field_and_force_have_finite_second_order_gradients() -> None:
    torch.manual_seed(7)
    model = SpinLatticeHamiltonian(8, SpinConfig(hidden=(24,))).double()
    _randomize(model)
    payload = _payload()
    prediction = model(**payload)
    force, spin_gradient = torch.autograd.grad(
        prediction.energy.sum(),
        (payload["positions"], payload["spin_vectors"]),
        create_graph=True,
    )
    objective = force.square().mean() + spin_gradient.square().mean()
    objective.backward()
    assert all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    torque = magnetic_torque(payload["spin_vectors"], -spin_gradient)
    assert torch.isfinite(torque).all()


def test_zeeman_gradient_has_exact_tesla_conversion() -> None:
    config = SpinConfig(
        hidden=(8,), exchange=False, biquadratic_exchange=False,
        anisotropy=False, dmi=False, neural_high_order=False,
        onsite_landau=False, external_field=True,
    )
    model = SpinLatticeHamiltonian(8, config).double()
    payload = _payload()
    field = torch.tensor([[1.2, -0.4, 2.0]], dtype=torch.float64)
    payload["conditions"] = {"external_magnetic_field": field}
    prediction = model(**payload)
    gradient = torch.autograd.grad(prediction.energy.sum(), payload["spin_vectors"])[0]
    effective_field_T = -gradient / config.bohr_magneton_eV_per_T
    assert torch.allclose(effective_field_T, field.expand_as(effective_field_T), atol=1.0e-12)
