from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from zynnova.ml.zivar.config import ElectronicConfig, OxidationConfig
from zynnova.ml.zivar.electronic import StableElectronicModel
from zynnova.ml.zivar.electrostatics import gaussian_monopole_kernel_and_field
from zynnova.ml.zivar.multipoles import multipole_slice, rotate_coefficients
from zynnova.ml.zivar.polar import positive_fukui_projection


def _randomize(model: object) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.05)


def _payload(dtype: object = torch.float64) -> dict[str, object]:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.2, 0.0], [0.0, 0.0, 0.0], [0.4, 1.4, 0.2]],
        dtype=dtype,
        requires_grad=True,
    )
    spins = torch.tensor(
        [[0.0, 0.0, 1.0], [0.2, 0.1, -0.8], [0.0, 1.2, 0.0], [0.1, -0.7, 0.3]],
        dtype=dtype,
    )
    return {
        "features": torch.randn(4, 8, dtype=dtype, requires_grad=True),
        "positions": positions,
        "atomic_numbers": torch.tensor([3, 8, 26, 8]),
        "batch": torch.tensor([0, 0, 1, 1]),
        "edge_index": torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        "shifts": torch.zeros(4, 3, dtype=dtype),
        "cell": None,
        "pbc": None,
        "cutoff_A": 5.0,
        "conditions": {
            "total_charge": torch.tensor([0.0, 1.0], dtype=dtype),
            "total_magnetization": torch.stack((spins[:2].sum(0), spins[2:].sum(0))),
        },
        "spin_vectors": spins,
    }


def test_fukui_projection_is_exact_and_positive() -> None:
    raw = torch.tensor([2.0, -4.0, 3.0, 1.0], dtype=torch.float64)
    logits = torch.tensor([-20.0, 0.0, 2.0, -1.0], dtype=torch.float64)
    batch = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([0.0, -1.0], dtype=torch.float64)
    charge, weights = positive_fukui_projection(
        raw, logits, batch, target, floor=1.0e-4
    )
    assert torch.allclose(
        torch.stack((charge[:2].sum(), charge[2:].sum())), target, atol=1.0e-12
    )
    assert bool(torch.all(weights > 0))


def test_gaussian_field_is_hessian_safe_at_coincident_centres() -> None:
    vector = torch.zeros(2, 2, 3, dtype=torch.float64, requires_grad=True)
    kernel, field = gaussian_monopole_kernel_and_field(vector, 0.75)
    first = torch.autograd.grad(kernel.sum() + field.square().sum(), vector, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), vector)[0]
    assert torch.isfinite(kernel).all() and torch.isfinite(field).all()
    assert torch.isfinite(first).all() and torch.isfinite(second).all()


def test_polar_density_conserves_charge_and_spin_after_every_update() -> None:
    config = ElectronicConfig(
        method="polar", energy_coupling="full", density_lmax=2,
        potential_lmax=2, hidden=(24,), polarization_updates=2,
        reciprocal_kmax=2, oxidation=OxidationConfig(enabled=False),
    )
    model = StableElectronicModel(8, (3, 8, 26), config).double()
    _randomize(model)
    payload = _payload()
    result = model(**payload)
    charge_sum = torch.stack((result.state.charges[:2].sum(), result.state.charges[2:].sum()))
    spin_sum = torch.stack((
        result.state.magnetic_moments[:2].sum(0),
        result.state.magnetic_moments[2:].sum(0),
    ))
    assert torch.allclose(charge_sum, payload["conditions"]["total_charge"], atol=1.0e-10)
    assert torch.allclose(spin_sum, payload["conditions"]["total_magnetization"], atol=1.0e-10)
    assert result.potential_coefficients.shape == (4, 9)


@pytest.mark.parametrize("method", ["polar", "direct", "qeq", "fukui_auxiliary"])
def test_all_electronic_paths_have_finite_force_training_gradients(method: str) -> None:
    coupling = "full" if method in {"polar", "qeq"} else (
        "learned" if method in {"fukui_auxiliary", "direct"} else "auxiliary"
    )
    config = ElectronicConfig(
        method=method, energy_coupling=coupling,
        density_lmax=(1 if method == "polar" else 0),
        potential_lmax=(1 if method == "polar" else 0),
        hidden=(16,), polarization_updates=(1 if method in {"polar", "fukui_auxiliary"} else 0),
        reciprocal_kmax=1, qeq_max_atoms=8,
        oxidation=OxidationConfig(enabled=False),
    )
    model = StableElectronicModel(8, (3, 8, 26), config).double()
    _randomize(model)
    payload = _payload()
    result = model(**payload)
    force = -torch.autograd.grad(result.energy.sum(), payload["positions"], create_graph=True)[0]
    objective = force.square().mean() + result.state.charges.square().mean()
    objective.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(value).all()) for value in gradients)


def test_polar_density_is_time_reversal_covariant() -> None:
    config = ElectronicConfig(
        method="polar", energy_coupling="full", density_lmax=1,
        potential_lmax=1, hidden=(16,), polarization_updates=1,
        oxidation=OxidationConfig(enabled=False),
    )
    model = StableElectronicModel(8, (3, 8, 26), config).double().eval()
    _randomize(model)
    payload = _payload()
    forward = model(**payload)
    payload["spin_vectors"] = -payload["spin_vectors"]
    payload["conditions"]["total_magnetization"] = -payload["conditions"]["total_magnetization"]
    reversed_state = model(**payload)
    assert torch.allclose(forward.energy, reversed_state.energy, atol=1.0e-10)
    assert torch.allclose(
        forward.state.spin, -reversed_state.state.spin, atol=1.0e-10
    )


def test_polar_density_is_joint_o3_covariant_under_reflection() -> None:
    torch.manual_seed(17)
    config = ElectronicConfig(
        method="polar", energy_coupling="full", density_lmax=2,
        potential_lmax=2, hidden=(16,), polarization_updates=1,
        reciprocal_kmax=1, oxidation=OxidationConfig(enabled=False),
    )
    model = StableElectronicModel(8, (3, 8, 26), config).double().eval()
    _randomize(model)
    payload = _payload()
    original = model(**payload)
    transform = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    det = torch.linalg.det(transform)
    reflected = dict(payload)
    reflected["positions"] = payload["positions"] @ transform.T
    reflected["shifts"] = payload["shifts"] @ transform.T
    reflected["spin_vectors"] = det * payload["spin_vectors"] @ transform.T
    reflected["conditions"] = dict(payload["conditions"])
    reflected["conditions"]["total_magnetization"] = (
        det * payload["conditions"]["total_magnetization"] @ transform.T
    )
    transformed = model(**reflected)
    assert torch.allclose(original.energy, transformed.energy, atol=2.0e-9)
    for ell in range(3):
        block = multipole_slice(ell)
        expected_charge = rotate_coefficients(
            original.state.charge[:, block], transform, ell
        )
        assert torch.allclose(
            transformed.state.charge[:, block], expected_charge, atol=2.0e-9
        )
        spatially_rotated = torch.stack(
            [
                rotate_coefficients(original.state.spin[:, axis, block], transform, ell)
                for axis in range(3)
            ],
            dim=1,
        )
        expected_spin = det * torch.einsum("ab,nbm->nam", transform, spatially_rotated)
        assert torch.allclose(
            transformed.state.spin[:, :, block], expected_spin, atol=3.0e-9
        )
