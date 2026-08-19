from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from zynnova.ml.zivar.config import ZIVARConfig
from zynnova.ml.zivar.model import build_zivar
from zynnova.ml.zivar.types import Conditions, ZIVARBatch, ZIVARPrediction


def _data(dtype: object = torch.float64) -> dict[str, object]:
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.4, 0.1, 0.5], [0.3, 1.6, 0.2]], dtype=dtype
    )
    numbers = torch.tensor([3, 8, 26])
    attrs = torch.zeros(3, 3, dtype=dtype)
    attrs[torch.arange(3), torch.arange(3)] = 1.0
    return {
        "positions": positions,
        "atomic_numbers": numbers,
        "node_attrs": attrs,
        "edge_index": torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long
        ),
        "shifts": torch.zeros(6, 3, dtype=dtype),
        "cell": torch.eye(3, dtype=dtype).mul(12.0).unsqueeze(0),
        "pbc": torch.zeros(1, 3, dtype=torch.bool),
        "batch": torch.zeros(3, dtype=torch.long),
    }


def _config() -> ZIVARConfig:
    config = ZIVARConfig.convolution(
        dft_level="test",
        backbone__atomic_numbers=(3, 8, 26),
        backbone__channels=16,
        backbone__num_interactions=2,
        backbone__num_bessel=4,
        backbone__radial_mlp=(16,),
        electronic__hidden=(16,),
        electronic__polarization_updates=0,
        electronic__reciprocal_kmax=1,
        electronic__oxidation__enabled=False,
        spin__mode="disabled",
        spin__require_spin_input=False,
    )
    return replace(
        config,
        electronic=replace(config.electronic, energy_coupling="full"),
    )


def test_convolution_model_rotation_reflection_and_second_order_training() -> None:
    torch.manual_seed(7)
    model = build_zivar(_config()).double()
    data = _data()
    conditions = {"total_charge": torch.tensor([0.0], dtype=torch.float64)}
    original = model.energy_forces_stress(
        data, conditions=conditions, create_graph=True, compute_stress=False,
        compute_spin_fields=False,
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=torch.float64,
    )
    transformed = dict(data)
    transformed["positions"] = data["positions"] @ rotation.T
    transformed["shifts"] = data["shifts"] @ rotation.T
    rotated = model.energy_forces_stress(
        transformed, conditions=conditions, create_graph=True, compute_stress=False,
        compute_spin_fields=False,
    )
    assert torch.allclose(original["energy"], rotated["energy"], atol=1.0e-10)
    assert torch.allclose(original["charges"], rotated["charges"], atol=1.0e-10)
    assert torch.allclose(
        rotated["forces"], original["forces"] @ rotation.T, atol=1.0e-9
    )
    loss = (
        original["forces"].square().mean()
        + original["charges"].square().mean()
        + original["magmoms"].square().mean()
    )
    loss.backward()
    assert all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_convolution_manifest_and_backbone_lock() -> None:
    model = build_zivar(_config()).double().seal_backbone()
    manifest = model.backbone_manifest
    assert manifest["kind"] == "convolution"
    assert manifest["capabilities"]["maximum_ell"] == 0
    with pytest.raises(RuntimeError, match="sealed"):
        model.backbone = model.backbone


def test_noncollinear_default_does_not_instantiate_or_run_scalar_moment_head() -> None:
    base = _config()
    config = replace(
        base,
        spin=replace(base.spin, mode="spin_lattice", require_spin_input=True),
    )
    model = build_zivar(config).double()
    assert model.magnetic.auxiliary_magnitude is None
    conditions = {
        "total_charge": torch.tensor([0.0], dtype=torch.float64),
        "spin_vectors": torch.tensor(
            [[0.8, 0.1, -0.2], [-0.3, 0.7, 0.4], [0.2, -0.5, 0.6]],
            dtype=torch.float64,
        ),
    }
    output = model(_data(), conditions=conditions)
    assert "backup_magnitude_head" not in output


def test_explicit_auxiliary_magnitude_head_is_trainable() -> None:
    base = _config()
    config = replace(
        base,
        electronic=replace(
            base.electronic,
            method="polar",
            polarization_updates=1,
        ),
        spin=replace(base.spin, mode="magnitude_auxiliary"),
    )
    model = build_zivar(config).double()
    assert model.magnetic.auxiliary_magnitude is not None
    output = model(
        _data(),
        conditions={"total_charge": torch.tensor([0.0], dtype=torch.float64)},
    )
    assert output["magmoms"] is output["backup_magnitude_head"]
    output["magmoms"].sum().backward()
    gradients = [
        parameter.grad
        for parameter in model.magnetic.auxiliary_magnitude.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert any(bool(torch.count_nonzero(gradient)) for gradient in gradients)


def test_external_spin_zeeman_term_is_counted_exactly_once() -> None:
    base = _config()
    config = replace(
        base,
        spin=replace(base.spin, mode="spin_lattice", require_spin_input=True),
    )
    model = build_zivar(config).double().eval()
    spins = torch.tensor(
        [[0.8, 0.1, -0.2], [-0.3, 0.7, 0.4], [0.2, -0.5, 0.6]],
        dtype=torch.float64,
    )
    field = torch.tensor([[0.4, -0.2, 0.3]], dtype=torch.float64)
    common = {
        "total_charge": torch.tensor([0.0], dtype=torch.float64),
        "spin_vectors": spins,
    }
    plus = model(_data(), conditions={**common, "external_magnetic_field": field})
    minus = model(_data(), conditions={**common, "external_magnetic_field": -field})
    expected = (
        -2.0
        * config.spin.bohr_magneton_eV_per_T
        * (spins * field).sum()
    )
    assert torch.allclose(
        plus["energy"] - minus["energy"],
        expected.reshape(1),
        atol=1.0e-12,
        rtol=1.0e-10,
    )


def test_typed_batch_and_prediction_are_live_model_interfaces() -> None:
    model = build_zivar(_config()).double().eval()
    data = _data()
    typed = ZIVARBatch(
        positions=data["positions"],
        atomic_numbers=data["atomic_numbers"],
        batch=data["batch"],
        edge_index=data["edge_index"],
        shifts=data["shifts"],
        cell=data["cell"],
        pbc=data["pbc"],
        node_attrs=data["node_attrs"],
    )
    prediction = model.predict_typed(
        typed,
        conditions=Conditions(total_charge=torch.tensor([0.0], dtype=torch.float64)),
    )
    assert isinstance(prediction, ZIVARPrediction)
    assert prediction.state.pack().shape == (3, 12)
    differentiated = model.energy_forces_stress(
        typed,
        conditions=Conditions(total_charge=torch.tensor([0.0], dtype=torch.float64)),
        compute_stress=False,
        compute_spin_fields=False,
    )["prediction"]
    assert isinstance(differentiated, ZIVARPrediction)
    assert differentiated.forces.shape == (3, 3)
