from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from zynnova.ml.zivar.config import ZIVARConfig
from zynnova.ml.zivar.errors import SCFConvergenceError
from zynnova.ml.zivar.losses import ZIVARLoss
from zynnova.ml.zivar.model import build_zivar


def _complete_edges(atom_count: int) -> torch.Tensor:
    return torch.tensor(
        [
            [
                source
                for source in range(atom_count)
                for target in range(atom_count)
                if source != target
            ],
            [
                target
                for source in range(atom_count)
                for target in range(atom_count)
                if source != target
            ],
        ],
        dtype=torch.long,
    )


def _structure(
    positions: torch.Tensor | None = None,
    atomic_numbers: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if positions is None:
        positions = torch.tensor(
            [[0.15, -0.20, 0.25], [1.30, 0.10, -0.35], [-0.40, 1.45, 0.30]],
            dtype=torch.float64,
        )
    if atomic_numbers is None:
        atomic_numbers = torch.tensor([1, 8, 1], dtype=torch.long)
    edge_index = _complete_edges(int(positions.shape[0]))
    return {
        "positions": positions,
        "atomic_numbers": atomic_numbers,
        "edge_index": edge_index,
        "shifts": positions.new_zeros((edge_index.shape[1], 3)),
        "pbc": torch.zeros((1, 3), dtype=torch.bool),
        "batch": torch.zeros(positions.shape[0], dtype=torch.long),
    }


def _config(
    *,
    spin_lattice: bool = False,
    max_iter: int = 128,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-11,
    electrostatic_boundary: str = "isolated",
    constrain_total_magnetization: bool = False,
) -> ZIVARConfig:
    return ZIVARConfig.convolution(
        dft_level="variational-integration-test",
        backbone__atomic_numbers=(1, 8),
        backbone__channels=8,
        backbone__num_interactions=1,
        backbone__num_bessel=3,
        backbone__radial_mlp=(8,),
        backbone__pair_repulsion=False,
        electronic__hidden=(8,),
        electronic__radial_basis=3,
        electronic__potential_scale_eV=1.0,
        electronic__oxidation__enabled=False,
        scf__atol=atol,
        scf__rtol=rtol,
        scf__max_iter=max_iter,
        electrostatics__boundary=electrostatic_boundary,
        spin__mode="spin_lattice" if spin_lattice else "disabled",
        spin__require_spin_input=spin_lattice,
        spin__constrain_total_magnetization=constrain_total_magnetization,
        spin__hidden=(8,),
    )


def _last_linear(module: torch.nn.Module) -> torch.nn.Linear:
    result = next(
        layer
        for layer in reversed(tuple(module.modules()))
        if isinstance(layer, torch.nn.Linear)
    )
    return result


def _activate_variational_heads(model: torch.nn.Module, *, magnetic: bool = False) -> None:
    """Replace zero initialisation by deterministic, non-degenerate test coefficients."""

    with torch.no_grad():
        local = _last_linear(model.variational.local)
        local.weight.zero_()
        local.bias.zero_()
        local.weight[0].copy_(torch.linspace(-0.015, 0.020, local.weight.shape[1]))
        local.weight[1].copy_(torch.linspace(-0.12, 0.09, local.weight.shape[1]))
        local.bias[1] = 0.23
        local.bias[2:6] = torch.tensor(
            [1.8, 1.5, 1.4, 1.6], dtype=local.weight.dtype
        )
        local.bias[6] = 0.31

        edge = _last_linear(model.variational.edge)
        edge.weight[0].copy_(torch.linspace(-0.020, 0.025, edge.weight.shape[1]))
        edge.weight[1].copy_(torch.linspace(0.018, -0.022, edge.weight.shape[1]))
        edge.bias.copy_(torch.tensor([0.07, -0.05], dtype=edge.weight.dtype))

        if magnetic:
            onsite = _last_linear(model.magnetic.onsite)
            onsite.weight.fill_(0.004)
            onsite.bias.copy_(
                torch.tensor(
                    [0.12, -0.08, 0.04, 0.09], dtype=onsite.weight.dtype
                )
            )
            pair = _last_linear(model.magnetic.pair)
            pair.weight.fill_(0.003)
            pair.bias.copy_(
                torch.tensor([0.13, -0.07, 0.08, 0.06], dtype=pair.weight.dtype)
            )
            high_order = _last_linear(model.magnetic.high_order)
            high_order.weight.fill_(0.002)
            high_order.bias.fill_(0.015)


def _model(*, spin_lattice: bool = False, **config_values: object) -> torch.nn.Module:
    torch.manual_seed(417)
    model = build_zivar(_config(spin_lattice=spin_lattice, **config_values)).double()
    _activate_variational_heads(model, magnetic=spin_lattice)
    return model


def test_nonzero_electronegativity_charge_constraint_and_projected_residual() -> None:
    model = _model()
    target = torch.tensor([0.37], dtype=torch.float64)
    data = _structure()
    output = model(data, conditions={"total_charge": target})

    assert float(output["electronegativity"].detach().abs().max()) > 1.0e-4
    assert torch.allclose(output["total_charge"], target, atol=1.0e-10, rtol=0.0)
    assert (
        float(output["charge_constraint_residual"].detach().abs().max()) <= 1.0e-10
    )
    report = output["scf_report"]
    threshold = model.config.scf.atol + model.config.scf.rtol * report.initial_residual
    assert report.converged
    assert report.termination == "converged"
    assert report.final_residual <= threshold
    assert report.constraint_residual <= 1.0e-10
    assert (
        report.energy_error
        <= model.config.scf.energy_atol_eV_per_atom * data["positions"].shape[0]
    )
    assert torch.allclose(
        output["electronic_residual"],
        output["electronic_residual"].new_full((1,), report.final_residual),
        atol=0.0,
        rtol=0.0,
    )


def test_configured_electrostatic_boundary_cannot_silently_change_backend() -> None:
    model = _model(electrostatic_boundary="periodic_3d").eval()
    with pytest.raises(ValueError, match="periodic_3d.*requires pbc"):
        model(
            _structure(),
            conditions={"total_charge": torch.tensor([0.0], dtype=torch.float64)},
        )


def test_total_energy_finite_difference_matches_autograd_force() -> None:
    model = _model().eval()
    data = _structure()
    conditions = {"total_charge": torch.tensor([0.37], dtype=torch.float64)}
    analytic = model.energy_forces_stress(
        data,
        conditions=conditions,
        create_graph=False,
        compute_stress=False,
        compute_spin_fields=False,
    )["forces"]

    step = 2.0e-5
    finite_difference = torch.empty_like(analytic)
    for atom in range(data["positions"].shape[0]):
        for component in range(3):
            plus = dict(data)
            minus = dict(data)
            plus["positions"] = data["positions"].clone()
            minus["positions"] = data["positions"].clone()
            plus["positions"][atom, component] += step
            minus["positions"][atom, component] -= step
            energy_plus = model(plus, conditions=conditions)["energy"].sum()
            energy_minus = model(minus, conditions=conditions)["energy"].sum()
            finite_difference[atom, component] = -(
                energy_plus - energy_minus
            ) / (2.0 * step)

    error = (analytic - finite_difference).abs().max()
    assert float(error.detach()) <= 1.0e-5


def test_periodic_pme_scf_force_matches_finite_difference() -> None:
    torch.manual_seed(417)
    config = _config(electrostatic_boundary="periodic_3d")
    config = replace(
        config,
        electrostatics=replace(
            config.electrostatics,
            error_target=1.0e-4,
            real_cutoff_A=3.0,
            alpha_per_A=0.6,
            mesh=(16, 16, 16),
            interpolation_order=4,
        ),
    )
    model = build_zivar(config).double().eval()
    _activate_variational_heads(model)
    data = _structure()
    data["cell"] = torch.tensor(
        [[[8.0, 0.0, 0.0], [0.5, 8.2, 0.0], [0.2, 0.3, 7.8]]],
        dtype=torch.float64,
    )
    data["pbc"] = torch.ones((1, 3), dtype=torch.bool)
    conditions = {"total_charge": torch.tensor([0.0], dtype=torch.float64)}
    analytic = model.energy_forces_stress(
        data,
        conditions=conditions,
        compute_stress=False,
        compute_spin_fields=False,
    )["forces"][0, 0]
    step = 2.0e-5
    plus = dict(data)
    minus = dict(data)
    plus["positions"] = data["positions"].clone()
    minus["positions"] = data["positions"].clone()
    plus["positions"][0, 0] += step
    minus["positions"][0, 0] -= step
    numerical = -(
        model(plus, conditions=conditions)["energy"].sum()
        - model(minus, conditions=conditions)["energy"].sum()
    ) / (2.0 * step)
    output = model(data, conditions=conditions)

    assert output["electrostatic_backend"] == ("pme",)
    assert abs(float(output["total_charge"].detach())) <= 1.0e-10
    assert torch.allclose(analytic, numerical, atol=2.0e-5, rtol=2.0e-4)


def test_full_model_stress_matches_strain_finite_difference() -> None:
    model = _model().eval()
    data = _structure()
    data["cell"] = 6.0 * torch.eye(3, dtype=torch.float64).unsqueeze(0)
    conditions = {"total_charge": torch.tensor([0.37], dtype=torch.float64)}
    stress_xx = model.energy_forces_stress(
        data,
        conditions=conditions,
        compute_stress=True,
        compute_spin_fields=False,
    )["stress"][0, 0]

    step = 2.0e-5

    def strained(sign: float) -> dict[str, torch.Tensor]:
        deformation = torch.eye(3, dtype=torch.float64)
        deformation[0, 0] += sign * step
        value = dict(data)
        value["positions"] = data["positions"] @ deformation.T
        value["shifts"] = data["shifts"] @ deformation.T
        value["cell"] = data["cell"] @ deformation.T
        return value

    numerical = (
        model(strained(1.0), conditions=conditions)["energy"].sum()
        - model(strained(-1.0), conditions=conditions)["energy"].sum()
    ) / (2.0 * step * torch.linalg.det(data["cell"][0]))
    assert torch.allclose(stress_xx, numerical, atol=2.0e-6, rtol=2.0e-4)


def test_charge_label_loss_backpropagates_through_implicit_scf() -> None:
    model = _model()
    output = model(
        _structure(),
        conditions={"total_charge": torch.tensor([0.37], dtype=torch.float64)},
    )
    labels = torch.tensor([0.44, -0.36, 0.29], dtype=torch.float64)
    loss = ZIVARLoss()(output, {"charges": labels})["total"]
    electronic_head = _last_linear(model.variational.local)
    gradient = torch.autograd.grad(loss, electronic_head.weight)[0]

    assert bool(torch.isfinite(loss))
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient[1].abs().max()) > 1.0e-10


def test_force_parameter_gradient_includes_implicit_stationary_state() -> None:
    model = _model().eval()
    data = _structure()
    conditions = {"total_charge": torch.tensor([0.37], dtype=torch.float64)}
    parameter = _last_linear(model.variational.local).weight
    force = model.energy_forces_stress(
        data,
        conditions=conditions,
        create_graph=True,
        compute_stress=False,
        compute_spin_fields=False,
    )["forces"][0, 0]
    analytic = torch.autograd.grad(force, parameter)[0][1, 0]

    step = 2.0e-5
    with torch.no_grad():
        original = parameter[1, 0].clone()
        parameter[1, 0] = original + step
    plus = model.energy_forces_stress(
        data,
        conditions=conditions,
        compute_stress=False,
        compute_spin_fields=False,
    )["forces"][0, 0]
    with torch.no_grad():
        parameter[1, 0] = original - step
    minus = model.energy_forces_stress(
        data,
        conditions=conditions,
        compute_stress=False,
        compute_spin_fields=False,
    )["forces"][0, 0]
    with torch.no_grad():
        parameter[1, 0] = original
    numerical = (plus - minus) / (2.0 * step)

    assert torch.allclose(analytic, numerical, atol=2.0e-5, rtol=2.0e-4)


def test_o3_inversion_and_time_reversal_covariance() -> None:
    model = _model(spin_lattice=True).eval()
    data = _structure()
    spins = torch.tensor(
        [[0.7, -0.2, 0.4], [-0.3, 0.8, 0.1], [0.2, 0.1, -0.6]],
        dtype=torch.float64,
    )
    magnetic_field = torch.tensor([[0.4, -0.3, 0.2]], dtype=torch.float64)
    conditions = {
        "total_charge": torch.tensor([0.21], dtype=torch.float64),
        "spin_vectors": spins,
        "external_magnetic_field": magnetic_field,
    }
    original = model(data, conditions=conditions)

    inverted_data = dict(data)
    inverted_data["positions"] = -data["positions"]
    inverted_data["shifts"] = -data["shifts"]
    inverted = model(inverted_data, conditions=conditions)
    for name in ("energy", "charges", "quadrupoles", "magmom_vectors"):
        assert torch.allclose(
            inverted[name], original[name], atol=1.0e-9, rtol=1.0e-9
        )
    assert torch.allclose(
        inverted["dipoles"], -original["dipoles"], atol=1.0e-9, rtol=1.0e-9
    )

    reversed_conditions = dict(conditions)
    reversed_conditions["spin_vectors"] = -spins
    reversed_conditions["external_magnetic_field"] = -magnetic_field
    time_reversed = model(data, conditions=reversed_conditions)
    for name in ("energy", "charges", "dipoles", "quadrupoles"):
        assert torch.allclose(
            time_reversed[name], original[name], atol=1.0e-9, rtol=1.0e-9
        )
    assert torch.allclose(
        time_reversed["magmom_vectors"],
        -original["magmom_vectors"],
        atol=1.0e-9,
        rtol=1.0e-9,
    )


def test_full_model_spin_field_is_total_energy_derivative() -> None:
    model = _model(spin_lattice=True).eval()
    data = _structure()
    spins = torch.tensor(
        [[0.7, -0.2, 0.4], [-0.3, 0.8, 0.1], [0.2, 0.1, -0.6]],
        dtype=torch.float64,
    )
    conditions = {
        "total_charge": torch.tensor([0.21], dtype=torch.float64),
        "spin_vectors": spins,
        "external_magnetic_field": torch.tensor(
            [[0.4, -0.3, 0.2]], dtype=torch.float64
        ),
    }
    analytic = model.energy_forces_stress(
        data,
        conditions=conditions,
        compute_stress=False,
        compute_spin_fields=True,
    )["effective_field_eV_per_muB"][0, 0]
    step = 2.0e-5
    plus_spins = spins.clone()
    minus_spins = spins.clone()
    plus_spins[0, 0] += step
    minus_spins[0, 0] -= step
    numerical = -(
        model(data, conditions={**conditions, "spin_vectors": plus_spins})["energy"].sum()
        - model(data, conditions={**conditions, "spin_vectors": minus_spins})[
            "energy"
        ].sum()
    ) / (2.0 * step)
    assert torch.allclose(analytic, numerical, atol=2.0e-6, rtol=2.0e-4)


def test_induced_magnetization_constraint_is_exact_and_explicit() -> None:
    model = _model(
        spin_lattice=True,
        constrain_total_magnetization=True,
    ).eval()
    target = torch.tensor([[0.12, -0.08, 0.05]], dtype=torch.float64)
    conditions = {
        "total_charge": torch.tensor([0.21], dtype=torch.float64),
        "spin_vectors": torch.tensor(
            [[0.7, -0.2, 0.4], [-0.3, 0.8, 0.1], [0.2, 0.1, -0.6]],
            dtype=torch.float64,
        ),
        "total_magnetization": target,
    }
    output = model(_structure(), conditions=conditions)
    assert torch.allclose(
        output["total_magnetic_moment"], target, atol=1.0e-10, rtol=0.0
    )
    assert float(output["spin_constraint_residual"].detach().abs().max()) <= 1.0e-10


def test_batch_matches_independent_single_structure_evaluations() -> None:
    model = _model().eval()
    first = _structure()
    second = _structure(
        torch.tensor([[0.35, 0.10, -0.25], [1.10, -0.45, 0.55]], dtype=torch.float64),
        torch.tensor([8, 1], dtype=torch.long),
    )
    offset = first["positions"].shape[0]
    batched = {
        "positions": torch.cat((first["positions"], second["positions"])),
        "atomic_numbers": torch.cat((first["atomic_numbers"], second["atomic_numbers"])),
        "edge_index": torch.cat((first["edge_index"], second["edge_index"] + offset), dim=1),
        "shifts": torch.cat((first["shifts"], second["shifts"])),
        "pbc": torch.zeros((2, 3), dtype=torch.bool),
        "batch": torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
    }
    targets = torch.tensor([0.37, -0.24], dtype=torch.float64)
    combined = model.energy_forces_stress(
        batched,
        conditions={"total_charge": targets},
        compute_stress=False,
        compute_spin_fields=False,
    )
    singles = [
        model.energy_forces_stress(
            structure,
            conditions={"total_charge": targets[index : index + 1]},
            compute_stress=False,
            compute_spin_fields=False,
        )
        for index, structure in enumerate((first, second))
    ]

    assert torch.allclose(
        combined["energy"],
        torch.cat([value["energy"] for value in singles]),
        atol=1.0e-9,
        rtol=1.0e-9,
    )
    assert torch.allclose(
        combined["charges"],
        torch.cat([value["charges"] for value in singles]),
        atol=1.0e-9,
        rtol=1.0e-9,
    )
    assert torch.allclose(
        combined["forces"],
        torch.cat([value["forces"] for value in singles]),
        atol=1.0e-8,
        rtol=1.0e-8,
    )


def test_scf_max_iterations_fails_closed() -> None:
    model = _model(max_iter=1, atol=1.0e-30, rtol=1.0e-30)
    with pytest.raises(SCFConvergenceError) as failure:
        model(
            _structure(),
            conditions={"total_charge": torch.tensor([0.37], dtype=torch.float64)},
        )

    assert not failure.value.report.converged
    assert failure.value.report.iterations <= 1
    assert failure.value.report.termination != "converged"
