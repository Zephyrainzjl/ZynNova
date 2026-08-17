from __future__ import annotations

import math
from dataclasses import asdict

import pytest
import torch
from conftest import compact_test_config, single_structure

from zynnova.ml.zynforge.field import (
    JouleWeaveModelConfig,
    ZynForgeSymmetryPotential,
    check_o3_equivariance,
    constant_potential_jouleweave_calculator,
    jouleweave_calculator,
    load_jouleweave,
)
from zynnova.ml.zynforge.field.chemistry import periodic_table_descriptors
from zynnova.ml.zynforge.field.config import jouleweave_model_config_from_dict
from zynnova.ml.zynforge.field.long_range import LatentEwaldSummation
from zynnova.ml.zynforge.field.physics import (
    COULOMB_EV_A,
    ScreenedChargeEquilibration,
    ZBLRepulsion,
    zbl_pair_energy,
)


@pytest.mark.parametrize(
    ("feature", "enabled"),
    (
        ("use_invariant_scale_context", True),
        ("use_invariant_gated_ffn", True),
        ("use_path_resolved_radial", True),
        ("use_adaptive_rank_gates", True),
        ("use_periodic_table_prior", True),
    ),
)
def test_optional_capacity_is_exact_zero_or_unit_initialized_submodel(
    feature: str,
    enabled: bool,
) -> None:
    common = {
        "num_layers": 1,
        "max_ell": 1,
        "correlation_order": 2,
        "use_invariant_scale_context": False,
        "use_invariant_gated_ffn": False,
        "use_path_resolved_radial": False,
        "use_adaptive_rank_gates": False,
        "use_periodic_table_prior": False,
    }
    torch.manual_seed(13579)
    reference = ZynForgeSymmetryPotential(
        compact_test_config(**common)
    ).double().eval()
    torch.manual_seed(13579)
    candidate_options = dict(common)
    candidate_options[feature] = enabled
    candidate = ZynForgeSymmetryPotential(
        compact_test_config(**candidate_options)
    ).double().eval()
    inputs = single_structure(periodic=False)
    expected = reference.energy_and_forces(inputs)
    actual = candidate.energy_and_forces(inputs)
    torch.testing.assert_close(actual["energy"], expected["energy"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual["forces"], expected["forces"], rtol=0.0, atol=0.0)

    candidate.zero_grad(set_to_none=True)
    candidate(inputs)["energy"].sum().backward()
    new_parameters = [
        parameter
        for name, parameter in candidate.named_parameters()
        if (
            (feature == "use_invariant_scale_context" and "invariant_scale_weight" in name)
            or (feature == "use_invariant_gated_ffn" and "grade_scales" in name)
            or (feature == "use_path_resolved_radial" and "radial_path_coefficients" in name)
            or (feature == "use_adaptive_rank_gates" and "rank_gate_logits" in name)
            or (
                feature == "use_periodic_table_prior"
                and "periodic_table_embedding.projection" in name
            )
        )
    ]
    assert new_parameters
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0).item())
        for parameter in new_parameters
    )


def test_periodic_table_coordinates_are_bounded_and_chemically_structured() -> None:
    table = periodic_table_descriptors(118)
    assert table.shape == (119, 17)
    assert torch.count_nonzero(table[0]) == 0
    assert torch.isfinite(table).all()
    # Period and group coordinates: H=(1,1), He=(1,18), Li=(2,1), O=(2,16).
    torch.testing.assert_close(table[1, 2:4], torch.tensor([1 / 7, 1 / 18]))
    torch.testing.assert_close(table[2, 2:4], torch.tensor([1 / 7, 1.0]))
    torch.testing.assert_close(table[3, 2:4], torch.tensor([2 / 7, 1 / 18]))
    torch.testing.assert_close(table[8, 2:4], torch.tensor([2 / 7, 16 / 18]))


def test_zbl_core_is_unlearnable_and_outer_switch_is_c4() -> None:
    module = ZBLRepulsion(inner_A=0.55, outer_A=1.80, learnable_scale=True).double()
    with torch.no_grad():
        module.raw_scale.fill_(20.0)
    core_distance = torch.tensor([1.0e-3, 0.10], dtype=torch.float64)
    z_i = torch.tensor([3, 28])
    z_j = torch.tensor([8, 8])
    expected = zbl_pair_energy(
        z_i, z_j, core_distance, inner_A=0.55, outer_A=1.80
    )
    torch.testing.assert_close(module(z_i, z_j, core_distance), expected, rtol=0.0, atol=0.0)

    distance = torch.tensor(1.80, dtype=torch.float64, requires_grad=True)
    value = module(torch.tensor(3), torch.tensor(8), distance)
    derivatives = [value]
    for _ in range(4):
        derivatives.append(
            torch.autograd.grad(
                derivatives[-1], distance, create_graph=True, allow_unused=False
            )[0]
        )
    for derivative in derivatives:
        torch.testing.assert_close(
            derivative, torch.zeros_like(derivative), rtol=0.0, atol=2.0e-11
        )


def test_total_charge_and_spin_conditioning_is_extensive() -> None:
    torch.manual_seed(97531)
    model = ZynForgeSymmetryPotential(
        compact_test_config(
            num_layers=1,
            max_ell=1,
            correlation_order=1,
            extensive_state_conditioning=True,
            use_periodic_table_prior=False,
        )
    ).double().eval()
    fragment = single_structure(periodic=False)
    fragment["total_charge"] = torch.tensor([1.0], dtype=torch.float64)
    fragment["spin"] = torch.tensor([0.6], dtype=torch.float64)
    one = model.energy_and_forces(fragment)

    duplicated = {
        "z": torch.cat((fragment["z"], fragment["z"])),
        "pos": torch.cat(
            (fragment["pos"], fragment["pos"] + torch.tensor([20.0, 0.0, 0.0]))
        ),
        "batch": torch.zeros(6, dtype=torch.long),
        "cell": torch.zeros((1, 3, 3), dtype=torch.float64),
        "pbc": torch.zeros((1, 3), dtype=torch.bool),
        "total_charge": torch.tensor([2.0], dtype=torch.float64),
        "spin": torch.tensor([1.2], dtype=torch.float64),
    }
    two = model.energy_and_forces(duplicated)
    torch.testing.assert_close(two["energy"], 2.0 * one["energy"], rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(two["forces"][:3], one["forces"], rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(two["forces"][3:], one["forces"], rtol=2e-11, atol=2e-12)


def test_latent_ewald_finite_kernel_charge_constraints_and_positive_scale() -> None:
    module = LatentEwaldSummation(
        2, softening_A=0.2, init_scale=0.05
    ).double()
    with torch.no_grad():
        module.raw_energy_scale.fill_(-2.0)
    latent = torch.tensor([[1.0, 0.4], [-1.0, -0.4]], dtype=torch.float64)
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    batch = torch.zeros(2, dtype=torch.long)
    cell = torch.zeros((1, 3, 3), dtype=torch.float64)
    pbc = torch.zeros((1, 3), dtype=torch.bool)
    energy, charges = module(
        latent,
        positions,
        batch,
        cell,
        pbc,
        total_charge=torch.tensor([0.0], dtype=torch.float64),
    )
    assert float(module.energy_scale) >= 0.0
    torch.testing.assert_close(charges.sum(dim=0), torch.zeros(2, dtype=torch.float64))
    softened = math.sqrt(2.0**2 + 0.2**2)
    channel_pair_sum = -(1.0**2 + 0.4**2) / softened
    expected = module.energy_scale * COULOMB_EV_A * channel_pair_sum / math.sqrt(2.0)
    torch.testing.assert_close(energy[0], expected, rtol=2e-13, atol=2e-13)

    translated, _ = module(
        latent,
        positions + torch.tensor([7.0, -3.0, 2.0]),
        batch,
        cell,
        pbc,
        total_charge=torch.tensor([0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(translated, energy, rtol=2e-13, atol=2e-13)


def test_qeq_is_variational_charge_conserving_and_periodic_safe_by_default() -> None:
    qeq = ScreenedChargeEquilibration(
        screening_A_inv=0.2,
        softening_A=0.35,
        max_atoms=16,
        min_curvature_eV=1.0e-4,
    ).double()
    chi = torch.tensor([-1.2, 0.3, 0.8], dtype=torch.float64, requires_grad=True)
    hardness = torch.tensor([2.0, 2.5, 3.0], dtype=torch.float64)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [-0.2, 1.3, 0.1]],
        dtype=torch.float64,
    )
    batch = torch.zeros(3, dtype=torch.long)
    cell = torch.zeros((1, 3, 3), dtype=torch.float64)
    pbc = torch.zeros((1, 3), dtype=torch.bool)
    charges, _, graph_energy = qeq(
        chi,
        hardness,
        positions,
        batch,
        cell,
        pbc,
        torch.tensor([0.7], dtype=torch.float64),
    )
    torch.testing.assert_close(charges.sum(), torch.tensor(0.7, dtype=torch.float64))
    envelope_gradient = torch.autograd.grad(graph_energy.sum(), chi)[0]
    torch.testing.assert_close(envelope_gradient, charges, rtol=2e-10, atol=2e-11)

    with pytest.raises(ValueError, match="periodic QEq"):
        qeq(
            chi,
            hardness,
            positions,
            batch,
            5.0 * torch.eye(3, dtype=torch.float64).unsqueeze(0),
            torch.ones((1, 3), dtype=torch.bool),
            torch.tensor([0.7], dtype=torch.float64),
        )


def test_periodic_latent_ewald_model_preserves_o3_energy_force_contract() -> None:
    torch.manual_seed(31415)
    model = ZynForgeSymmetryPotential(
        compact_test_config(
            num_layers=1,
            max_ell=1,
            correlation_order=1,
            interaction_cutoff_A=3.0,
            use_periodic_table_prior=False,
            use_latent_ewald=True,
            latent_ewald_channels=2,
            latent_ewald_real_cutoff_A=3.0,
            latent_ewald_kmax=1,
        )
    ).double().eval()
    with torch.no_grad():
        model.latent_charge_head[-1].weight.normal_(std=0.08)
    inputs = single_structure(periodic=True)
    inputs["total_charge"] = torch.zeros(1, dtype=torch.float64)
    raw = torch.randn((3, 3), generator=torch.Generator().manual_seed(2718), dtype=torch.float64)
    rotation, _ = torch.linalg.qr(raw)
    if torch.det(rotation) < 0:
        rotation[:, 0] *= -1.0
    report = check_o3_equivariance(
        model,
        inputs,
        rotation,
        atol=3.0e-8,
        rtol=3.0e-7,
    )
    assert report.passed, report


def test_triclinic_latent_ewald_is_lattice_image_and_rotation_invariant() -> None:
    module = LatentEwaldSummation(
        2,
        alpha_A_inv=0.35,
        real_cutoff_A=4.5,
        kmax=2,
        softening_A=0.2,
        init_scale=0.05,
    ).double()
    cell = torch.tensor(
        [[4.0, 0.0, 0.0], [1.2, 3.8, 0.0], [0.4, 0.7, 4.2]],
        dtype=torch.float64,
    )
    fractional = torch.tensor(
        [[0.1, 0.2, 0.3], [0.7, 0.8, 0.2], [0.4, 0.3, 0.9]],
        dtype=torch.float64,
    )
    positions = fractional @ cell
    latent = torch.tensor(
        [[0.4, -0.2], [-0.1, 0.6], [0.7, -0.3]],
        dtype=torch.float64,
    )
    batch = torch.zeros(3, dtype=torch.long)
    pbc = torch.ones((1, 3), dtype=torch.bool)
    total_charge = torch.zeros(1, dtype=torch.float64)

    reference, _ = module(
        latent, positions, batch, cell[None], pbc, total_charge
    )
    image_positions = positions.clone()
    image_positions[1] = image_positions[1] + cell[1] - 2.0 * cell[0]
    image_energy, _ = module(
        latent,
        image_positions,
        batch,
        cell[None],
        pbc,
        total_charge,
    )
    torch.testing.assert_close(image_energy, reference, rtol=0.0, atol=3.0e-14)

    raw = torch.randn(
        (3, 3),
        generator=torch.Generator().manual_seed(4242),
        dtype=torch.float64,
    )
    rotation, _ = torch.linalg.qr(raw)
    if torch.det(rotation) < 0:
        rotation[:, 0] *= -1.0
    rotated_energy, _ = module(
        latent,
        positions @ rotation.T,
        batch,
        (cell @ rotation.T)[None],
        pbc,
        total_charge,
    )
    torch.testing.assert_close(rotated_energy, reference, rtol=0.0, atol=8.0e-14)


def test_model_qeq_finite_force_surface_and_reported_charge_are_consistent() -> None:
    torch.manual_seed(16180)
    model = ZynForgeSymmetryPotential(
        compact_test_config(
            num_layers=1,
            max_ell=1,
            correlation_order=1,
            use_periodic_table_prior=False,
            use_qeq=True,
        )
    ).double().eval()
    inputs = single_structure(periodic=False)
    inputs["total_charge"] = torch.tensor([0.4], dtype=torch.float64)
    output = model.energy_and_forces(inputs)
    torch.testing.assert_close(
        output["qeq_charges"].sum(),
        torch.tensor(0.4, dtype=torch.float64),
        rtol=0.0,
        atol=2.0e-15,
    )
    assert torch.isfinite(output["energy"]).all()
    assert torch.isfinite(output["forces"]).all()


def test_independent_long_range_functionals_require_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        JouleWeaveModelConfig.specialist(use_qeq=True, use_latent_ewald=True)
    config = JouleWeaveModelConfig.specialist(
        use_qeq=True,
        use_latent_ewald=True,
        allow_combined_electrostatics=True,
    )
    assert config.allow_combined_electrostatics


def test_unverified_capacity_refinements_are_evidence_gated_by_default() -> None:
    config = JouleWeaveModelConfig.universal()
    assert not config.use_invariant_scale_context
    assert not config.use_invariant_gated_ffn
    assert not config.use_path_resolved_radial
    assert not config.use_adaptive_rank_gates
    assert not config.use_periodic_table_prior
    assert config.extensive_state_conditioning


def test_deployment_calculators_reject_unvalidated_nested_compile_paths() -> None:
    model = ZynForgeSymmetryPotential(
        compact_test_config(use_periodic_table_prior=False)
    )
    with pytest.raises(ValueError, match="analytic stress"):
        jouleweave_calculator(
            model,
            analytic_stress=True,
            compile_model=True,
        )
    with pytest.raises(ValueError, match="self-consistent grand-potential"):
        constant_potential_jouleweave_calculator(
            model,  # type: ignore[arg-type]
            compile_model=True,
        )


def test_checkpoint_loading_preserves_or_rejects_changed_physics_contracts(
    tmp_path,
) -> None:
    config = compact_test_config(use_periodic_table_prior=False)
    legacy_config = asdict(config)
    legacy_config.pop("extensive_state_conditioning")
    assert not jouleweave_model_config_from_dict(
        legacy_config
    ).extensive_state_conditioning

    model = ZynForgeSymmetryPotential(config).double().eval()
    unsafe_signature = dict(model.architecture_signature)
    unsafe_signature.pop("extensive_charge_spin_conditioning")
    checkpoint = tmp_path / "unsafe_physics_contract.pt"
    torch.save(
        {
            "model_name": "zynforge",
            "model_config": asdict(config),
            "architecture_signature": unsafe_signature,
            "model_state": model.state_dict(),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="strict Hamiltonian contract"):
        load_jouleweave(
            checkpoint,
            device="cpu",
            dtype="float64",
            use_ema=False,
        )


def test_compiled_conservative_evaluator_matches_eager_energy_force_and_gradients() -> None:
    torch.manual_seed(86420)
    model = ZynForgeSymmetryPotential(
        compact_test_config(
            num_layers=1,
            max_ell=1,
            correlation_order=1,
            use_periodic_table_prior=False,
        )
    ).double().train()
    inputs = single_structure(periodic=False)
    eager = model.energy_and_forces(inputs, create_graph=True)
    compiled_evaluator = model.compile_conservative_evaluator(
        backend="eager",
        dynamic=True,
    )
    compiled = compiled_evaluator(inputs, create_graph=True)
    torch.testing.assert_close(compiled["energy"], eager["energy"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(compiled["forces"], eager["forces"], rtol=0.0, atol=0.0)
    objective = compiled["energy"].square().sum() + compiled["forces"].square().mean()
    objective.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
