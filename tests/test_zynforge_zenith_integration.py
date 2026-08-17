from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from conftest import compact_test_config, single_structure

from zynnova.ml.zynforge.field import (
    JouleWeaveMLIAP,
    JouleWeaveModelConfig,
    average_intercalation_voltage,
    build_periodic_radius_graph,
    diffusion_from_msd,
    jouleweave_calculator,
    load_jouleweave,
    nernst_einstein_conductivity,
)
from zynnova.ml.zynforge.lammps.deployment import export_lammps_bundle
from zynnova.ml.zynforge.lammps.validation import (
    finite_difference_force_check,
    validate_mliap_model,
)


def test_ase_calculator_and_short_md_match_direct_energy_force(tiny_model) -> None:
    from ase import Atoms, units
    from ase.md.verlet import VelocityVerlet

    inputs = single_structure(periodic=False)
    direct = tiny_model.energy_and_forces(inputs)
    atoms = Atoms(
        numbers=inputs["z"].numpy(),
        positions=inputs["pos"].numpy(),
        cell=inputs["cell"][0].numpy(),
        pbc=False,
    )
    atoms.calc = jouleweave_calculator(
        tiny_model,
        device="cpu",
        dtype="float64",
    )
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    np.testing.assert_allclose(energy, direct["energy"].sum().item(), rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(
        forces,
        direct["forces"].detach().numpy(),
        rtol=2e-11,
        atol=2e-12,
    )

    atoms.set_momenta(np.zeros((len(atoms), 3)))
    dynamics = VelocityVerlet(atoms, timestep=0.05 * units.fs)
    dynamics.run(2)
    assert np.isfinite(atoms.get_positions()).all()
    assert np.isfinite(atoms.get_forces()).all()
    assert np.isfinite(atoms.get_potential_energy())


def test_current_checkpoint_load_is_strict_and_prediction_exact(tiny_model, tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_name": "zynforge",
            "model_config": asdict(tiny_model.config),
            "architecture_signature": tiny_model.architecture_signature,
            "model_state": tiny_model.state_dict(),
        },
        checkpoint,
    )
    restored = load_jouleweave(
        checkpoint,
        device="cpu",
        dtype="float64",
        use_ema=False,
    )
    inputs = single_structure(periodic=False)
    expected = tiny_model.energy_and_forces(inputs)
    actual = restored.energy_and_forces(inputs)
    torch.testing.assert_close(actual["energy"], expected["energy"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual["forces"], expected["forces"], rtol=0.0, atol=0.0)


def test_ase_analytic_stress_matches_model_energy_derivative(tiny_model) -> None:
    from ase import Atoms

    inputs = single_structure(periodic=True)
    direct = tiny_model.energy_forces_stress(inputs, compute_stress=True)
    atoms = Atoms(
        numbers=inputs["z"].numpy(),
        positions=inputs["pos"].numpy(),
        cell=inputs["cell"][0].numpy(),
        pbc=True,
    )
    atoms.calc = jouleweave_calculator(
        tiny_model,
        device="cpu",
        dtype="float64",
        analytic_stress=True,
    )
    np.testing.assert_allclose(
        atoms.get_stress(),
        direct["stress"][0].detach().numpy(),
        rtol=3e-11,
        atol=3e-12,
    )


def test_redox_calculator_conserves_reported_total_charge() -> None:
    from ase import Atoms

    from zynnova.ml.zynforge.field import ZynForgeSymmetryPotential

    torch.manual_seed(2468)
    model = ZynForgeSymmetryPotential(
        compact_test_config(
            use_charge_head=True,
            use_oxidation_states=True,
            charge_label_scheme="ddec6",
        )
    ).double().eval()
    atoms = Atoms(
        symbols=["Li", "Ni", "O"],
        positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.2, 1.3, 0.0]],
        cell=6.0 * np.eye(3),
        pbc=False,
    )
    atoms.info["total_charge"] = 1.0
    atoms.calc = jouleweave_calculator(model, device="cpu", dtype="float64")
    charges = np.asarray(atoms.calc.get_property("charges", atoms), dtype=float)
    probabilities = np.asarray(
        atoms.calc.get_property("oxidation_state_probabilities", atoms),
        dtype=float,
    )
    assert charges.sum() == pytest.approx(1.0, rel=0.0, abs=2e-12)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, rtol=2e-13, atol=2e-13)
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()


class _SingleRankMLIAPData:
    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_vector: torch.Tensor,
        element_indices: torch.Tensor,
    ) -> None:
        self.rij = edge_vector
        self.pair_i = edge_index[0]
        self.pair_j = edge_index[1]
        self.elems = element_indices
        self.ntotal = int(element_indices.numel())
        self.iatoms = torch.arange(self.ntotal)
        self.energy: float | None = None
        self.pair_gradient: torch.Tensor | None = None

    def update_pair_forces(self, value: torch.Tensor) -> None:
        self.pair_gradient = value.detach().clone()


def test_lammps_mliap_energy_and_pair_gradient_match_direct_model(tiny_model) -> None:
    inputs = single_structure(periodic=False)
    edge_index, edge_vector, _, _ = build_periodic_radius_graph(
        inputs["pos"],
        inputs["batch"],
        inputs["cell"],
        inputs["pbc"],
        cutoff=tiny_model.config.interaction_cutoff_A,
        max_neighbors=None,
    )
    reference_vector = edge_vector.detach().clone().requires_grad_(True)
    reference = tiny_model.forward_edges(
        inputs["z"],
        edge_index,
        reference_vector,
        batch=inputs["batch"],
        allow_qeq=False,
        return_auxiliary_fields=False,
    )
    reference_energy = reference["atomic_energies"].sum()
    reference_gradient = torch.autograd.grad(reference_energy, reference_vector)[0]

    data = _SingleRankMLIAPData(
        edge_index,
        edge_vector.detach(),
        torch.tensor([0, 1, 2], dtype=torch.long),
    )
    adapter = JouleWeaveMLIAP(
        tiny_model,
        ["H", "O", "C"],
        dtype="float64",
        device="cpu",
    )
    adapter.compute_forces(data)
    assert data.energy is not None
    assert data.pair_gradient is not None
    assert data.energy == pytest.approx(reference_energy.item(), rel=0.0, abs=2e-12)
    torch.testing.assert_close(
        data.pair_gradient,
        reference_gradient,
        rtol=2e-11,
        atol=2e-12,
    )


def test_generic_deployment_force_check_accepts_current_field_model(tiny_model) -> None:
    from zynnova.ml.zynforge.core import AtomicBatch

    inputs = single_structure(periodic=False)
    atoms = AtomicBatch(
        inputs["z"],
        inputs["pos"],
        batch=inputs["batch"],
        cell=inputs["cell"],
        pbc=inputs["pbc"],
    )
    report = finite_difference_force_check(
        tiny_model,
        atoms,
        step=2.0e-5,
    )
    assert report.passed, report


def test_field_lammps_bundle_uses_full_message_passing_halo(tiny_model, tmp_path) -> None:
    assert validate_mliap_model(tiny_model, strict=True) == []
    files = export_lammps_bundle(
        tiny_model,
        tmp_path,
        ["H", "O", "C"],
        atomic_numbers=[1, 8, 6],
        dtype="float64",
        device="cpu",
        neighbor_skin_A=1.5,
    )
    metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
    expected_halo = (
        tiny_model.config.num_layers * tiny_model.config.interaction_cutoff_A
    )
    assert metadata["model_receptive_field_cutoff_A"] == pytest.approx(expected_halo)
    assert metadata["communication_cutoff_A"] == pytest.approx(expected_halo + 1.5)
    commands = files["commands"].read_text(encoding="utf-8")
    assert f"comm_modify cutoff {expected_halo + 1.5:.8g}" in commands
    assert "pair_style mliap unified" in commands


def test_global_electrostatics_are_rejected_but_local_redox_is_deployable() -> None:
    local_redox = JouleWeaveModelConfig.cathode(
        hidden_dim=8,
        num_layers=2,
        max_ell=1,
        correlation_order=1,
        tensor_product_rank=2,
        directional_edge_rank=2,
        num_attention_heads=1,
        num_experts=1,
        expert_top_k=1,
        use_magmoms=False,
    )
    from zynnova.ml.zynforge.field import ZynForgeSymmetryPotential

    warnings = validate_mliap_model(
        ZynForgeSymmetryPotential(local_redox), strict=True
    )
    assert any("partition-charge" in warning for warning in warnings)

    global_model = ZynForgeSymmetryPotential(
        JouleWeaveModelConfig.long_range(
            hidden_dim=8,
            num_layers=2,
            max_ell=1,
            correlation_order=1,
            tensor_product_rank=2,
            directional_edge_rank=2,
            num_attention_heads=1,
            num_experts=1,
            expert_top_k=1,
            use_magmoms=False,
            use_dispersion=False,
        )
    )
    with pytest.raises(ValueError, match="latent-Ewald"):
        validate_mliap_model(global_model, strict=True)


def test_transport_and_cathode_equations_remain_physically_calibrated() -> None:
    time_s = np.linspace(0.0, 1.0e-9, 101)
    expected_diffusion = 2.0e-10
    msd_m2 = 6.0 * expected_diffusion * time_s
    diffusion, uncertainty = diffusion_from_msd(time_s, msd_m2, blocks=5)
    assert diffusion == pytest.approx(expected_diffusion, rel=2e-13)
    assert uncertainty < 1.0e-22

    conductivity = nernst_einstein_conductivity(
        [1000.0, 1000.0],
        [diffusion, 0.5 * diffusion],
        [1.0, -1.0],
        300.0,
    )
    assert np.isfinite(conductivity)
    assert conductivity > 0.0

    voltage = average_intercalation_voltage(
        lithium_rich_energy_eV=-100.0,
        lithium_poor_energy_eV=-95.0,
        lithium_reference_energy_eV_per_atom=-2.0,
        lithium_removed=4,
    )
    assert voltage == pytest.approx(0.75)
