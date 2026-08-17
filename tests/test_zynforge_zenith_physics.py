from __future__ import annotations

from io import BytesIO

import pytest
import torch
from conftest import compact_test_config, single_structure

from zynnova.ml.zynforge.field import (
    ZynForgeSymmetryPotential,
    build_periodic_radius_graph,
    check_conservative_forces,
    check_hessian_symmetry,
    check_o3_equivariance,
    check_permutation_translation_invariance,
    check_stress_energy_derivative,
    check_time_reversal_equivariance,
)
from zynnova.ml.zynforge.field.graph import smooth_cutoff


def _orthogonal_matrices() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(314159)
    matrices = []
    for index in range(3):
        raw = torch.randn((3, 3), generator=generator, dtype=torch.float64)
        matrix, _ = torch.linalg.qr(raw)
        if torch.det(matrix) < 0:
            matrix[:, 0] *= -1.0
        if index == 2:
            matrix = torch.diag(
                torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
            ) @ matrix
        matrices.append(matrix)
    return tuple(matrices)


@pytest.mark.parametrize("matrix", _orthogonal_matrices())
def test_random_o3_energy_and_force_equivariance(
    tiny_model: ZynForgeSymmetryPotential,
    nonperiodic_inputs: dict[str, torch.Tensor],
    matrix: torch.Tensor,
) -> None:
    report = check_o3_equivariance(
        tiny_model,
        nonperiodic_inputs,
        matrix,
        atol=2.0e-8,
        rtol=2.0e-7,
    )
    assert report.passed, report


def test_nonzero_grade_routes_preserve_o3_energy_and_force_equivariance(
    tiny_model: ZynForgeSymmetryPotential,
    nonperiodic_inputs: dict[str, torch.Tensor],
) -> None:
    generator = torch.Generator().manual_seed(424242)
    with torch.no_grad():
        for block in tiny_model.backbone.blocks:
            block.edge_attention.output_projection.weight.copy_(
                torch.randn(
                    block.edge_attention.output_projection.weight.shape,
                    generator=generator,
                    dtype=torch.float64,
                )
                * 0.15
            )
            block.edge_attention.grade_projection.weight.copy_(
                torch.randn(
                    block.edge_attention.grade_projection.weight.shape,
                    generator=generator,
                    dtype=torch.float64,
                )
                * 0.15
            )
            assert block.norm.channel_balance is not None
            for balance in block.norm.channel_balance.values():
                balance.normal_(std=0.4)
    for matrix in _orthogonal_matrices():
        report = check_o3_equivariance(
            tiny_model,
            nonperiodic_inputs,
            matrix,
            atol=3.0e-8,
            rtol=3.0e-7,
        )
        assert report.passed, report


def test_nonzero_grade_routes_preserve_time_reversal_contract() -> None:
    torch.manual_seed(5150)
    model = ZynForgeSymmetryPotential(
        compact_test_config(
            num_layers=1,
            max_ell=1,
            correlation_order=1,
            include_time_odd=True,
            use_spin_vectors=True,
        )
    ).double().eval()
    with torch.no_grad():
        for block in model.backbone.blocks:
            block.edge_attention.output_projection.weight.normal_(std=0.15)
            block.edge_attention.grade_projection.weight.normal_(std=0.15)
            assert block.norm.channel_balance is not None
            for balance in block.norm.channel_balance.values():
                balance.normal_(std=0.4)
        model.magmom_vector_head.weight.normal_(std=0.1)
    inputs = single_structure(periodic=False)
    inputs["spin_vectors"] = torch.tensor(
        [[0.4, -0.2, 0.1], [-0.3, 0.5, 0.2], [0.1, 0.2, -0.6]],
        dtype=torch.float64,
    )
    report = check_time_reversal_equivariance(model, inputs)
    assert report.passed, report


@pytest.mark.parametrize("matrix", _orthogonal_matrices())
def test_internal_polar_axial_and_rank_two_fields_transform_exactly(
    tiny_model: ZynForgeSymmetryPotential,
    nonperiodic_inputs: dict[str, torch.Tensor],
    matrix: torch.Tensor,
) -> None:
    reference = tiny_model(nonperiodic_inputs)
    transformed_inputs = dict(nonperiodic_inputs)
    transformed_inputs["pos"] = nonperiodic_inputs["pos"] @ matrix.T
    transformed_inputs["cell"] = nonperiodic_inputs["cell"] @ matrix.T
    transformed = tiny_model(transformed_inputs)

    expected_vector = reference["node_vector"] @ matrix.T
    expected_tensor = torch.einsum(
        "ai,ncij,bj->ncab",
        matrix,
        reference["node_tensor"],
        matrix,
    )
    expected_axial = torch.det(matrix) * reference["node_axial_vector"] @ matrix.T
    torch.testing.assert_close(
        transformed["node_vector"], expected_vector, rtol=3.0e-9, atol=3.0e-10
    )
    torch.testing.assert_close(
        transformed["node_tensor"], expected_tensor, rtol=3.0e-9, atol=3.0e-10
    )
    torch.testing.assert_close(
        transformed["node_axial_vector"], expected_axial, rtol=3.0e-9, atol=3.0e-10
    )
    assert float(reference["node_axial_vector"].abs().max()) > 1.0e-10


def test_translation_permutation_and_zero_net_force(
    tiny_model: ZynForgeSymmetryPotential,
    nonperiodic_inputs: dict[str, torch.Tensor],
) -> None:
    report = check_permutation_translation_invariance(
        tiny_model,
        nonperiodic_inputs,
        atol=2.0e-8,
        rtol=2.0e-7,
    )
    assert report.passed, report


def test_diagnostics_rebuild_stale_cached_edges(
    tiny_model: ZynForgeSymmetryPotential,
    nonperiodic_inputs: dict[str, torch.Tensor],
) -> None:
    evaluated = tiny_model(nonperiodic_inputs)
    cached = dict(nonperiodic_inputs)
    for name in ("edge_index", "edge_vector", "edge_distance", "edge_shift"):
        if name in evaluated:
            cached[name] = evaluated[name].detach()
    report = check_permutation_translation_invariance(tiny_model, cached)
    assert report.passed, report
    rotation = _orthogonal_matrices()[0]
    equivariance = check_o3_equivariance(tiny_model, cached, rotation)
    assert equivariance.passed, equivariance


def test_force_is_energy_derivative(
    tiny_model: ZynForgeSymmetryPotential,
    nonperiodic_inputs: dict[str, torch.Tensor],
) -> None:
    report = check_conservative_forces(
        tiny_model,
        nonperiodic_inputs,
        step=2.0e-5,
        atol=2.0e-6,
        rtol=2.0e-5,
    )
    assert report.passed, report


def test_hessian_symmetry_and_acoustic_sum_rule(
    tiny_model: ZynForgeSymmetryPotential,
    nonperiodic_inputs: dict[str, torch.Tensor],
) -> None:
    report = check_hessian_symmetry(
        tiny_model,
        nonperiodic_inputs,
        atol=2.0e-7,
        rtol=2.0e-6,
    )
    assert report.passed, report


def test_stress_matches_symmetric_strain_derivative(
    tiny_model: ZynForgeSymmetryPotential,
) -> None:
    report = check_stress_energy_derivative(
        tiny_model,
        single_structure(periodic=True),
        step=2.0e-5,
        atol=2.0e-5,
        rtol=2.0e-4,
    )
    assert report.passed, report


def test_axis_aligned_bonds_have_finite_second_derivatives() -> None:
    torch.manual_seed(7)
    model = ZynForgeSymmetryPotential(
        compact_test_config(num_layers=1, max_ell=2)
    ).double()
    with torch.no_grad():
        for block in model.backbone.blocks:
            block.edge_attention.output_projection.weight.normal_(std=0.12)
            block.edge_attention.grade_projection.weight.normal_(std=0.12)
            assert block.norm.channel_balance is not None
            for balance in block.norm.channel_balance.values():
                balance.normal_(std=0.35)
    inputs = single_structure(periodic=False)
    inputs["pos"] = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 0.0, 1.3]],
        dtype=torch.float64,
    )
    report = check_hessian_symmetry(model, inputs, atol=5.0e-7, rtol=5.0e-6)
    assert report.finite
    assert report.passed, report


def test_checkpoint_round_trip_preserves_energy_and_force(
    tiny_model: ZynForgeSymmetryPotential,
    nonperiodic_inputs: dict[str, torch.Tensor],
) -> None:
    reference = tiny_model.energy_and_forces(nonperiodic_inputs)
    buffer = BytesIO()
    torch.save(
        {
            "model_config": tiny_model.config,
            "model_state": tiny_model.state_dict(),
        },
        buffer,
    )
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=False)
    restored = ZynForgeSymmetryPotential(payload["model_config"]).double().eval()
    restored.load_state_dict(payload["model_state"], strict=True)
    result = restored.energy_and_forces(nonperiodic_inputs)
    torch.testing.assert_close(result["energy"], reference["energy"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(result["forces"], reference["forces"], rtol=0.0, atol=0.0)
    assert restored.config.architecture_name == "zynforge-zenith"


def test_complete_radius_graph_is_continuous_at_neighbor_rank_exchange() -> None:
    """A nearest-K cap jumps when chemically different neighbours swap rank."""

    coefficients = torch.tensor([0.0, 0.5, 1.0, 3.0], dtype=torch.float64)

    def local_pair_energy(delta: float, max_neighbors: int | None) -> torch.Tensor:
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.80, 0.0],
                [1.0 - delta, 0.0, 0.0],
                [0.0, 0.0, 1.0 + delta],
            ],
            dtype=torch.float64,
        )
        edge_index, _, edge_distance, _ = build_periodic_radius_graph(
            positions,
            cutoff=2.0,
            max_neighbors=max_neighbors,
        )
        receiver, sender = edge_index
        selected = receiver == 0
        return torch.sum(
            coefficients[sender[selected]]
            * smooth_cutoff(edge_distance[selected], 2.0, order=4)
        )

    epsilon = 1.0e-7
    full_jump = abs(
        float(local_pair_energy(epsilon, None) - local_pair_energy(-epsilon, None))
    )
    capped_jump = abs(
        float(local_pair_energy(epsilon, 2) - local_pair_energy(-epsilon, 2))
    )
    assert full_jump < 1.0e-5
    assert capped_jump > 1.0e-2
