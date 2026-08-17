from __future__ import annotations

import numpy as np
import pytest
import torch

from zynnova.ml.zynforge.field import (
    GradedIrrepNorm,
    IrrepKey,
    IrrepLayout,
    JouleWeaveModelConfig,
    OrthonormalSmoothRadialBasis,
    SmoothInvariantEdgeAttention,
    architecture_efficiency_report,
    build_distilled_student_config,
    cartesian_to_real_irrep,
    formal_completeness_certificate,
    real_clebsch_gordan,
    real_irrep_to_cartesian,
    real_spherical_harmonics,
    representation_matrix,
)


def test_single_graph_ace_spine_and_finite_basis_certificate(tiny_model) -> None:
    report = architecture_efficiency_report(tiny_model)
    certificate = formal_completeness_certificate(tiny_model)
    layer_count = tiny_model.config.num_layers

    assert report.single_graph_ace_spine
    assert report.complete_edge_cg_layers == layer_count
    assert report.directional_edge_layers == layer_count
    assert report.full_correlation_layers == layer_count
    assert report.grace_finite_tree_basis_contained
    assert certificate.complete_node_product_layers == layer_count
    assert certificate.all_spatial_cg_paths
    assert certificate.all_rooted_tree_topologies
    assert certificate.signed_graph_path_coefficients
    assert not certificate.graph_path_positive_floor
    assert certificate.finite_truncation_feature_complete
    assert not certificate.finite_truncation_linear_coefficient_complete
    assert certificate.unrestricted_rank_bound > tiny_model.config.tensor_product_rank
    assert certificate.single_energy_path
    assert report.complete_radius_graph
    assert report.hard_neighbor_cap is None
    assert report.merged_graded_rms_norm
    assert report.grade_aware_invariant_edge_routing

    class_names = {module.__class__.__name__ for module in tiny_model.modules()}
    assert "IndependentTopologyWordMixer" not in class_names
    assert "FactorizedInvariantEdgeKernel" not in class_names
    assert all(
        block.directional.max_harmonic_ell == tiny_model.config.max_ell
        for block in tiny_model.backbone.blocks
    )


def test_one_stable_architecture_name_and_no_weakening_switches() -> None:
    compact = JouleWeaveModelConfig.specialist()
    foundation = JouleWeaveModelConfig.universal()
    assert compact.architecture_name == foundation.architecture_name == "zynforge-zenith"
    assert not hasattr(compact, "use_linear_fast_path")
    assert not hasattr(compact, "topology_channels")
    assert compact.max_neighbors is None
    assert foundation.max_neighbors is None
    assert compact.num_experts == compact.expert_top_k == 1
    with pytest.raises(TypeError):
        JouleWeaveModelConfig(use_linear_fast_path=True)  # type: ignore[call-arg]


def test_merged_irrep_norm_uses_one_invariant_scale() -> None:
    torch.manual_seed(2027)
    layout = IrrepLayout(
        max_ell=2,
        include_pseudotensors=True,
        include_time_odd=True,
    )
    channels = 3
    fields = {
        key: torch.randn((4, channels, key.dimension), dtype=torch.float64)
        for key in layout.keys
    }
    norm = GradedIrrepNorm(layout, channels).double()
    output = norm(fields)
    square_sum = sum(value.square().sum(dim=(-1, -2)) for value in fields.values())
    component_count = sum(value.shape[-2] * value.shape[-1] for value in fields.values())
    expected_scale = torch.rsqrt(square_sum / component_count + norm.eps)[:, None, None]
    for key in layout.keys:
        torch.testing.assert_close(
            output[key],
            fields[key] * expected_scale,
            rtol=2.0e-15,
            atol=2.0e-15,
        )


def test_grade_aware_attention_starts_as_exact_identity_route() -> None:
    torch.manual_seed(2028)
    router = SmoothInvariantEdgeAttention(
        channels=8,
        attention_heads=2,
        hidden_rank=4,
        grade_count=7,
    ).double()
    state = torch.randn((11, 8), dtype=torch.float64)
    initial = router(state)
    assert initial.shape == (11, 7, 2)
    torch.testing.assert_close(initial, torch.ones_like(initial), rtol=0.0, atol=0.0)

    with torch.no_grad():
        router.grade_projection.weight.normal_(std=0.2)
    routed = router(state)
    assert torch.isfinite(routed).all()
    assert bool(torch.any((routed[:, 1:, :] - routed[:, :1, :]).abs() > 1.0e-8))
    assert bool(torch.all((routed > 0.0) & (routed < 2.0)))


def test_distilled_student_retains_safe_graph_and_valid_expert_route() -> None:
    teacher = JouleWeaveModelConfig.universal()
    student = build_distilled_student_config(
        teacher,
        width_multiplier=0.5,
        num_experts=1,
    )
    assert student.architecture_name == teacher.architecture_name == "zynforge-zenith"
    assert student.max_neighbors is None
    assert student.num_experts == student.expert_top_k == 1


def test_smooth_radial_basis_is_orthonormal_at_initialization() -> None:
    basis = OrthonormalSmoothRadialBasis(
        12,
        5.0,
        cutoff_order=4,
        trainable=False,
    ).double()
    nodes, weights = np.polynomial.legendre.leggauss(512)
    radius = torch.tensor(2.5 * (nodes + 1.0), dtype=torch.float64)
    radial_weights = torch.tensor(2.5 * weights, dtype=torch.float64)
    values = basis(radius)
    gram = values.T @ (radial_weights[:, None] * radius[:, None].square() * values)
    torch.testing.assert_close(
        gram,
        torch.eye(12, dtype=torch.float64),
        rtol=3.0e-10,
        atol=3.0e-10,
    )


def test_radial_value_and_first_four_derivatives_vanish_at_cutoff() -> None:
    basis = OrthonormalSmoothRadialBasis(8, 4.5, cutoff_order=4).double()
    radius = torch.tensor([4.5], dtype=torch.float64, requires_grad=True)
    derivative = basis(radius).sum()
    for _ in range(5):
        assert float(derivative.detach().abs().max()) < 2.0e-10
        derivative = torch.autograd.grad(
            derivative,
            radius,
            create_graph=True,
        )[0].sum()


@pytest.mark.parametrize("ell", [5, 6])
def test_high_angular_ceiling_is_equivariant_and_cartesian_exact(ell: int) -> None:
    generator = torch.Generator().manual_seed(161803 + ell)
    matrix, _ = torch.linalg.qr(
        torch.randn((3, 3), generator=generator, dtype=torch.float64)
    )
    if torch.det(matrix) < 0:
        matrix[:, 0] *= -1.0
    directions = torch.randn((20, 3), generator=generator, dtype=torch.float64)
    directions = directions / torch.linalg.vector_norm(
        directions,
        dim=-1,
        keepdim=True,
    )
    key = IrrepKey(ell, -1 if ell % 2 else 1, 1)
    representation = representation_matrix(matrix, key)
    reference = real_spherical_harmonics(directions, ell)
    transformed = real_spherical_harmonics(directions @ matrix.T, ell)
    expected = torch.einsum("ab,nb->na", representation, reference)
    torch.testing.assert_close(transformed, expected, rtol=2.0e-12, atol=2.0e-12)

    values = torch.randn((2, 3, 2 * ell + 1), generator=generator, dtype=torch.float64)
    recovered = cartesian_to_real_irrep(real_irrep_to_cartesian(values, ell), ell)
    torch.testing.assert_close(recovered, values, rtol=5.0e-12, atol=5.0e-12)


def test_sixth_order_cg_rows_are_orthonormal() -> None:
    coupling = real_clebsch_gordan(
        6,
        6,
        6,
        torch.empty((), dtype=torch.float64),
    )
    gram = torch.einsum("amn,bmn->ab", coupling, coupling)
    torch.testing.assert_close(
        gram,
        torch.eye(13, dtype=torch.float64),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
