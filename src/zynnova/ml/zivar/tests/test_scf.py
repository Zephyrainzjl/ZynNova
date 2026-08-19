from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from zynnova.ml.zivar.implicit import implicit_quadratic_solve
from zynnova.ml.zivar.operators import (
    BlockDiagonalPreconditioner,
    CallableLinearOperator,
    DenseLinearOperator,
    DiagonalPreconditioner,
)
from zynnova.ml.zivar.scf import (
    SCFConvergenceError,
    SCFSolverConfig,
    solve_quadratic_scf,
)


def _dense_kkt_solution(
    hessian: torch.Tensor,
    linear: torch.Tensor,
    constraint: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = hessian.shape[0]
    constrained = constraint.shape[0]
    kkt = torch.cat(
        (
            torch.cat((hessian, constraint.T), dim=1),
            torch.cat(
                (
                    constraint,
                    hessian.new_zeros((constrained, constrained)),
                ),
                dim=1,
            ),
        ),
        dim=0,
    )
    reference = torch.linalg.solve(kkt, torch.cat((-linear, target)))
    return reference[:count], reference[count:]


def test_matrix_free_block_pcg_matches_dense_kkt_oracle() -> None:
    generator = torch.Generator().manual_seed(431)
    count = 12
    raw = torch.randn((count, count), generator=generator, dtype=torch.float64)
    hessian = raw @ raw.T / count + 1.5 * torch.eye(count, dtype=torch.float64)
    linear = torch.randn(count, generator=generator, dtype=torch.float64)
    constraint = torch.stack(
        (
            torch.ones(count, dtype=torch.float64),
            torch.cat(
                (
                    torch.ones(count // 2, dtype=torch.float64),
                    torch.zeros(count // 2, dtype=torch.float64),
                )
            ),
        )
    )
    target = torch.tensor([1.0, 0.35], dtype=torch.float64)
    operator = CallableLinearOperator(
        count,
        lambda value: hessian @ value,
        dtype=hessian.dtype,
        device=hessian.device,
    )
    blocks = tuple(hessian[index : index + 3, index : index + 3] for index in range(0, count, 3))
    result = solve_quadratic_scf(
        operator,
        linear,
        constraint=constraint,
        target=target,
        preconditioner=BlockDiagonalPreconditioner(blocks),
        config=SCFSolverConfig(
            atol=1.0e-12,
            rtol=1.0e-11,
            constraint_atol=1.0e-12,
            max_iter=128,
            recompute_interval=16,
        ),
    )
    expected, expected_multiplier = _dense_kkt_solution(
        hessian, linear, constraint, target
    )
    assert result.report.converged
    assert result.report.termination == "converged"
    assert result.report.final_residual <= 1.0e-10
    assert result.report.constraint_residual <= 1.0e-12
    assert torch.allclose(result.solution, expected, atol=2.0e-11, rtol=2.0e-11)
    assert torch.allclose(
        result.lagrange_multipliers,
        expected_multiplier,
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    assert torch.allclose(constraint @ result.solution, target, atol=1.0e-12, rtol=0.0)


def test_sparse_total_charge_constraint_and_warm_start() -> None:
    diagonal = torch.tensor([1.1, 1.7, 2.2, 3.0, 4.1], dtype=torch.float64)
    hessian = torch.diag(diagonal)
    linear = torch.tensor([0.4, -0.2, 0.1, -0.5, 0.3], dtype=torch.float64)
    indices = torch.stack(
        (
            torch.zeros(5, dtype=torch.long),
            torch.arange(5, dtype=torch.long),
        )
    )
    constraint = torch.sparse_coo_tensor(
        indices,
        torch.ones(5, dtype=torch.float64),
        size=(1, 5),
        check_invariants=True,
    ).coalesce()
    target = torch.tensor([-1.25], dtype=torch.float64)
    settings = SCFSolverConfig(
        atol=1.0e-12,
        rtol=1.0e-11,
        constraint_atol=1.0e-12,
        max_iter=32,
    )
    first = solve_quadratic_scf(
        DenseLinearOperator(hessian),
        linear,
        constraint=constraint,
        target=target,
        preconditioner=DiagonalPreconditioner(diagonal),
        config=settings,
    )
    restarted = solve_quadratic_scf(
        DenseLinearOperator(hessian),
        linear,
        constraint=constraint,
        target=target,
        preconditioner=DiagonalPreconditioner(diagonal),
        warm_start=first.solution,
        config=settings,
    )
    assert torch.allclose(first.solution.sum(), target[0], atol=1.0e-12, rtol=0.0)
    assert restarted.report.iterations == 0
    assert torch.equal(restarted.solution, first.solution)


def test_energy_stationarity_tolerance_prevents_premature_residual_acceptance() -> None:
    operator = DenseLinearOperator(torch.ones((1, 1), dtype=torch.float64))
    result = solve_quadratic_scf(
        operator,
        torch.ones(1, dtype=torch.float64),
        config=SCFSolverConfig(
            atol=2.0,
            rtol=0.0,
            energy_atol=1.0e-12,
            max_iter=2,
        ),
    )
    assert result.report.iterations == 1
    assert result.report.energy_error <= 1.0e-12
    assert torch.equal(result.solution, torch.tensor([-1.0], dtype=torch.float64))


def test_solver_is_differentiable_and_stationary_energy_obeys_envelope() -> None:
    hessian = torch.tensor(
        [[3.0, 0.4, -0.1], [0.4, 2.2, 0.2], [-0.1, 0.2, 1.7]],
        dtype=torch.float64,
    )
    linear = torch.tensor(
        [0.2, -0.4, 0.1], dtype=torch.float64, requires_grad=True
    )
    result = solve_quadratic_scf(
        DenseLinearOperator(hessian),
        linear,
        preconditioner=DiagonalPreconditioner(hessian.diag()),
        config=SCFSolverConfig(atol=1.0e-13, rtol=1.0e-12, max_iter=16),
    )
    gradient = torch.autograd.grad(result.energy, linear)[0]
    assert torch.allclose(gradient, result.solution.detach(), atol=2.0e-11, rtol=2.0e-11)


def test_implicit_kkt_backward_passes_gradcheck() -> None:
    hessian = torch.tensor(
        [[4.0, 0.3, -0.1], [0.3, 2.5, 0.2], [-0.1, 0.2, 1.8]],
        dtype=torch.float64,
        requires_grad=True,
    )
    linear = torch.tensor(
        [0.2, -0.3, 0.1], dtype=torch.float64, requires_grad=True
    )
    constraint = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.tensor([0.7, -0.2], dtype=torch.float64, requires_grad=True)
    settings = SCFSolverConfig(
        atol=1.0e-13,
        rtol=1.0e-12,
        constraint_atol=1.0e-12,
        max_iter=32,
        recompute_interval=8,
    )

    def solve(
        local_hessian: torch.Tensor,
        local_linear: torch.Tensor,
        local_constraint: torch.Tensor,
        local_target: torch.Tensor,
    ) -> torch.Tensor:
        return implicit_quadratic_solve(
            local_hessian,
            local_linear,
            constraint=local_constraint,
            target=local_target,
            config=settings,
        )

    assert torch.autograd.gradcheck(
        solve,
        (hessian, linear, constraint, target),
        eps=1.0e-6,
        atol=2.0e-5,
        rtol=2.0e-4,
    )


def test_nonconvergence_and_negative_curvature_fail_closed() -> None:
    hessian = torch.diag(torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64))
    linear = torch.ones(4, dtype=torch.float64)
    with pytest.raises(SCFConvergenceError, match="maximum iterations") as failure:
        solve_quadratic_scf(
            DenseLinearOperator(hessian),
            linear,
            config=SCFSolverConfig(atol=1.0e-14, rtol=0.0, max_iter=1),
        )
    assert not failure.value.report.converged
    assert failure.value.report.iterations == 1

    indefinite = torch.diag(torch.tensor([-1.0, 2.0], dtype=torch.float64))
    with pytest.raises(SCFConvergenceError, match="negative curvature") as curvature:
        solve_quadratic_scf(
            DenseLinearOperator(indefinite),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            config=SCFSolverConfig(max_iter=4),
        )
    assert curvature.value.report.termination == "negative curvature"


def test_singular_constraint_is_rejected_instead_of_pseudoinverted() -> None:
    hessian = torch.eye(3, dtype=torch.float64)
    constraint = torch.tensor(
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float64
    )
    with pytest.raises(ValueError, match="linearly independent"):
        solve_quadratic_scf(
            DenseLinearOperator(hessian),
            torch.zeros(3, dtype=torch.float64),
            constraint=constraint,
            target=torch.tensor([1.0, 2.0], dtype=torch.float64),
        )
