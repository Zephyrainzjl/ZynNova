from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from zynnova.ml.zivar.config import SpinConfig
from zynnova.ml.zivar.spin_dynamics import llg_midpoint_step, llg_rhs


def test_zero_damping_precession_and_midpoint_preserve_magnitude() -> None:
    config = SpinConfig(hidden=(8,), llg_damping=0.0)
    spins = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    field = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    rhs = llg_rhs(
        spins, field,
        gyromagnetic_ratio=config.llg_gyromagnetic_ratio,
        damping=0.0,
    )
    assert torch.allclose((rhs * spins).sum(-1), torch.zeros(1, dtype=torch.float64))
    stepped = llg_midpoint_step(
        spins, lambda _: field, 1.0e-15, config, iterations=12, damping=0.0
    )
    assert torch.allclose(
        torch.linalg.vector_norm(stepped, dim=-1),
        torch.linalg.vector_norm(spins, dim=-1),
        atol=2.0e-12,
    )


def test_midpoint_solver_fails_closed_when_not_converged() -> None:
    config = SpinConfig(hidden=(8,), llg_damping=0.0)
    spins = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    field = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="did not converge"):
        llg_midpoint_step(
            spins, lambda _: field, 1.0e-12, config,
            iterations=1, damping=0.0, rtol=1.0e-14, atol=1.0e-16,
        )
