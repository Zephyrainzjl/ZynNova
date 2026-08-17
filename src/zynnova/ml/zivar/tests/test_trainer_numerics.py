from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from zynnova.ml.zivar.trainer import assert_model_optimizer_finite


def test_optimizer_finite_check_handles_cpu_step_and_parameter_tensors() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    assert_model_optimizer_finite(model, optimizer)
    next(iter(optimizer.state.values()))["step"].fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        assert_model_optimizer_finite(model, optimizer)
