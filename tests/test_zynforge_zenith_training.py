from __future__ import annotations

import pytest
import torch
from conftest import compact_test_config

from zynnova.ml.zynforge.field import (
    JouleWeaveTrainConfig,
    ZynForgeSymmetryPotential,
)
from zynnova.ml.zynforge.field.trainer import (
    ExponentialMovingAverage,
    _optimizer_parameter_groups,
)


@pytest.mark.parametrize("seed", [7, 123, 991])
def test_small_data_energy_force_fit_converges(seed: int) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(seed)
        config = compact_test_config(
            hidden_dim=8,
            num_layers=1,
            max_ell=1,
            correlation_order=2,
            tensor_product_rank=4,
            directional_edge_rank=4,
            num_attention_heads=1,
            num_experts=1,
            expert_top_k=1,
            use_layer_energy_mixing=False,
        )
        model = ZynForgeSymmetryPotential(config).float().train()

        distances = torch.linspace(0.72, 1.48, 8)
        positions = torch.zeros((16, 3))
        positions[1::2, 0] = distances
        inputs = {
            "z": torch.ones(16, dtype=torch.long),
            "pos": positions,
            "batch": torch.arange(8).repeat_interleave(2),
            "cell": torch.zeros((8, 3, 3)),
            "pbc": torch.zeros((8, 3), dtype=torch.bool),
        }
        stiffness = 1.7
        equilibrium = 1.05
        target_energy = 0.5 * stiffness * (distances - equilibrium).square()
        target_force = torch.zeros_like(positions)
        radial_force = stiffness * (distances - equilibrium)
        target_force[::2, 0] = radial_force
        target_force[1::2, 0] = -radial_force

        def objective(*, create_graph: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            output = model.energy_and_forces(inputs, create_graph=create_graph)
            loss = (output["energy"] - target_energy).square().mean()
            loss = loss + (output["forces"] - target_force).square().mean()
            return loss, output

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5.0e-3,
            weight_decay=0.0,
            amsgrad=True,
        )
        initial = float(objective(create_graph=False)[0])
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = objective(create_graph=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        final_loss, output = objective(create_graph=False)
        force_mae = float((output["forces"] - target_force).abs().mean())
        assert float(final_loss) < 0.006 * initial
        assert force_mae < 0.01
    finally:
        torch.set_num_threads(previous_threads)


def test_ema_warm_start_is_not_biased_to_initialization() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    ema = ExponentialMovingAverage(model, decay=0.999)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)
    shadow = ema.shadow["weight"]
    assert torch.all(shadow > 0.75)
    assert ema.updates == 1

    state = ema.state_dict()
    restored = ExponentialMovingAverage(model, decay=0.5)
    restored.load_state_dict(state, model=model)
    assert restored.decay == ema.decay
    assert restored.updates == 1
    torch.testing.assert_close(restored.shadow["weight"], shadow)


def test_optimizer_excludes_scales_biases_and_norms_from_weight_decay() -> None:
    model = ZynForgeSymmetryPotential(compact_test_config(num_layers=1))
    train_config = JouleWeaveTrainConfig(weight_decay=1.0e-3)
    groups = _optimizer_parameter_groups(model, train_config)
    decay_by_parameter = {
        id(parameter): float(group["weight_decay"])
        for group in groups
        for parameter in group["params"]
    }
    assert train_config.amsgrad
    assert {group["name"] for group in groups} <= {
        "backbone",
        "embedding",
        "radial",
        "readout",
    }
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        no_decay_tokens = ("bias", "norm", "scale", "logit", "spacing", "width")
        if any(token in lowered for token in no_decay_tokens):
            assert decay_by_parameter[id(parameter)] == 0.0, name
