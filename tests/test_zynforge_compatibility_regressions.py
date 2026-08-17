from __future__ import annotations

import math
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import torch

from zynnova.ml.zynforge.field import (
    EquivariantDistillationConfig,
    ZynFieldConfig,
    ZynFieldDataConfig,
    ZynFieldModelConfig,
    ZynFieldPotential,
    ZynFieldTrainConfig,
    build_distilled_student_config,
    distill_zynfield,
    resume_zynfield,
    rotate_irrep,
    train_zynfield,
)
from zynnova.ml.zynforge.data.matpes import build_smoke_samples
from zynnova.ml.zynforge.field import trainer as trainer_module
from zynnova.ml.zynforge.field.config import (
    ZYNFORGE_CHECKPOINT_MODEL_NAME,
    jouleweave_model_config_from_dict,
)


def _config(**overrides):
    values = {
        "hidden_dim": 16,
        "num_layers": 2,
        "num_radial": 8,
        "num_attention_heads": 4,
        "num_experts": 2,
        "expert_top_k": 2,
        "max_ell": 3,
        "correlation_order": 3,
        "tensor_product_rank": 4,
        "directional_edge_rank": 4,
        "interaction_cutoff_A": 4.0,
        "max_neighbors": None,
        "use_zbl": False,
        "use_dispersion": False,
        "use_qeq": False,
    }
    values.update(overrides)
    return ZynFieldModelConfig.specialist(**values)


def _inputs(dtype=torch.float64):
    return {
        "z": torch.tensor([1, 8, 6, 3], dtype=torch.long),
        "pos": torch.tensor(
            [
                [0.2, 0.3, 0.4],
                [1.3, 0.4, 0.5],
                [0.5, 1.6, 0.7],
                [1.1, 1.2, 1.5],
            ],
            dtype=dtype,
        ),
        "batch": torch.zeros(4, dtype=torch.long),
        "cell": 6.0 * torch.eye(3, dtype=dtype).unsqueeze(0),
        "pbc": torch.zeros((1, 3), dtype=torch.bool),
    }


def test_public_force_paths_keep_auxiliary_equivariant_fields() -> None:
    model = ZynFieldPotential(_config()).double().eval()
    inputs = _inputs()
    output = model.energy_and_forces(inputs)
    stressed = model.energy_forces_stress(_inputs(), compute_stress=False)
    for result in (output, stressed):
        assert result["node_scalar"].shape == (4, 16)
        assert result["node_vector"].shape == (4, 16, 3)
        assert result["node_irrep_l3"].shape[-1] == 7

    lean_inputs = dict(inputs)
    lean_inputs["_return_auxiliary_fields"] = False
    lean = model.energy_and_forces(lean_inputs)
    assert "node_scalar" not in lean
    assert "node_vector" not in lean
    torch.testing.assert_close(lean["energy"], output["energy"])
    torch.testing.assert_close(lean["forces"], output["forces"])

    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    rotated_inputs = dict(inputs)
    rotated_inputs["pos"] = inputs["pos"] @ rotation.T
    rotated = model.energy_and_forces(rotated_inputs)
    torch.testing.assert_close(
        rotated["node_vector"],
        rotate_irrep(output["node_vector"], rotation, 1),
        atol=5.0e-10,
        rtol=5.0e-10,
    )


def test_force_training_opt_out_does_not_remove_distillation_fields(tmp_path) -> None:
    teacher_config = _config()
    student_config = build_distilled_student_config(
        teacher_config,
        width_multiplier=0.5,
        correlation_order=2,
        num_layers=2,
    )
    result = distill_zynfield(
        ZynFieldPotential(teacher_config),
        ZynFieldPotential(student_config),
        [_inputs(torch.float32)],
        EquivariantDistillationConfig(
            epochs=1,
            force_weight=1.0,
            stress_weight=0.0,
            relation_weight=0.01,
            irrep_power_weight=0.01,
            checkpoint_dir=tmp_path,
        ),
    )
    assert result.best_checkpoint.is_file()
    assert math.isfinite(result.best_loss)


def test_historical_identifier_is_metadata_only_migration() -> None:
    payload = asdict(_config())
    payload.pop("architecture_name")
    payload["architecture_version"] = "zynforge-symmetry-legacy"
    restored = jouleweave_model_config_from_dict(payload)
    assert restored.architecture_name == "zynforge-zenith"


def test_epoch_aggregation_accepts_minimal_validation_config() -> None:
    statistics = [
        {
            "loss": 1.0,
            "_energy_abs_sum": 1.0,
            "_energy_sq_sum": 1.0,
            "_energy_count": 1.0,
            "_force_abs_sum": 3.0,
            "_force_sq_sum": 5.0,
            "_force_count": 2.0,
            "_charge_abs_sum": 0.0,
            "_charge_sq_sum": 0.0,
            "_charge_count": 0.0,
            "_magmom_abs_sum": 0.0,
            "_magmom_sq_sum": 0.0,
            "_magmom_count": 0.0,
            "_oxidation_correct_sum": 0.0,
            "_oxidation_abs_sum": 0.0,
            "_oxidation_count": 0.0,
        },
        {
            "loss": 3.0,
            "_energy_abs_sum": 3.0,
            "_energy_sq_sum": 9.0,
            "_energy_count": 1.0,
            "_force_abs_sum": 10.0,
            "_force_sq_sum": 50.0,
            "_force_count": 4.0,
            "_charge_abs_sum": 0.0,
            "_charge_sq_sum": 0.0,
            "_charge_count": 0.0,
            "_magmom_abs_sum": 0.0,
            "_magmom_sq_sum": 0.0,
            "_magmom_count": 0.0,
            "_oxidation_correct_sum": 0.0,
            "_oxidation_abs_sum": 0.0,
            "_oxidation_count": 0.0,
        },
    ]

    class DummyModel:
        def train(self, _mode):
            return self

    def fake_batch_loss(_model, batch, **_kwargs):
        return torch.tensor(0.0), statistics[batch["index"]]

    config = SimpleNamespace(gradient_accumulation=1, progress_bar=False)
    with patch.object(trainer_module, "_batch_loss", fake_batch_loss):
        metrics = trainer_module._run_epoch(
            DummyModel(),
            [{"index": 0}, {"index": 1}],
            optimizer=None,
            ema=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
            config=config,
        )
    assert metrics["energy_mae_eV_per_atom"] == 2.0
    assert metrics["force_mae_eV_per_A"] == 13.0 / 6.0
    assert metrics["force_rmse_eV_per_A"] == math.sqrt(55.0 / 6.0)


def test_ema_flat_tensor_state_is_metadata_preserving_and_alignable() -> None:
    model = torch.nn.Linear(3, 2).float()
    ema = trainer_module.ExponentialMovingAverage(model, decay=0.99)
    ema.updates = 7
    direct = trainer_module.ExponentialMovingAverage(model, decay=0.5)
    direct.load_state_dict(ema.state_dict(), model=model)
    assert direct.decay == 0.99
    assert direct.updates == 7

    checkpoint_state = {
        name: value.detach().cpu().to(torch.float64)
        for name, value in ema.state_dict().items()
    }
    assert set(checkpoint_state) == set(model.state_dict())
    restored = trainer_module.ExponentialMovingAverage(model, decay=0.99)
    restored.load_state_dict(checkpoint_state, model=model)
    assert restored.decay == 0.99
    assert restored.updates == 0
    assert all(value.dtype == torch.float32 for value in restored.shadow.values())
    assert ZYNFORGE_CHECKPOINT_MODEL_NAME == "zynforge-field"


def test_checkpoint_name_metric_schema_and_resume_are_compatible(tmp_path) -> None:
    config = ZynFieldConfig(
        model=_config(
            hidden_dim=8,
            num_layers=1,
            num_attention_heads=2,
            num_experts=1,
            expert_top_k=1,
            max_ell=1,
            correlation_order=1,
            tensor_product_rank=2,
            directional_edge_rank=2,
        ),
        data=ZynFieldDataConfig(
            dataset="compatibility-smoke",
            energy_source="labels.energy",
            forces_source="labels.forces",
            stress_source=None,
            material_types=("crystal",),
            batch_size=3,
            num_workers=0,
            seed=42,
        ),
        train=ZynFieldTrainConfig(
            epochs=1,
            learning_rate=5.0e-4,
            warmup_epochs=0,
            energy_weight=1.0,
            force_weight=1.0,
            stress_weight=0.0,
            reference_fit="none",
            gradient_accumulation=1,
            ema_decay=0.99,
            patience=2,
            device="cpu",
            dtype="float32",
            workspace_root=tmp_path / "workspace",
            run_name="compatibility-resume",
            progress_bar=False,
        ),
    )
    samples = build_smoke_samples(12, seed=42)
    first = train_zynfield(config, source=samples)
    first_payload = torch.load(
        first.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert first_payload["model_name"] == "zynforge-field"
    assert first_payload["metric_aggregation"] == "global-element-weighted-v1"
    assert set(first_payload["ema_state"]) == {"decay", "updates", "shadow"}

    resumed = resume_zynfield(
        config,
        checkpoint=first.last_checkpoint,
        source=samples,
        total_epochs=2,
        in_place=True,
    )
    second_payload = torch.load(
        resumed.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert second_payload["epoch"] == 2
    assert second_payload["metric_aggregation"] == "global-element-weighted-v1"
