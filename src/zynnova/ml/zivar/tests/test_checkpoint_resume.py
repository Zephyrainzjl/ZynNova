from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from zynnova.ml.zivar.checkpoint import (
    load_zivar,
    restore_training_state,
    save_zivar,
)
from zynnova.ml.zivar.config import ZIVARConfig
from zynnova.ml.zivar.model import build_zivar
from zynnova.ml.zivar.trainer import TrainerConfig, ZIVARTrainer


def _problem() -> tuple[ZIVARConfig, dict[str, object]]:
    config = ZIVARConfig.convolution(
        dft_level="resume-test",
        backbone__atomic_numbers=(3, 8),
        backbone__channels=8,
        backbone__num_interactions=1,
        backbone__num_bessel=2,
        backbone__radial_mlp=(8,),
        electronic__hidden=(8,),
        spin__mode="disabled",
        spin__require_spin_input=False,
    )
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.7, 0.1, 0.0]], dtype=torch.float64)
    data = {
        "positions": positions,
        "atomic_numbers": torch.tensor([3, 8]),
        "node_attrs": torch.eye(2, dtype=torch.float64),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "shifts": torch.zeros(2, 3, dtype=torch.float64),
        "cell": torch.eye(3, dtype=torch.float64).mul(10.0).unsqueeze(0),
        "pbc": torch.zeros(1, 3, dtype=torch.bool),
        "batch": torch.zeros(2, dtype=torch.long),
    }
    return config, data


def _trainer(model: object) -> ZIVARTrainer:
    return ZIVARTrainer(
        model,
        config=TrainerConfig(
            learning_rate=1.0e-4,
            force_training=False,
            spin_field_training=False,
        ),
    )


def test_interrupted_resume_reproduces_the_next_cpu_step(tmp_path: Path) -> None:
    torch.manual_seed(1234)
    config, data = _problem()
    model = build_zivar(config).double()
    trainer = _trainer(model)
    batch = {
        "data": data,
        "conditions": {"total_charge": torch.tensor([0.0], dtype=torch.float64)},
        "targets": {"energy": torch.tensor([0.25], dtype=torch.float64)},
    }
    trainer.train_step(batch)
    checkpoint = save_zivar(model, tmp_path / "resume.pt", trainer=trainer)
    trainer.train_step(batch)
    expected = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    restored_model = load_zivar(checkpoint).double()
    restored_trainer = _trainer(restored_model)
    restore_training_state(restored_trainer, checkpoint)
    restored_trainer.train_step(batch)
    for name, value in restored_model.state_dict().items():
        assert torch.equal(value, expected[name]), name


def test_resume_rejects_numerics_model_and_trainer_mismatches(tmp_path: Path) -> None:
    torch.manual_seed(1234)
    config, data = _problem()
    model = build_zivar(config).double()
    trainer = _trainer(model)
    batch = {
        "data": data,
        "conditions": {"total_charge": torch.tensor([0.0], dtype=torch.float64)},
        "targets": {"energy": torch.tensor([0.25], dtype=torch.float64)},
    }
    trainer.train_step(batch)
    checkpoint = save_zivar(model, tmp_path / "resume.pt", trainer=trainer)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["numerics_revision"] = "unknown-numerics"
    forged = tmp_path / "unknown-numerics.pt"
    torch.save(payload, forged)
    with pytest.raises(ValueError, match="numerics"):
        restore_training_state(_trainer(load_zivar(checkpoint).double()), forged)

    different_config = replace(
        config,
        electrostatics=replace(config.electrostatics, method="direct_ewald"),
    )
    with pytest.raises(ValueError, match="model configuration"):
        restore_training_state(
            _trainer(build_zivar(different_config).double()), checkpoint
        )

    restored = load_zivar(checkpoint).double()
    different_trainer = ZIVARTrainer(
        restored,
        config=TrainerConfig(
            learning_rate=2.0e-4,
            force_training=False,
            spin_field_training=False,
        ),
    )
    with pytest.raises(ValueError, match="trainer configuration"):
        restore_training_state(different_trainer, checkpoint)

    reordered_model = load_zivar(checkpoint).double()
    reordered_optimizer = torch.optim.AdamW(
        reversed(tuple(reordered_model.parameters())),
        lr=1.0e-4,
        weight_decay=1.0e-8,
    )
    reordered_trainer = ZIVARTrainer(
        reordered_model,
        config=TrainerConfig(
            learning_rate=1.0e-4,
            force_training=False,
            spin_field_training=False,
        ),
        optimizer=reordered_optimizer,
    )
    with pytest.raises(ValueError, match="groups/order"):
        restore_training_state(reordered_trainer, checkpoint)

    altered = load_zivar(checkpoint).double()
    with torch.no_grad():
        next(altered.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="model weights"):
        restore_training_state(_trainer(altered), checkpoint)


def test_save_rejects_cross_wired_trainer_optimizer_and_epoch(tmp_path: Path) -> None:
    config, _ = _problem()
    first = build_zivar(config).double()
    second = build_zivar(config).double()
    trainer = _trainer(second)
    with pytest.raises(ValueError, match="different model instance"):
        save_zivar(first, tmp_path / "cross-wired.pt", trainer=trainer)

    correct = _trainer(first)
    unrelated_optimizer = torch.optim.AdamW(first.parameters(), lr=3.0e-4)
    with pytest.raises(ValueError, match="optimizer differs"):
        save_zivar(
            first,
            tmp_path / "optimizer.pt",
            trainer=correct,
            optimizer=unrelated_optimizer,
        )

    correct.epoch = 3
    with pytest.raises(ValueError, match="epoch differs"):
        save_zivar(first, tmp_path / "epoch.pt", trainer=correct, epoch=2)

    first_optimizer = torch.optim.AdamW(first.parameters(), lr=1.0e-4)
    second_optimizer = torch.optim.AdamW(first.parameters(), lr=1.0e-4)
    crossed_scheduler = torch.optim.lr_scheduler.StepLR(
        second_optimizer, step_size=1
    )
    with pytest.raises(ValueError, match="different optimizer instance"):
        ZIVARTrainer(
            first,
            config=TrainerConfig(
                learning_rate=1.0e-4,
                force_training=False,
                spin_field_training=False,
            ),
            optimizer=first_optimizer,
            scheduler=crossed_scheduler,
        )

    lambda_scheduler = torch.optim.lr_scheduler.LambdaLR(
        first_optimizer, lr_lambda=lambda epoch: 0.95**epoch
    )
    with pytest.raises(ValueError, match="not certified for exact resume"):
        ZIVARTrainer(
            first,
            config=TrainerConfig(
                learning_rate=1.0e-4,
                force_training=False,
                spin_field_training=False,
            ),
            optimizer=first_optimizer,
            scheduler=lambda_scheduler,
        )


def test_supported_scheduler_continues_exactly_after_resume(tmp_path: Path) -> None:
    torch.manual_seed(902)
    config, data = _problem()
    model = build_zivar(config).double()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-4, weight_decay=1.0e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = ZIVARTrainer(
        model,
        config=TrainerConfig(
            learning_rate=1.0e-4,
            force_training=False,
            spin_field_training=False,
        ),
        optimizer=optimizer,
        scheduler=scheduler,
    )
    batch = {
        "data": data,
        "conditions": {"total_charge": torch.tensor([0.0], dtype=torch.float64)},
        "targets": {"energy": torch.tensor([0.25], dtype=torch.float64)},
    }
    trainer.train_step(batch)
    scheduler.step()
    checkpoint = save_zivar(model, tmp_path / "scheduler.pt", trainer=trainer)
    scheduler.step()
    expected_lr = optimizer.param_groups[0]["lr"]

    restored_model = load_zivar(checkpoint).double()
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=1.0e-4, weight_decay=1.0e-8
    )
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=1, gamma=0.5
    )
    restored_trainer = ZIVARTrainer(
        restored_model,
        config=trainer.config,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )
    restore_training_state(restored_trainer, checkpoint)
    restored_scheduler.step()
    assert restored_optimizer.param_groups[0]["lr"] == expected_lr
