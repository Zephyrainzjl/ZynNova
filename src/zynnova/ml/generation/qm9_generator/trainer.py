from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from ...common import (
    TrainingResult,
    append_jsonl,
    move_to_device,
    resolve_device,
    save_checkpoint,
    seed_everything,
)
from ...workspace import MLWorkspace
from .config import QM9GeneratorConfig, QM9_PROPERTY_UNITS
from .data import center_coordinates, prepare_qm9_generator_data
from .model import QM9ConditionalGenerator


def _masked_mean(values, mask):
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _losses(model, batch, *, condition_dropout: float, train: bool):
    import torch

    z = batch["z"]
    mask = batch["mask"]
    target_positions = center_coordinates(batch["positions"], mask)
    properties = batch["properties"]
    property_mask = batch["property_mask"]

    noise = center_coordinates(torch.randn_like(target_positions), mask)
    time = torch.rand(z.shape[0], device=z.device, dtype=target_positions.dtype)
    interpolation = (
        (1.0 - time[:, None, None]) * noise
        + time[:, None, None] * target_positions
    )
    target_velocity = target_positions - noise

    condition_mask = property_mask.clone()
    if train and condition_dropout > 0:
        dropped = torch.rand(z.shape[0], device=z.device) < condition_dropout
        condition_mask[dropped] = False
    predicted_velocity = model(
        z,
        interpolation,
        time,
        mask,
        properties,
        condition_mask,
    )
    velocity_error = (predicted_velocity - target_velocity).square().sum(dim=-1)
    flow_loss = _masked_mean(velocity_error, mask)
    velocity_mae = _masked_mean(
        torch.linalg.vector_norm(predicted_velocity - target_velocity, dim=-1),
        mask,
    )

    property_prediction = model.predict_properties(z, target_positions, mask)
    property_squared = (property_prediction - properties).square()
    property_loss = _masked_mean(property_squared, property_mask)
    property_mae = _masked_mean(
        (property_prediction - properties).abs(),
        property_mask,
    )
    per_property: dict[str, float] = {}
    for index, name in enumerate(model.config.property_names):
        observed = property_mask[:, index]
        if observed.any():
            value = (
                (property_prediction[:, index] - properties[:, index]).abs()[observed]
                .mean()
                .detach()
                .cpu()
            )
            per_property[f"property_mae_z_{name}"] = float(value)
    return flow_loss, property_loss, {
        "flow_loss": float(flow_loss.detach().cpu()),
        "property_loss": float(property_loss.detach().cpu()),
        "velocity_mae_A": float(velocity_mae.detach().cpu()),
        "property_mae_z": float(property_mae.detach().cpu()),
        **per_property,
    }


def _run_epoch(
    model,
    loader: Any,
    *,
    optimizer: Any | None,
    device: Any,
    dtype: Any,
    config: QM9GeneratorConfig,
):
    import torch

    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        for key in ("positions", "properties", "raw_properties"):
            batch[key] = batch[key].to(dtype=dtype)
        if train:
            optimizer.zero_grad(set_to_none=True)
        flow_loss, property_loss, metrics = _losses(
            model,
            batch,
            condition_dropout=config.train.condition_dropout,
            train=train,
        )
        total_loss = (
            config.train.flow_loss_weight * flow_loss
            + config.train.property_loss_weight * property_loss
        )
        if train:
            total_loss.backward()
            if config.train.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.train.gradient_clip_norm,
                )
            optimizer.step()
        metrics["loss"] = float(total_loss.detach().cpu())
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    if not count:
        raise ValueError("empty data loader")
    return {key: value / count for key, value in totals.items()}


def train_qm9_generator(
    config: QM9GeneratorConfig | None = None,
) -> TrainingResult:
    import torch

    config = config or QM9GeneratorConfig()
    config.model.__post_init__()
    config.data.__post_init__()
    config.train.__post_init__()
    seed_everything(config.train.seed, deterministic=config.train.deterministic)
    workspace = MLWorkspace(config.train.workspace_root)
    run = workspace.create_run(
        "generation",
        "qm9-generator",
        name=config.train.run_name,
        config=config,
    )
    data = prepare_qm9_generator_data(
        config.data,
        config.model,
        workspace=workspace,
    )
    model = QM9ConditionalGenerator(config.model)
    model.property_normalizer = data.normalizer
    device = resolve_device(config.train.device)
    dtype = getattr(torch, config.train.dtype)
    model.to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(config.train.patience // 4, 5),
    )
    best_metric = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    best_path = run.checkpoints / "best.pt"
    last_path = run.checkpoints / "last.pt"

    for epoch in range(1, config.train.epochs + 1):
        train_metrics = _run_epoch(
            model,
            data.train_dataloader(),
            optimizer=optimizer,
            device=device,
            dtype=dtype,
            config=config,
        )
        with torch.no_grad():
            valid_metrics = _run_epoch(
                model,
                data.val_dataloader(),
                optimizer=None,
                device=device,
                dtype=dtype,
                config=config,
            )
        scheduler.step(valid_metrics["loss"])
        row = {
            "epoch": float(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
        }
        history.append(row)
        append_jsonl(run.history, row)
        payload = {
            "model_name": "qm9_generator",
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "property_normalizer": data.normalizer.state_dict(),
            "property_units": {
                name: QM9_PROPERTY_UNITS[name]
                for name in model.config.property_names
            },
            "split_sizes": data.split_sizes,
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": row,
        }
        save_checkpoint(last_path, payload)
        if valid_metrics["loss"] < best_metric - config.train.min_delta:
            best_metric = valid_metrics["loss"]
            stale = 0
            save_checkpoint(best_path, payload)
        else:
            stale += 1
            if stale >= config.train.patience:
                break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    model.property_normalizer = data.normalizer
    model.eval()

    test_metrics: dict[str, float] | None = None
    if data.split_sizes["test"]:
        with torch.no_grad():
            test_metrics = _run_epoch(
                model,
                data.test_dataloader(),
                optimizer=None,
                device=device,
                dtype=dtype,
                config=config,
            )
    summary = {
        "best_validation_loss": best_metric,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "split_sizes": data.split_sizes,
        "property_names": list(model.config.property_names),
        "property_units": {
            name: QM9_PROPERTY_UNITS[name]
            for name in model.config.property_names
        },
        "property_normalizer": data.normalizer.state_dict(),
        "test_metrics": test_metrics,
    }
    (run.logs / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return TrainingResult(
        run_dir=run.root,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        history=history,
        best_metric=best_metric,
        model=model,
    )


__all__ = ["train_qm9_generator"]
