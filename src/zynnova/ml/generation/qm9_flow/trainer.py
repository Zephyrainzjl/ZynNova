from __future__ import annotations

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
from .config import QM9FlowConfig
from .data import center_coordinates, prepare_qm9_flow_data
from .model import QM9EquivariantFlow


def _flow_loss(model, batch, *, train: bool):
    import torch

    z = batch["z"]
    target_positions = center_coordinates(batch["positions"], batch["mask"])
    mask = batch["mask"]
    noise = center_coordinates(torch.randn_like(target_positions), mask)
    time = torch.rand(z.shape[0], device=z.device, dtype=target_positions.dtype)
    interpolation = (
        (1.0 - time[:, None, None]) * noise
        + time[:, None, None] * target_positions
    )
    target_velocity = target_positions - noise
    predicted_velocity = model(z, interpolation, time, mask)
    squared = (predicted_velocity - target_velocity).square().sum(dim=-1)
    loss = (squared * mask.to(squared.dtype)).sum() / mask.sum().clamp_min(1)
    mae = (
        torch.linalg.vector_norm(predicted_velocity - target_velocity, dim=-1)
        * mask.to(squared.dtype)
    ).sum() / mask.sum().clamp_min(1)
    return loss, {"loss": float(loss.detach().cpu()), "velocity_mae_A": float(mae.detach().cpu())}


def _run_epoch(model, loader: Any, *, optimizer: Any | None, device: Any, dtype: Any, clip):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_mae = 0.0
    count = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        batch["positions"] = batch["positions"].to(dtype=dtype)
        if train:
            optimizer.zero_grad(set_to_none=True)
        loss, metrics = _flow_loss(model, batch, train=train)
        if train:
            loss.backward()
            if clip is not None:
                import torch

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        total_loss += metrics["loss"]
        total_mae += metrics["velocity_mae_A"]
        count += 1
    if not count:
        raise ValueError("empty data loader")
    return {"loss": total_loss / count, "velocity_mae_A": total_mae / count}


def train_qm9_flow(config: QM9FlowConfig | None = None) -> TrainingResult:
    import torch

    config = config or QM9FlowConfig()
    seed_everything(config.train.seed, deterministic=config.train.deterministic)
    workspace = MLWorkspace(config.train.workspace_root)
    run = workspace.create_run("generation", "qm9-flow", name=config.train.run_name, config=config)
    data = prepare_qm9_flow_data(config.data, config.model, workspace=workspace)
    model = QM9EquivariantFlow(config.model)
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
            clip=config.train.gradient_clip_norm,
        )
        with torch.no_grad():
            valid_metrics = _run_epoch(
                model,
                data.val_dataloader(),
                optimizer=None,
                device=device,
                dtype=dtype,
                clip=None,
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
            "model_name": "qm9_flow",
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
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
    model.eval()
    return TrainingResult(
        run_dir=run.root,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        history=history,
        best_metric=best_metric,
        model=model,
    )


__all__ = ["train_qm9_flow"]
