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
from .config import CrystalGNNConfig, CrystalGNNModelConfig
from .data import fit_target_normalization, prepare_matbench_data
from .model import CrystalGNN


def _run_epoch(model, loader: Any, *, optimizer: Any | None, device: Any, dtype: Any, clip):
    import torch

    train = optimizer is not None
    model.train(train)
    totals = {"loss": 0.0, "mae_eV_per_atom": 0.0, "rmse_eV_per_atom": 0.0}
    count = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        batch["edge_distance"] = batch["edge_distance"].to(dtype=dtype)
        batch["cell"] = batch["cell"].to(dtype=dtype)
        target = batch["target"].to(dtype=dtype)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            prediction = model(batch)
            residual = prediction - target
            loss = torch.mean(residual.square())
        if train:
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        totals["mae_eV_per_atom"] += float(torch.mean(torch.abs(residual)).detach().cpu())
        totals["rmse_eV_per_atom"] += float(torch.sqrt(loss).detach().cpu())
        count += 1
    if not count:
        raise ValueError("empty data loader")
    return {key: value / count for key, value in totals.items()}


def train_crystal_gnn(config: CrystalGNNConfig | None = None) -> TrainingResult:
    import torch

    config = config or CrystalGNNConfig()
    seed_everything(config.train.seed, deterministic=config.train.deterministic)
    workspace = MLWorkspace(config.train.workspace_root)
    run = workspace.create_run(
        "prediction",
        "crystal-gnn",
        name=config.train.run_name,
        config=config,
    )
    data = prepare_matbench_data(config.data, config.model, workspace=workspace)
    mean, std = fit_target_normalization(data.train_dataloader())
    model_config = CrystalGNNModelConfig(**asdict(config.model))
    model_config.target_mean = mean
    model_config.target_std = std
    model = CrystalGNN(model_config)
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
            "model_name": "crystal_gnn",
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": row,
            "target": config.data.target,
            "dataset": config.data.task,
        }
        save_checkpoint(last_path, payload)
        if valid_metrics["mae_eV_per_atom"] < best_metric - config.train.min_delta:
            best_metric = valid_metrics["mae_eV_per_atom"]
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


__all__ = ["train_crystal_gnn"]
