from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
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
from .config import ZNNPConfig, ZNNPModelConfig
from .data import fit_energy_normalization, prepare_rmd17_datamodule
from .model import ZNNP


def _dtype(name: str):
    import torch

    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype: {name}") from exc


def _batch_loss(
    model: ZNNP,
    batch: dict[str, Any],
    *,
    energy_weight: float,
    force_weight: float,
    train: bool,
):
    import torch

    structure = dict(batch["structure"])
    positions = structure["pos"].clone().requires_grad_(True)
    structure["pos"] = positions
    structure["positions"] = positions
    output = model(structure)
    predicted_energy = output["energy"].reshape(-1)
    target_energy = batch["targets"]["energy"].reshape(-1).to(predicted_energy)
    natoms = structure["natoms"].reshape(-1).to(predicted_energy)
    predicted_forces = -torch.autograd.grad(
        predicted_energy.sum(),
        positions,
        create_graph=train,
        retain_graph=train,
    )[0]
    target_forces = batch["targets"]["forces"].to(predicted_forces)
    energy_residual = (predicted_energy - target_energy) / natoms
    energy_loss = torch.mean(energy_residual.square())
    force_loss = torch.mean((predicted_forces - target_forces).square())
    total = energy_weight * energy_loss + force_weight * force_loss
    metrics = {
        "loss": float(total.detach().cpu()),
        "energy_rmse_eV_per_atom": float(torch.sqrt(energy_loss).detach().cpu()),
        "force_rmse_eV_per_A": float(torch.sqrt(force_loss).detach().cpu()),
        "energy_mae_eV_per_atom": float(torch.mean(torch.abs(energy_residual)).detach().cpu()),
        "force_mae_eV_per_A": float(
            torch.mean(torch.abs(predicted_forces - target_forces)).detach().cpu()
        ),
    }
    return total, metrics


def _run_epoch(
    model: ZNNP,
    loader: Any,
    *,
    optimizer: Any | None,
    device: Any,
    dtype: Any,
    energy_weight: float,
    force_weight: float,
    gradient_clip_norm: float | None,
) -> dict[str, float]:
    import torch

    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        structure = batch["structure"]
        for key in ("pos", "cell"):
            structure[key] = structure[key].to(dtype=dtype)
        for key, value in batch["targets"].items():
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                batch["targets"][key] = value.to(dtype=dtype)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss, metrics = _batch_loss(
                model,
                batch,
                energy_weight=energy_weight,
                force_weight=force_weight,
                train=train,
            )
        if train:
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    if count == 0:
        raise ValueError("empty data loader")
    return {key: value / count for key, value in totals.items()}


def train_znnp(config: ZNNPConfig | None = None) -> TrainingResult:
    import torch

    config = config or ZNNPConfig()
    seed_everything(config.train.seed, deterministic=config.train.deterministic)
    workspace = MLWorkspace(config.train.workspace_root)
    run = workspace.create_run("mlff", "znnp", name=config.train.run_name, config=config)
    datamodule = prepare_rmd17_datamodule(config.data, workspace=workspace)
    train_loader = datamodule.train_dataloader()
    valid_loader = datamodule.val_dataloader()
    shift, scale = fit_energy_normalization(train_loader)
    model_config = ZNNPModelConfig(**asdict(config.model))
    model_config.energy_shift_eV_per_atom = shift
    model_config.energy_scale_eV = scale
    model = ZNNP(model_config)
    device = resolve_device(config.train.device)
    dtype = _dtype(config.train.dtype)
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
            train_loader,
            optimizer=optimizer,
            device=device,
            dtype=dtype,
            energy_weight=config.train.energy_weight,
            force_weight=config.train.force_weight,
            gradient_clip_norm=config.train.gradient_clip_norm,
        )
        valid_metrics = _run_epoch(
            model,
            valid_loader,
            optimizer=None,
            device=device,
            dtype=dtype,
            energy_weight=config.train.energy_weight,
            force_weight=config.train.force_weight,
            gradient_clip_norm=None,
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
        checkpoint = {
            "model_name": "znnp",
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": row,
        }
        save_checkpoint(last_path, checkpoint)
        current = valid_metrics["loss"]
        if current < best_metric - config.train.min_delta:
            best_metric = current
            stale = 0
            save_checkpoint(best_path, checkpoint)
        else:
            stale += 1
            if stale >= config.train.patience:
                break
    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state"])
    model.eval()
    return TrainingResult(
        run_dir=run.root,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        history=history,
        best_metric=best_metric,
        model=model,
    )


__all__ = ["train_znnp"]
