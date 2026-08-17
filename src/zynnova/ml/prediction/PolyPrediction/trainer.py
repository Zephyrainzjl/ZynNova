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
from .config import PolyPredictionConfig
from .data import prepare_poly_prediction_data
from .model import PolyPredictionNetwork
from .physics import physics_consistency_loss


def balanced_heteroscedastic_loss(mean, log_variance, target, mask):
    import torch

    squared = (mean - target).square()
    nll = 0.5 * (torch.exp(-log_variance) * squared + log_variance)
    per_property = []
    for index in range(target.shape[1]):
        observed = mask[:, index]
        if observed.any():
            per_property.append(nll[observed, index].mean())
    if not per_property:
        raise ValueError("batch contains no observed property targets")
    return torch.stack(per_property).mean()


def _losses(model, batch, *, normalizer, config: PolyPredictionConfig):
    output = model(batch)
    prediction = output["mean"]
    log_variance = output["log_variance"]
    mask = batch["target_mask"]
    nll = balanced_heteroscedastic_loss(
        prediction,
        log_variance,
        batch["targets"],
        mask,
    )
    physics = physics_consistency_loss(
        prediction,
        batch,
        normalizer=normalizer,
        property_names=model.property_names,
        condition_names=model.config.condition_names,
        entropy_weight=config.train.entropy_consistency_weight,
    )
    total = nll + config.train.physics_loss_weight * physics
    absolute = (prediction - batch["targets"]).abs()
    observed_count = mask.sum().clamp_min(1)
    mae = (absolute * mask).sum() / observed_count
    metrics = {
        "loss": float(total.detach().cpu()),
        "nll": float(nll.detach().cpu()),
        "physics_loss": float(physics.detach().cpu()),
        "mae_z": float(mae.detach().cpu()),
    }
    for index, name in enumerate(model.property_names):
        observed = mask[:, index]
        if observed.any():
            metrics[f"mae_z_{name}"] = float(absolute[observed, index].mean().detach().cpu())
    return total, metrics


def _run_epoch(
    model,
    loader: Any,
    *,
    optimizer: Any | None,
    device: Any,
    dtype: Any,
    normalizer,
    config: PolyPredictionConfig,
) -> dict[str, float]:
    import torch

    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        for name in (
            "node_features",
            "edge_features",
            "node_weights",
            "conditions",
            "raw_conditions",
            "targets",
            "physics_descriptors",
        ):
            batch[name] = batch[name].to(dtype=dtype)
        if train:
            optimizer.zero_grad(set_to_none=True)
        loss, metrics = _losses(
            model,
            batch,
            normalizer=normalizer,
            config=config,
        )
        if train:
            loss.backward()
            if config.train.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.train.gradient_clip_norm,
                )
            optimizer.step()
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
        batches += 1
    if not batches:
        raise ValueError("empty data loader")
    return {name: value / batches for name, value in totals.items()}


def train_poly_prediction(
    config: PolyPredictionConfig | None = None,
    *,
    samples=None,
) -> TrainingResult:
    import torch

    config = config or PolyPredictionConfig()
    config.model.__post_init__()
    seed_everything(config.train.seed, deterministic=config.train.deterministic)
    workspace = MLWorkspace(config.train.workspace_root)
    run = workspace.create_run(
        "prediction",
        "poly-prediction",
        name=config.train.run_name,
        config=config,
    )
    data = prepare_poly_prediction_data(
        config.data,
        config.model,
        workspace=workspace,
        samples=samples,
    )
    if not data.split_sizes["valid"]:
        raise ValueError(
            "validation split is empty; provide at least two polymer families or "
            "predefined train/valid splits"
        )
    model = PolyPredictionNetwork(config.model)
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
            normalizer=data.target_normalizer,
            config=config,
        )
        with torch.no_grad():
            valid_metrics = _run_epoch(
                model,
                data.val_dataloader(),
                optimizer=None,
                device=device,
                dtype=dtype,
                normalizer=data.target_normalizer,
                config=config,
            )
        scheduler.step(valid_metrics["loss"])
        row = {
            "epoch": float(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"valid_{name}": value for name, value in valid_metrics.items()},
        }
        history.append(row)
        append_jsonl(run.history, row)
        payload = {
            "model_name": "poly_prediction",
            "model_config": asdict(config.model),
            "model_state": model.state_dict(),
            "tokenizer": data.tokenizer.state_dict(),
            "target_normalizer": data.target_normalizer.state_dict(),
            "condition_normalizer": data.condition_normalizer.state_dict(),
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
    model.eval()
    if data.split_sizes["test"]:
        with torch.no_grad():
            test_metrics = _run_epoch(
                model,
                data.test_dataloader(),
                optimizer=None,
                device=device,
                dtype=dtype,
                normalizer=data.target_normalizer,
                config=config,
            )
        append_jsonl(run.history, {f"test_{name}": value for name, value in test_metrics.items()})
    return TrainingResult(
        run_dir=run.root,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        history=history,
        best_metric=best_metric,
        model=model,
    )


__all__ = ["balanced_heteroscedastic_loss", "train_poly_prediction"]
