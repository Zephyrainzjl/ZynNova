from __future__ import annotations

from dataclasses import asdict

from ...common import (
    TrainingResult,
    append_jsonl,
    move_to_device,
    resolve_device,
    save_checkpoint,
    seed_everything,
)
from ...workspace import MLWorkspace
from .config import PolyLoomConfig
from .data import prepare_poly_loom_data
from .model import PolyLoomNetwork
from .objectives import polyloom_losses


def _run(model, loader, *, optimizer, device, dtype, tokenizer, config):
    import torch

    training = optimizer is not None
    model.train(training)
    totals, batches = {}, 0
    for batch in loader:
        batch = move_to_device(batch, device)
        for key in ("properties", "process_conditions"):
            batch[key] = batch[key].to(dtype=dtype)
        if training:
            optimizer.zero_grad(set_to_none=True)
        loss, metrics = polyloom_losses(
            model, batch, tokenizer=tokenizer, config=config, training=training
        )
        if training:
            loss.backward()
            if config.train.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.train.gradient_clip_norm
                )
            optimizer.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
    if not batches:
        raise ValueError("empty data loader")
    return {key: value / batches for key, value in totals.items()}


def train_poly_loom(config: PolyLoomConfig | None = None, *, samples=None) -> TrainingResult:
    import torch

    config = config or PolyLoomConfig()
    config.model.__post_init__()
    seed_everything(config.train.seed, deterministic=config.train.deterministic)
    workspace = MLWorkspace(config.train.workspace_root)
    run = workspace.create_run(
        "generation", "poly-loom", name=config.train.run_name, config=config
    )
    data = prepare_poly_loom_data(
        config.data, config.model, workspace=workspace, samples=samples
    )
    if not data.split_sizes["valid"]:
        raise ValueError("PolyLoom requires a non-empty validation split")
    model = PolyLoomNetwork(config.model)
    device = resolve_device(config.train.device)
    dtype = getattr(torch, config.train.dtype)
    model.to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.train.epochs, 1), eta_min=1.0e-6
    )
    best, stale, history = float("inf"), 0, []
    best_path, last_path = run.checkpoints / "best.pt", run.checkpoints / "last.pt"
    for epoch in range(1, config.train.epochs + 1):
        train_metrics = _run(
            model, data.train, optimizer=optimizer, device=device, dtype=dtype,
            tokenizer=data.tokenizer, config=config,
        )
        with torch.no_grad():
            valid_metrics = _run(
                model, data.valid, optimizer=None, device=device, dtype=dtype,
                tokenizer=data.tokenizer, config=config,
            )
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        }
        history.append(row)
        append_jsonl(run.history, row)
        payload = {
            "model_name": "poly_loom",
            "model_config": asdict(config.model),
            "model_state": model.state_dict(),
            "tokenizer": data.tokenizer.state_dict(),
            "property_normalizer": data.property_normalizer.state_dict(),
            "process_normalizer": data.process_normalizer.state_dict(),
            "split_sizes": data.split_sizes,
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": row,
        }
        save_checkpoint(last_path, payload)
        if valid_metrics["loss"] < best - config.train.min_delta:
            best, stale = valid_metrics["loss"], 0
            save_checkpoint(best_path, payload)
        else:
            stale += 1
            if stale >= config.train.patience:
                break
    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state"])
    model.eval()
    if data.split_sizes["test"]:
        with torch.no_grad():
            test_metrics = _run(
                model,
                data.test_dataloader(),
                optimizer=None,
                device=device,
                dtype=dtype,
                tokenizer=data.tokenizer,
                config=config,
            )
        append_jsonl(
            run.history,
            {f"test_{name}": value for name, value in test_metrics.items()},
        )
    return TrainingResult(run.root, best_path, last_path, history, best, model)


__all__ = ["train_poly_loom"]
