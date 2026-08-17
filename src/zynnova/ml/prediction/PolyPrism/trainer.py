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
from ..PolyPrediction.physics import physics_consistency_loss
from ..PolyPrediction.trainer import balanced_heteroscedastic_loss
from .config import PolyPrismConfig
from .data import prepare_poly_prism_data
from .model import PolyPrismNetwork
from .objectives import balanced_student_t_nll, evidential_regularizer


def _run(model, loader, *, optimizer, device, dtype, normalizer, config):
    import torch

    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        for key in (
            "node_features", "edge_features", "node_weights", "conditions",
            "raw_conditions", "targets", "physics_descriptors",
        ):
            batch[key] = batch[key].to(dtype=dtype)
        if training:
            optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        if config.model.uncertainty == "evidential":
            data_loss = balanced_student_t_nll(
                output, batch["targets"], batch["target_mask"]
            )
            evidence = evidential_regularizer(
                output, batch["targets"], batch["target_mask"]
            )
        else:
            data_loss = balanced_heteroscedastic_loss(
                output["mean"], output["log_variance"],
                batch["targets"], batch["target_mask"],
            )
            evidence = data_loss * 0.0
        physics = physics_consistency_loss(
            output["mean"], batch, normalizer=normalizer,
            property_names=model.property_names,
            condition_names=model.config.condition_names,
            entropy_weight=config.train.entropy_consistency_weight,
        )
        total = (
            data_loss
            + config.train.evidential_regularization * evidence
            + config.train.physics_loss_weight * physics
            + config.train.expert_balance_weight * output["expert_balance_loss"]
        )
        if training:
            total.backward()
            if config.train.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.train.gradient_clip_norm
                )
            optimizer.step()
        absolute = (output["mean"] - batch["targets"]).abs()
        mae = (absolute * batch["target_mask"]).sum() / batch["target_mask"].sum().clamp_min(1)
        metrics = {
            "loss": float(total.detach().cpu()),
            "data_nll": float(data_loss.detach().cpu()),
            "physics_loss": float(physics.detach().cpu()),
            "expert_balance": float(output["expert_balance_loss"].detach().cpu()),
            "mae_z": float(mae.detach().cpu()),
        }
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
    if not batches:
        raise ValueError("empty data loader")
    return {key: value / batches for key, value in totals.items()}


def train_poly_prism(config: PolyPrismConfig | None = None, *, samples=None) -> TrainingResult:
    import torch

    config = config or PolyPrismConfig()
    config.model.__post_init__()
    seed_everything(config.train.seed, deterministic=config.train.deterministic)
    workspace = MLWorkspace(config.train.workspace_root)
    run = workspace.create_run(
        "prediction", "poly-prism", name=config.train.run_name, config=config
    )
    data = prepare_poly_prism_data(
        config.data, config.model, workspace=workspace, samples=samples
    )
    if not data.split_sizes["valid"]:
        raise ValueError("PolyPrism requires a non-empty validation split")
    model = PolyPrismNetwork(config.model)
    device = resolve_device(config.train.device)
    dtype = getattr(torch, config.train.dtype)
    model.to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
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
            normalizer=data.target_normalizer, config=config,
        )
        with torch.no_grad():
            valid_metrics = _run(
                model, data.valid, optimizer=None, device=device, dtype=dtype,
                normalizer=data.target_normalizer, config=config,
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
            "model_name": "poly_prism",
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
                normalizer=data.target_normalizer,
                config=config,
            )
        append_jsonl(
            run.history,
            {f"test_{name}": value for name, value in test_metrics.items()},
        )
    return TrainingResult(run.root, best_path, last_path, history, best, model)


__all__ = ["train_poly_prism"]
