from __future__ import annotations

import math
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
from .config import PolyGenConfig
from .data import prepare_poly_gen_data
from .model import PolymerMaskedFlow


def corrupt_discrete_flow(token_ids, attention_mask, time, *, tokenizer):
    import torch

    eligible = attention_mask.bool()
    eligible[:, 0] = False
    eligible &= token_ids.ne(tokenizer.eos_id)
    eligible &= token_ids.ne(tokenizer.pad_id)
    mask_probability = torch.sin(0.5 * math.pi * time).square()
    selected = torch.rand_like(token_ids, dtype=torch.float32) < mask_probability[:, None]
    selected &= eligible
    for row in range(token_ids.shape[0]):
        if not selected[row].any():
            positions = torch.nonzero(eligible[row], as_tuple=False).flatten()
            if positions.numel():
                choice = positions[torch.randint(positions.numel(), (1,), device=token_ids.device)]
                selected[row, choice] = True
    corrupted = token_ids.clone()
    corrupted[selected] = tokenizer.mask_id
    return corrupted, selected


def _masked_property_loss(prediction, target, mask):
    import torch

    if not mask.any():
        return prediction.sum() * 0.0
    losses = []
    for column in range(target.shape[1]):
        observed = mask[:, column]
        if observed.any():
            losses.append(
                torch.nn.functional.smooth_l1_loss(
                    prediction[observed, column],
                    target[observed, column],
                )
            )
    return torch.stack(losses).mean()


def _losses(
    model,
    batch,
    *,
    tokenizer,
    condition_dropout: float,
    train: bool,
    config: PolyGenConfig,
):
    import torch

    batch_size = batch["token_ids"].shape[0]
    time = torch.rand(
        batch_size,
        device=batch["token_ids"].device,
        dtype=batch["properties"].dtype,
    ).clamp_(1.0e-3, 1.0 - 1.0e-3)
    corrupted, flow_mask = corrupt_discrete_flow(
        batch["token_ids"],
        batch["attention_mask"],
        time,
        tokenizer=tokenizer,
    )
    property_mask = batch["property_mask"].clone()
    process_mask = batch["process_condition_mask"].clone()
    if train and condition_dropout > 0:
        dropped = torch.rand(batch_size, device=time.device) < condition_dropout
        property_mask[dropped] = False
        process_mask[dropped] = False
    logits = model(
        corrupted,
        batch["attention_mask"],
        time,
        batch["properties"],
        property_mask,
        batch["process_conditions"],
        process_mask,
    )
    token_loss = torch.nn.functional.cross_entropy(
        logits[flow_mask],
        batch["token_ids"][flow_mask],
    )
    token_accuracy = (
        logits[flow_mask].argmax(dim=-1).eq(batch["token_ids"][flow_mask]).float().mean()
    )
    property_prediction = model.predict_properties(
        batch["token_ids"],
        batch["attention_mask"],
    )
    property_loss = _masked_property_loss(
        property_prediction,
        batch["properties"],
        batch["property_mask"],
    )
    length_logits = model.predict_length(
        batch["properties"],
        property_mask,
        batch["process_conditions"],
        process_mask,
    )
    length_loss = torch.nn.functional.cross_entropy(length_logits, batch["length"])
    total = (
        token_loss
        + config.train.property_loss_weight * property_loss
        + config.train.length_loss_weight * length_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "flow_token_nll": float(token_loss.detach().cpu()),
        "masked_token_accuracy": float(token_accuracy.detach().cpu()),
        "property_loss": float(property_loss.detach().cpu()),
        "length_loss": float(length_loss.detach().cpu()),
    }


def _run_epoch(
    model,
    loader: Any,
    *,
    optimizer: Any | None,
    device: Any,
    dtype: Any,
    tokenizer,
    config: PolyGenConfig,
) -> dict[str, float]:
    import torch

    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        for name in ("properties", "process_conditions"):
            batch[name] = batch[name].to(dtype=dtype)
        if train:
            optimizer.zero_grad(set_to_none=True)
        loss, metrics = _losses(
            model,
            batch,
            tokenizer=tokenizer,
            condition_dropout=config.train.condition_dropout,
            train=train,
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


def train_poly_gen(
    config: PolyGenConfig | None = None,
    *,
    samples=None,
) -> TrainingResult:
    import torch

    config = config or PolyGenConfig()
    config.model.__post_init__()
    seed_everything(config.train.seed, deterministic=config.train.deterministic)
    workspace = MLWorkspace(config.train.workspace_root)
    run = workspace.create_run(
        "generation",
        "poly-gen",
        name=config.train.run_name,
        config=config,
    )
    data = prepare_poly_gen_data(
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
    model = PolymerMaskedFlow(config.model)
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
            tokenizer=data.tokenizer,
            config=config,
        )
        with torch.no_grad():
            valid_metrics = _run_epoch(
                model,
                data.val_dataloader(),
                optimizer=None,
                device=device,
                dtype=dtype,
                tokenizer=data.tokenizer,
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
            "model_name": "poly_gen",
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
                tokenizer=data.tokenizer,
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


__all__ = ["corrupt_discrete_flow", "train_poly_gen"]
