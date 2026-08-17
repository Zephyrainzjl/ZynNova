from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any

from ...common import (
    TrainingResult,
    append_jsonl,
    load_checkpoint,
    move_to_device,
    resolve_device,
    save_checkpoint,
    seed_everything,
)
from ...workspace import MLWorkspace
from .config import JouleWeaveConfig, jouleweave_model_config_from_dict
from .data import fit_energy_statistics, prepare_jouleweave_datamodule
from .model import JouleWeave


def _dtype(name: str):
    import torch

    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype: {name}") from exc


def _regression_loss(prediction: Any, target: Any, *, kind: str, delta: float):
    import torch

    if kind == "mse":
        return torch.mean((prediction - target).square())
    return torch.nn.functional.huber_loss(
        prediction,
        target,
        reduction="mean",
        delta=delta,
    )


def _atom_mask(batch: dict[str, Any], name: str, reference: Any) -> Any:
    import torch

    value = batch.get("masks", {}).get(name)
    if value is None:
        return torch.ones(
            reference.shape[0],
            device=reference.device,
            dtype=torch.bool,
        )
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.to(device=reference.device, dtype=torch.bool).reshape(-1)
    if value.shape != (reference.shape[0],):
        raise ValueError(
            f"{name} mask must have shape ({reference.shape[0]},); "
            f"got {tuple(value.shape)}"
        )
    return value


def _masked_regression_loss(
    prediction: Any,
    target: Any,
    mask: Any,
    *,
    kind: str,
    delta: float,
) -> Any:
    if not bool(mask.any().item()):
        return prediction.sum() * 0.0
    return _regression_loss(
        prediction[mask],
        target[mask],
        kind=kind,
        delta=delta,
    )


def _voigt(stress: Any) -> Any:
    if stress.ndim >= 2 and stress.shape[-2:] == (3, 3):
        return stress[
            ...,
            [0, 1, 2, 1, 0, 0],
            [0, 1, 2, 2, 2, 1],
        ]
    if stress.shape[-1] == 6:
        return stress
    raise ValueError("stress target must end in [6] or [3, 3]")


class ExponentialMovingAverage:
    def __init__(self, model: Any, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }

    def update(self, model: Any) -> None:
        with __import__("torch").no_grad():
            for name, value in model.state_dict().items():
                if name not in self.shadow:
                    continue
                self.shadow[name].lerp_(value.detach(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {name: value.detach().clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.shadow = {name: value.detach().clone() for name, value in state.items()}

    def copy_to(self, model: Any) -> None:
        state = model.state_dict()
        with __import__("torch").no_grad():
            for name, value in self.shadow.items():
                if name in state:
                    state[name].copy_(value.to(state[name]))

    @contextmanager
    def average_parameters(self, model: Any):
        backup = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if name in self.shadow
        }
        self.copy_to(model)
        try:
            yield
        finally:
            state = model.state_dict()
            with __import__("torch").no_grad():
                for name, value in backup.items():
                    state[name].copy_(value)


def _model_inputs(batch: dict[str, Any], *, dtype: Any) -> dict[str, Any]:
    structure = dict(batch["structure"])
    for key in ("pos", "cell"):
        if key in structure:
            structure[key] = structure[key].to(dtype=dtype)
    structure["positions"] = structure["pos"]
    conditions = batch.get("conditions", {})
    for name in ("total_charge", "spin", "fidelity"):
        if name in conditions:
            structure[name] = conditions[name]
    return structure


def _batch_loss(
    model: JouleWeave,
    batch: dict[str, Any],
    *,
    train: bool,
    dtype: Any,
    config: Any,
) -> tuple[Any, dict[str, float]]:
    import torch

    inputs = _model_inputs(batch, dtype=dtype)
    positions = inputs["pos"].clone().requires_grad_(True)
    inputs["pos"] = positions
    inputs["positions"] = positions
    targets = {
        name: value.to(dtype=dtype) if value.is_floating_point() else value
        for name, value in batch["targets"].items()
    }
    need_stress = config.stress_weight > 0 and "stress" in targets
    if need_stress:
        output = model.energy_forces_stress(
            inputs,
            create_graph=train,
            compute_stress=True,
        )
    else:
        output = model.energy_and_forces(inputs, create_graph=train)

    predicted_energy = output["energy"].reshape(-1)
    target_energy = targets["energy"].reshape(-1).to(predicted_energy)
    natoms = inputs["natoms"].reshape(-1).to(predicted_energy).clamp_min(1.0)
    energy_residual = (predicted_energy - target_energy) / natoms
    energy_loss = _regression_loss(
        energy_residual,
        torch.zeros_like(energy_residual),
        kind=config.loss,
        delta=config.huber_delta,
    )

    predicted_forces = output["forces"]
    target_forces = targets["forces"].to(predicted_forces)
    force_residual = predicted_forces - target_forces
    force_loss = _regression_loss(
        force_residual,
        torch.zeros_like(force_residual),
        kind=config.loss,
        delta=config.huber_delta,
    )
    total = config.energy_weight * energy_loss + config.force_weight * force_loss

    stress_loss = predicted_energy.new_zeros(())
    if need_stress:
        predicted_stress = _voigt(output["stress"])
        target_stress = _voigt(targets["stress"]).to(predicted_stress)
        stress_loss = _regression_loss(
            predicted_stress,
            target_stress,
            kind=config.loss,
            delta=config.huber_delta,
        )
        total = total + config.stress_weight * stress_loss

    charge_loss = predicted_energy.new_zeros(())
    charge_mae = predicted_energy.new_zeros(())
    charge_rmse = predicted_energy.new_zeros(())
    if config.charge_weight > 0 and "charges" in targets:
        predicted_charges = output["charges"].reshape(-1)
        charge_target = targets["charges"].reshape(-1).to(predicted_charges)
        if charge_target.shape != predicted_charges.shape:
            raise ValueError(
                "charges target must contain exactly one value per atom; "
                f"got {tuple(charge_target.shape)} for {tuple(predicted_charges.shape)}"
            )
        charge_mask = _atom_mask(batch, "charges", predicted_charges)
        charge_loss = _masked_regression_loss(
            predicted_charges,
            charge_target,
            charge_mask,
            kind=config.loss,
            delta=config.huber_delta,
        )
        total = total + config.charge_weight * charge_loss
        if bool(charge_mask.any().item()):
            charge_residual = predicted_charges[charge_mask] - charge_target[charge_mask]
            charge_mae = torch.mean(torch.abs(charge_residual))
            charge_rmse = torch.sqrt(torch.mean(charge_residual.square()))

    qeq_consistency_loss = predicted_energy.new_zeros(())
    if config.charge_qeq_consistency_weight > 0:
        partition_charges = output.get("partition_charges")
        qeq_charges = output.get("qeq_charges")
        if partition_charges is None or qeq_charges is None:
            raise ValueError(
                "charge_qeq_consistency_weight > 0 requires both the charge head and QEq"
            )
        qeq_consistency_loss = _regression_loss(
            partition_charges,
            qeq_charges,
            kind=config.loss,
            delta=config.huber_delta,
        )
        total = total + config.charge_qeq_consistency_weight * qeq_consistency_loss

    dipole_loss = predicted_energy.new_zeros(())
    if config.dipole_weight > 0 and "dipole" in targets:
        dipole_target = targets["dipole"].to(output["dipole"])
        dipole_loss = _regression_loss(
            output["dipole"],
            dipole_target,
            kind=config.loss,
            delta=config.huber_delta,
        )
        total = total + config.dipole_weight * dipole_loss

    magmom_loss = predicted_energy.new_zeros(())
    magmom_mae = predicted_energy.new_zeros(())
    magmom_rmse = predicted_energy.new_zeros(())
    if config.magmom_weight > 0 and "magmoms" in targets:
        if "magmoms" not in output:
            raise ValueError(
                "magmom_weight > 0 requires a model configured with use_magmoms=True"
            )
        predicted_magmoms = output["magmoms"].reshape(-1)
        target_magmoms = targets["magmoms"].reshape(-1).to(predicted_magmoms)
        if model.config.magmom_nonnegative:
            target_magmoms = torch.abs(target_magmoms)
        if target_magmoms.shape != predicted_magmoms.shape:
            raise ValueError(
                "magmoms target must contain exactly one value per atom; "
                f"got {tuple(target_magmoms.shape)} for {tuple(predicted_magmoms.shape)}"
            )
        magmom_mask = _atom_mask(batch, "magmoms", predicted_magmoms)
        magmom_loss = _masked_regression_loss(
            predicted_magmoms,
            target_magmoms,
            magmom_mask,
            kind=config.loss,
            delta=config.huber_delta,
        )
        total = total + config.magmom_weight * magmom_loss
        if bool(magmom_mask.any().item()):
            magmom_residual = (
                predicted_magmoms[magmom_mask] - target_magmoms[magmom_mask]
            )
            magmom_mae = torch.mean(torch.abs(magmom_residual))
            magmom_rmse = torch.sqrt(torch.mean(magmom_residual.square()))

    oxidation_loss = predicted_energy.new_zeros(())
    oxidation_accuracy = predicted_energy.new_zeros(())
    oxidation_mae = predicted_energy.new_zeros(())
    if config.oxidation_state_weight > 0 and "oxidation_states" in targets:
        logits = output.get("oxidation_state_logits")
        if logits is None:
            raise ValueError(
                "oxidation_state_weight > 0 requires use_oxidation_states=True"
            )
        expected = output["oxidation_states"].reshape(-1)
        target_values = targets["oxidation_states"].reshape(-1).to(expected)
        if target_values.shape != expected.shape:
            raise ValueError(
                "oxidation_states target must contain one value per atom; "
                f"got {tuple(target_values.shape)} for {tuple(expected.shape)}"
            )
        oxidation_mask = _atom_mask(batch, "oxidation_states", expected)
        if bool(oxidation_mask.any().item()):
            labelled = target_values[oxidation_mask]
            rounded = torch.round(labelled)
            if bool(torch.any(torch.abs(labelled - rounded) > 1.0e-4).item()):
                raise ValueError("oxidation-state labels must be integer-valued")
            class_target = rounded.long() - int(model.config.oxidation_state_min)
            if bool(
                torch.any(
                    (class_target < 0) | (class_target >= logits.shape[-1])
                ).item()
            ):
                raise ValueError(
                    "oxidation-state label lies outside the configured class range "
                    f"[{model.config.oxidation_state_min}, "
                    f"{model.config.oxidation_state_max}]"
                )
            oxidation_loss = torch.nn.functional.cross_entropy(
                logits[oxidation_mask],
                class_target,
                label_smoothing=config.oxidation_label_smoothing,
            )
            total = total + config.oxidation_state_weight * oxidation_loss
            predicted_class = (
                torch.argmax(logits[oxidation_mask], dim=-1)
                + int(model.config.oxidation_state_min)
            )
            oxidation_accuracy = torch.mean(
                (predicted_class == rounded.long()).to(expected.dtype)
            )
            oxidation_mae = torch.mean(torch.abs(expected[oxidation_mask] - labelled))

    metrics = {
        "loss": float(total.detach().cpu()),
        "energy_mae_eV_per_atom": float(torch.mean(torch.abs(energy_residual)).detach().cpu()),
        "energy_rmse_eV_per_atom": float(
            torch.sqrt(torch.mean(energy_residual.square())).detach().cpu()
        ),
        "force_mae_eV_per_A": float(torch.mean(torch.abs(force_residual)).detach().cpu()),
        "force_rmse_eV_per_A": float(
            torch.sqrt(torch.mean(force_residual.square())).detach().cpu()
        ),
        "stress_loss": float(stress_loss.detach().cpu()),
        "magmom_loss": float(magmom_loss.detach().cpu()),
        "magmom_mae_mu_B": float(magmom_mae.detach().cpu()),
        "magmom_rmse_mu_B": float(magmom_rmse.detach().cpu()),
        "charge_loss": float(charge_loss.detach().cpu()),
        "charge_mae_e": float(charge_mae.detach().cpu()),
        "charge_rmse_e": float(charge_rmse.detach().cpu()),
        "charge_qeq_consistency_loss": float(qeq_consistency_loss.detach().cpu()),
        "oxidation_state_loss": float(oxidation_loss.detach().cpu()),
        "oxidation_state_accuracy": float(oxidation_accuracy.detach().cpu()),
        "oxidation_state_mae": float(oxidation_mae.detach().cpu()),
        "dipole_loss": float(dipole_loss.detach().cpu()),
    }
    return total, metrics


def _run_epoch(
    model: JouleWeave,
    loader: Any,
    *,
    optimizer: Any | None,
    ema: ExponentialMovingAverage | None,
    device: Any,
    dtype: Any,
    config: Any,
) -> dict[str, float]:
    import torch

    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    count = 0
    if train:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader, start=1):
        batch = move_to_device(batch, device)
        with torch.enable_grad():
            loss, metrics = _batch_loss(
                model,
                batch,
                train=train,
                dtype=dtype,
                config=config,
            )
        if train:
            (loss / config.gradient_accumulation).backward()
            should_step = batch_index % config.gradient_accumulation == 0 or batch_index == len(
                loader
            )
            if should_step:
                if config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.gradient_clip_norm,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
        count += 1
    if count == 0:
        raise ValueError("empty data loader")
    return {name: value / count for name, value in totals.items()}


def _scheduler(optimizer: Any, config: Any):
    import torch

    minimum_ratio = config.min_learning_rate / config.learning_rate

    def scale(epoch: int) -> float:
        if config.warmup_epochs and epoch < config.warmup_epochs:
            return max((epoch + 1) / config.warmup_epochs, minimum_ratio)
        remaining = max(config.epochs - config.warmup_epochs, 1)
        progress = min(max((epoch - config.warmup_epochs) / remaining, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _set_backbone_trainable(model: JouleWeave, trainable: bool) -> None:
    head_prefixes = (
        "readouts.",
        "magnetic_constraint.",
        "redox_constraint.",
        "qeq_head.",
        "dispersion.",
        "zbl.",
    )
    for name, parameter in model.named_parameters():
        if not name.startswith(head_prefixes):
            parameter.requires_grad_(trainable)


def train_jouleweave(
    config: JouleWeaveConfig | None = None,
    *,
    source: Any | None = None,
) -> TrainingResult:
    """Train from scratch or fine-tune on any ZynNova potential dataset source."""

    import torch

    config = config or JouleWeaveConfig()
    if config.train.magmom_weight > 0 and not config.model.use_magmoms:
        raise ValueError("magmom_weight > 0 requires config.model.use_magmoms=True")
    if config.train.magmom_weight > 0 and not config.data.magmoms_source:
        raise ValueError("magmom_weight > 0 requires config.data.magmoms_source")
    if config.train.charge_weight > 0 and not (
        config.model.use_charge_head or config.model.use_qeq
    ):
        raise ValueError(
            "charge_weight > 0 requires use_charge_head=True or use_qeq=True"
        )
    if config.train.charge_weight > 0 and not config.data.charges_source:
        raise ValueError("charge_weight > 0 requires config.data.charges_source")
    if (
        config.train.charge_weight > 0
        and config.train.strict_electronic_labels
        and config.data.charge_label_scheme == "unspecified"
    ):
        raise ValueError(
            "strict electronic supervision requires charge_label_scheme "
            "(for example 'bader' or 'ddec6')"
        )
    if config.train.oxidation_state_weight > 0 and not config.model.use_oxidation_states:
        raise ValueError(
            "oxidation_state_weight > 0 requires config.model.use_oxidation_states=True"
        )
    if (
        config.train.oxidation_state_weight > 0
        and not config.data.oxidation_states_source
    ):
        raise ValueError(
            "oxidation_state_weight > 0 requires config.data.oxidation_states_source"
        )
    if (
        config.train.oxidation_state_weight > 0
        and config.train.strict_electronic_labels
        and not config.data.oxidation_label_method
    ):
        raise ValueError(
            "strict electronic supervision requires oxidation_label_method provenance"
        )
    if config.train.charge_qeq_consistency_weight > 0 and not (
        config.model.use_charge_head and config.model.use_qeq
    ):
        raise ValueError(
            "charge_qeq_consistency_weight > 0 requires use_charge_head=True "
            "and use_qeq=True"
        )
    seed_everything(config.train.seed, deterministic=config.train.deterministic)
    workspace = MLWorkspace(config.train.workspace_root)
    run = workspace.create_run(
        "mlff",
        "jouleweave",
        name=config.train.run_name,
        config=config,
    )
    datamodule = prepare_jouleweave_datamodule(
        config.data,
        workspace=workspace,
        source=source,
    )
    train_loader = datamodule.train_dataloader()
    valid_loader = datamodule.val_dataloader()

    model_config = jouleweave_model_config_from_dict(asdict(config.model))
    if model_config.charge_label_scheme == "unspecified":
        model_config.charge_label_scheme = config.data.charge_label_scheme
    elif (
        config.data.charge_label_scheme != "unspecified"
        and model_config.charge_label_scheme != config.data.charge_label_scheme
    ):
        raise ValueError(
            "model and data charge_label_scheme values must match; charge "
            "partitioning conventions cannot be mixed implicitly"
        )
    if model_config.oxidation_label_method is None:
        model_config.oxidation_label_method = config.data.oxidation_label_method
    elif (
        config.data.oxidation_label_method is not None
        and model_config.oxidation_label_method != config.data.oxidation_label_method
    ):
        raise ValueError(
            "model and data oxidation_label_method values must match"
        )
    fitted_normalization: tuple[dict[int, float], float, float] | None = None
    if config.train.reference_fit != "none":
        fitted_normalization = fit_energy_statistics(
            train_loader,
            max_atomic_number=model_config.max_atomic_number,
            mode=config.train.reference_fit,
            ridge=config.train.reference_ridge,
        )
        references, shift, scale = fitted_normalization
        model_config.atomic_reference_energies = references
        model_config.residual_shift_eV_per_atom = shift
        model_config.residual_scale_eV = scale
    model = JouleWeave(model_config)
    device = resolve_device(config.train.device)
    dtype = _dtype(config.train.dtype)
    model.to(device=device, dtype=dtype)

    if config.train.fine_tune_checkpoint is not None:
        payload = load_checkpoint(
            config.train.fine_tune_checkpoint,
            map_location=device,
        )
        state = payload.get("ema_model_state", payload["model_state"])
        model.load_state_dict(state, strict=config.train.strict_fine_tune)
        if fitted_normalization is not None:
            references, shift, scale = fitted_normalization
            model.set_energy_normalization(
                atomic_reference_energies=references,
                residual_shift_eV_per_atom=shift,
                residual_scale_eV=scale,
            )

    frozen = config.train.freeze_backbone_epochs > 0
    if frozen:
        _set_backbone_trainable(model, False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = _scheduler(optimizer, config.train)
    ema = ExponentialMovingAverage(model, config.train.ema_decay)

    best_metric = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    best_path = run.checkpoints / "best.pt"
    last_path = run.checkpoints / "last.pt"
    for epoch in range(1, config.train.epochs + 1):
        if frozen and epoch > config.train.freeze_backbone_epochs:
            _set_backbone_trainable(model, True)
            frozen = False
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            ema=ema,
            device=device,
            dtype=dtype,
            config=config.train,
        )
        with ema.average_parameters(model):
            valid_metrics = _run_epoch(
                model,
                valid_loader,
                optimizer=None,
                ema=None,
                device=device,
                dtype=dtype,
                config=config.train,
            )
            ema_model_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"valid_{name}": value for name, value in valid_metrics.items()},
        }
        history.append(row)
        append_jsonl(run.history, row)
        checkpoint = {
            "model_name": "jouleweave",
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "ema_model_state": ema_model_state,
            "ema_state": ema.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_config": asdict(config.train),
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

    best_payload = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(best_payload.get("ema_model_state", best_payload["model_state"]))
    model.eval()
    return TrainingResult(
        run_dir=run.root,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        history=history,
        best_metric=best_metric,
        model=model,
    )


__all__ = ["ExponentialMovingAverage", "train_jouleweave"]
