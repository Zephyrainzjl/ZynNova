"""Training utilities for the constant-potential JouleWeave extension."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ...common import move_to_device, require_torch, resolve_device, save_checkpoint


torch = require_torch()


@dataclass(slots=True)
class ConstantPotentialTrainConfig:
    epochs: int = 100
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-6
    energy_weight: float = 1.0
    force_weight: float = 20.0
    fermi_weight: float = 5.0
    electron_count_weight: float = 1.0
    charge_weight: float = 1.0
    reaction_weight: float = 1.0
    self_consistency_weight: float = 2.0
    uncertainty_weight: float = 0.1
    gradient_clip_norm: float = 10.0
    device: str = "auto"
    dtype: str = "float32"
    output_directory: str | Path = "jouleweave-constant-potential"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid constant-potential training controls")
        weights = (
            self.energy_weight,
            self.force_weight,
            self.fermi_weight,
            self.electron_count_weight,
            self.charge_weight,
            self.reaction_weight,
            self.self_consistency_weight,
            self.uncertainty_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("loss weights cannot be negative")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")


def _mse(prediction: Any, target: Any) -> Any:
    return torch.mean((prediction - target).square())


def constant_potential_loss(
    model: Any,
    batch: Mapping[str, Any],
    config: ConstantPotentialTrainConfig,
    *,
    create_graph: bool,
) -> tuple[Any, dict[str, float]]:
    inputs = dict(batch["inputs"])
    targets = batch["targets"]
    output = model.grand_potential_and_forces(inputs, create_graph=create_graph)
    zero = output["grand_potential"].sum() * 0.0
    losses: dict[str, Any] = {}

    if "grand_potential_eV" in targets:
        target = torch.as_tensor(targets["grand_potential_eV"]).to(output["grand_potential"])
        losses["energy"] = _mse(output["grand_potential"], target.reshape_as(output["grand_potential"]))
    elif "energy_eV" in targets:
        target = torch.as_tensor(targets["energy_eV"]).to(output["canonical_energy"])
        losses["energy"] = _mse(output["canonical_energy"], target.reshape_as(output["canonical_energy"]))
    else:
        losses["energy"] = zero

    if "forces_eV_A" in targets:
        target = torch.as_tensor(targets["forces_eV_A"]).to(output["forces"])
        losses["forces"] = _mse(output["forces"], target)
    else:
        losses["forces"] = zero
    if "fermi_level_eV" in targets:
        target = torch.as_tensor(targets["fermi_level_eV"]).to(output["fermi_level_eV"])
        losses["fermi"] = _mse(output["fermi_level_eV"], target.reshape_as(output["fermi_level_eV"]))
    else:
        losses["fermi"] = zero
    if "electron_count" in targets:
        target = torch.as_tensor(targets["electron_count"]).to(output["electron_count"])
        losses["electron_count"] = _mse(output["electron_count"], target.reshape_as(output["electron_count"]))
    else:
        losses["electron_count"] = zero
    if "charges_e" in targets:
        target = torch.as_tensor(targets["charges_e"]).to(output["charges"])
        losses["charges"] = _mse(output["charges"], target)
    else:
        losses["charges"] = zero
    if "reaction_labels" in targets:
        target = torch.as_tensor(targets["reaction_labels"]).to(output["reaction_logits"])
        losses["reaction"] = torch.nn.functional.binary_cross_entropy_with_logits(
            output["reaction_logits"], target
        )
    else:
        losses["reaction"] = zero
    losses["self_consistency"] = torch.mean(
        output["self_consistency_residual_eV"].square()
    )

    residual = None
    if "grand_potential_eV" in targets:
        target = torch.as_tensor(targets["grand_potential_eV"]).to(output["grand_potential"])
        residual = output["grand_potential"] - target.reshape_as(output["grand_potential"])
    elif "energy_eV" in targets:
        target = torch.as_tensor(targets["energy_eV"]).to(output["canonical_energy"])
        residual = output["canonical_energy"] - target.reshape_as(output["canonical_energy"])
    if residual is None:
        losses["uncertainty"] = zero
    else:
        sigma = output["energy_standard_uncertainty_eV"].clamp_min(1.0e-8)
        losses["uncertainty"] = torch.mean(
            0.5 * (residual / sigma).square() + torch.log(sigma)
        )

    total = (
        config.energy_weight * losses["energy"]
        + config.force_weight * losses["forces"]
        + config.fermi_weight * losses["fermi"]
        + config.electron_count_weight * losses["electron_count"]
        + config.charge_weight * losses["charges"]
        + config.reaction_weight * losses["reaction"]
        + config.self_consistency_weight * losses["self_consistency"]
        + config.uncertainty_weight * losses["uncertainty"]
    )
    metrics = {name: float(value.detach().cpu().item()) for name, value in losses.items()}
    metrics["total"] = float(total.detach().cpu().item())
    return total, metrics


def train_constant_potential(
    model: Any,
    train_loader: Iterable[Mapping[str, Any]],
    valid_loader: Iterable[Mapping[str, Any]],
    config: ConstantPotentialTrainConfig | None = None,
) -> dict[str, Any]:
    resolved = config or ConstantPotentialTrainConfig()
    device = resolve_device(resolved.device)
    try:
        dtype = getattr(torch, resolved.dtype)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype {resolved.dtype!r}") from exc
    model = model.to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=resolved.learning_rate,
        weight_decay=resolved.weight_decay,
    )
    output_directory = Path(resolved.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    best_path = output_directory / "best.pt"
    last_path = output_directory / "last.pt"
    best_metric = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(resolved.epochs):
        model.train()
        train_totals: list[float] = []
        for raw_batch in train_loader:
            batch = move_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = constant_potential_loss(
                model, batch, resolved, create_graph=True
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), resolved.gradient_clip_norm)
            optimizer.step()
            train_totals.append(float(loss.detach().cpu().item()))
        model.eval()
        valid_totals: list[float] = []
        # Force targets require coordinate derivatives, so evaluation cannot use no_grad.
        for raw_batch in valid_loader:
            batch = move_to_device(raw_batch, device)
            loss, _ = constant_potential_loss(
                model, batch, resolved, create_graph=False
            )
            valid_totals.append(float(loss.detach().cpu().item()))
        record = {
            "epoch": float(epoch),
            "train_loss": float(sum(train_totals) / max(len(train_totals), 1)),
            "valid_loss": float(sum(valid_totals) / max(len(valid_totals), 1)),
        }
        history.append(record)
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(resolved),
            "epoch": epoch,
            "history": history,
        }
        save_checkpoint(last_path, payload)
        if record["valid_loss"] < best_metric:
            best_metric = record["valid_loss"]
            save_checkpoint(best_path, payload)
    return {
        "model": model,
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "best_metric": best_metric,
        "history": history,
    }


__all__ = [
    "ConstantPotentialTrainConfig",
    "constant_potential_loss",
    "train_constant_potential",
]
