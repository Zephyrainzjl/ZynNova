"""Masked multi-observable losses for one conservative energy surface."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from ._deps import require_torch

torch = require_torch()
nn = torch.nn
functional = torch.nn.functional


@dataclass(frozen=True, slots=True)
class LossWeights:
    energy: float = 1.0
    forces: float = 10.0
    stress: float = 1.0
    charges: float = 1.0
    magnetic_moments: float = 0.2
    collinear_spins: float = 0.0
    oxidation_states: float = 0.0
    dipoles: float = 0.1
    quadrupoles: float = 0.05
    charge_multipoles: float = 0.0
    spin_multipoles: float = 0.0
    electronic_stationarity: float = 0.0
    effective_fields: float = 0.0
    magnetic_torques: float = 0.0
    spin_constraint: float = 0.0
    oxidation_charge_consistency: float = 0.0

    def __post_init__(self) -> None:
        if any(getattr(self, item.name) < 0 for item in fields(self)):
            raise ValueError("loss weights must be nonnegative")


def _masked_mse(prediction: Any, target: Any, mask: Any | None = None) -> Any:
    difference = (prediction - target).square()
    if mask is not None:
        expanded = mask
        while expanded.ndim < difference.ndim:
            expanded = expanded.unsqueeze(-1)
        difference = difference * expanded
        denominator = expanded.expand_as(difference).sum()
        if not bool((denominator > 0).detach()):
            raise ValueError("a supervised target mask selects no values")
        return difference.sum() / denominator
    return difference.mean()


def robust_squared(values: Any, delta: float = 1.0) -> Any:
    """Smooth Huber-like stationarity penalty with finite large gradients."""

    if delta <= 0:
        raise ValueError("robust loss delta must be positive")
    scale = values.new_tensor(float(delta))
    return 2.0 * scale * (torch.hypot(values, scale) - scale)


class ZIVARLoss(nn.Module):
    """Loss for datasets where any electronic label may be absent."""

    def __init__(
        self,
        weights: LossWeights | None = None,
        *,
        stationarity_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        if stationarity_delta <= 0:
            raise ValueError("stationarity_delta must be positive")
        self.stationarity_delta = float(stationarity_delta)

    def forward(self, output: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
        if "magmoms" in targets and "magmom_vectors" in targets:
            raise ValueError(
                "scalar and vector magnetic-moment labels cannot supervise the "
                "same production batch"
            )
        terms: dict[str, Any] = {}
        mappings = (
            ("energy", "energy", self.weights.energy),
            ("forces", "forces", self.weights.forces),
            ("stress", "stress", self.weights.stress),
            ("charges", "charges", self.weights.charges),
            ("magmoms", "magmoms", self.weights.magnetic_moments),
            ("magmom_vectors", "magmom_vectors", self.weights.magnetic_moments),
            ("collinear_spins", "collinear_spins", self.weights.collinear_spins),
            ("dipoles", "dipoles", self.weights.dipoles),
            ("quadrupoles", "quadrupoles", self.weights.quadrupoles),
            (
                "charge_multipoles",
                "charge_multipoles",
                self.weights.charge_multipoles,
            ),
            ("spin_multipoles", "spin_multipoles", self.weights.spin_multipoles),
            ("effective_field_T", "effective_field_T", self.weights.effective_fields),
            ("magnetic_torque_eV", "magnetic_torque_eV", self.weights.magnetic_torques),
        )
        magnetic_added = False
        for output_name, target_name, weight in mappings:
            if output_name == "magmom_vectors" and magnetic_added:
                continue
            if weight and target_name in targets and output_name in output:
                mask = targets.get(f"{target_name}_mask")
                terms[target_name] = weight * _masked_mse(
                    output[output_name], targets[target_name], mask
                )
                if output_name in {"magmoms", "magmom_vectors"}:
                    magnetic_added = True
        if (
            self.weights.oxidation_states
            and "oxidation_states" in targets
            and "oxidation_logits" in output
        ):
            values = output["oxidation_state_values"]
            labels = targets["oxidation_states"].to(dtype=torch.long)
            indices = labels - int(values[0].item())
            valid = targets.get("oxidation_states_mask")
            if valid is None:
                valid = torch.ones_like(indices, dtype=torch.bool)
            else:
                valid = valid.to(dtype=torch.bool)
            if bool(valid.any().detach()):
                outside = (indices[valid] < 0) | (indices[valid] >= values.numel())
                if bool(torch.any(outside).detach()):
                    raise ValueError("oxidation-state target is outside the configured range")
                allowed = output.get("oxidation_allowed_mask")
                if allowed is not None and not bool(
                    torch.all(allowed[valid, indices[valid]]).detach()
                ):
                    raise ValueError("oxidation-state target is disallowed for its element")
                terms["oxidation_states"] = self.weights.oxidation_states * (
                    functional.cross_entropy(
                        output["oxidation_logits"][valid], indices[valid]
                    )
                )
        if (
            self.weights.oxidation_charge_consistency
            and "oxidation_expectation" in output
            and "oxidation_total_charge" in targets
        ):
            batch = targets.get("batch")
            if batch is None:
                raise ValueError("oxidation charge consistency requires targets['batch']")
            graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
            summed = output["oxidation_expectation"].new_zeros(graph_count)
            summed.index_add_(0, batch, output["oxidation_expectation"])
            terms["oxidation_charge_consistency"] = (
                self.weights.oxidation_charge_consistency
                * _masked_mse(summed, targets["oxidation_total_charge"])
            )
        if self.weights.electronic_stationarity:
            terms["electronic_stationarity"] = (
                self.weights.electronic_stationarity
                * robust_squared(
                    output["electronic_residual"], self.stationarity_delta
                ).mean()
            )
        if self.weights.spin_constraint and "spin_constraint_residual" in output:
            terms["spin_constraint"] = self.weights.spin_constraint * robust_squared(
                output["spin_constraint_residual"], self.stationarity_delta
            ).mean()
        if not terms:
            raise ValueError("batch contains no supervised loss")
        terms["total"] = sum(terms.values())
        return terms


__all__ = ["LossWeights", "ZIVARLoss", "robust_squared"]
