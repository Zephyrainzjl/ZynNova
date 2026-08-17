from __future__ import annotations

from collections.abc import Mapping


def _index(names: tuple[str, ...], name: str) -> int | None:
    try:
        return names.index(name)
    except ValueError:
        return None


def physics_consistency_loss(
    normalized_mean,
    batch,
    *,
    normalizer,
    property_names: tuple[str, ...],
    condition_names: tuple[str, ...],
    entropy_weight: float = 0.20,
):
    """Differentiable constraints derived from the paper's energy-storage physics."""

    import torch

    predictions = normalizer.decode_tensor(normalized_mean)
    zero = predictions.sum() * 0.0
    penalties = []

    pm_index = _index(property_names, "maximum_polarization_C_m2")
    pr_index = _index(property_names, "remanent_polarization_C_m2")
    if pm_index is not None and pr_index is not None:
        scale = predictions[:, pm_index].detach().abs().mean().clamp_min(1.0e-3)
        penalties.append(
            torch.relu(predictions[:, pr_index] - predictions[:, pm_index]).mean() / scale
        )

    entropy_index = _index(property_names, "configurational_entropy_R")
    if entropy_index is not None:
        exact_entropy = batch["physics_descriptors"][:, 0]
        penalties.append(
            entropy_weight
            * torch.nn.functional.smooth_l1_loss(
                predictions[:, entropy_index],
                exact_entropy,
            )
        )

    field_index = _index(condition_names, "electric_field_MV_m")
    if field_index is not None:
        observed_field = batch["condition_mask"][:, field_index]
        field = batch["raw_conditions"][:, field_index].abs()
        breakdown_index = _index(property_names, "breakdown_strength_MV_m")
        if breakdown_index is not None and observed_field.any():
            breakdown = predictions[:, breakdown_index]
            scale = breakdown.detach().abs().mean().clamp_min(1.0)
            penalties.append(
                torch.relu(field[observed_field] - breakdown[observed_field]).mean() / scale
            )
        energy_index = _index(property_names, "recoverable_energy_density_J_cm3")
        if (
            energy_index is not None
            and pm_index is not None
            and pr_index is not None
            and observed_field.any()
        ):
            delta_p = torch.relu(predictions[:, pm_index] - predictions[:, pr_index])
            upper_bound = field * delta_p
            energy = predictions[:, energy_index]
            scale = energy.detach().abs().mean().clamp_min(1.0)
            penalties.append(
                torch.relu(energy[observed_field] - upper_bound[observed_field]).mean() / scale
            )
    return sum(penalties, zero)


def high_entropy_report(
    descriptors: Mapping[str, float],
    *,
    threshold_R: float = 1.5,
) -> dict[str, float | bool]:
    entropy = float(descriptors["configurational_entropy_R"])
    bond_types = float(descriptors["bond_type_count"])
    return {
        "configurational_entropy_R": entropy,
        "bond_type_count": bond_types,
        "high_entropy_threshold_R": float(threshold_R),
        "meets_entropy_threshold": entropy >= threshold_R,
        "has_at_least_five_bond_types": bond_types >= 5,
    }


__all__ = ["high_entropy_report", "physics_consistency_loss"]
