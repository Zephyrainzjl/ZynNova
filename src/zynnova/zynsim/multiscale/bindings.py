"""Default atomistic-to-continuum bindings for full battery coupling."""

from __future__ import annotations

from .coupling import PropertyBinding


def full_p2d_bindings() -> tuple[PropertyBinding, ...]:
    return (
        PropertyBinding(
            "negative_solid_diffusivity",
            "negative.solid_diffusivity_m2_s",
            "m2 s-1",
        ),
        PropertyBinding(
            "positive_solid_diffusivity",
            "positive.solid_diffusivity_m2_s",
            "m2 s-1",
        ),
        PropertyBinding(
            "negative_reaction_rate",
            "negative.reaction_rate_m2p5_mol_m0p5_s",
            "m2.5 mol-0.5 s-1",
        ),
        PropertyBinding(
            "positive_reaction_rate",
            "positive.reaction_rate_m2p5_mol_m0p5_s",
            "m2.5 mol-0.5 s-1",
        ),
        PropertyBinding(
            "negative_electronic_conductivity",
            "negative.electronic_conductivity_S_m",
            "S m-1",
        ),
        PropertyBinding(
            "positive_electronic_conductivity",
            "positive.electronic_conductivity_S_m",
            "S m-1",
        ),
        PropertyBinding(
            "electrolyte_diffusivity",
            "electrolyte.diffusivity_m2_s",
            "m2 s-1",
        ),
        PropertyBinding(
            "electrolyte_conductivity",
            "electrolyte.ionic_conductivity_S_m",
            "S m-1",
        ),
        PropertyBinding(
            "electrolyte_transference_number",
            "electrolyte.transference_number",
            "1",
            validator=lambda value: 0.0 < value < 1.0,
        ),
    )


def multiphysics_material_bindings() -> tuple[PropertyBinding, ...]:
    return (
        PropertyBinding("young_modulus", "config.young_modulus_Pa", "Pa"),
        PropertyBinding(
            "thermal_conductivity",
            "config.thermal_conductivity_W_m_K",
            "W m-1 K-1",
        ),
        PropertyBinding(
            "chemical_expansion",
            "config.chemical_expansion_coefficient",
            "1",
        ),
        PropertyBinding(
            "sei_growth_rate",
            "config.sei_rate_m_sqrt_s",
            "m s-0.5",
            required=False,
        ),
    )


__all__ = ["full_p2d_bindings", "multiphysics_material_bindings"]
