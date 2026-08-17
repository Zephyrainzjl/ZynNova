from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MechanismPrior:
    """A literature-derived path to test, never a result imposed on the data."""

    prior_id: str
    exposure: str
    mediators: tuple[str, ...]
    outcomes: tuple[str, ...]
    context_features: tuple[str, ...]
    expected_relation: str
    rationale: str
    falsification_tests: tuple[str, ...]
    citations: tuple[str, ...]
    preferred_oracles: tuple[str, ...]


MECHANISM_PRIORS: tuple[MechanismPrior, ...] = (
    MechanismPrior(
        prior_id="high_entropy_superparaelectric_storage",
        exposure="bond_configurational_entropy_R",
        mediators=(
            "barrier_standard_deviation_eV",
            "random_field_strength_MV_m",
            "remanent_polarization_C_m2",
            "polarization_saturation_field_MV_m",
        ),
        outcomes=(
            "recoverable_energy_density_J_cm3",
            "efficiency",
            "breakdown_strength_MV_m",
        ),
        context_features=("crystallinity_fraction", "proton_dose_Mrad"),
        expected_relation=(
            "A possible non-monotonic entropy effect mediated by bond/barrier "
            "heterogeneity, delayed saturation, and reduced remanence."
        ),
        rationale=(
            "Low-dose irradiation can create additional bond types and a "
            "high-entropy superparaelectric state. Entropy alone is not sufficient: "
            "the mediator chain and operating field must be measured."
        ),
        falsification_tests=(
            "Hold bond-type entropy fixed while changing bond identity and compare P-E loops.",
            "Match crystallinity and dose, then test whether barrier heterogeneity mediates Ud.",
            "Verify the trend in a second polymer family and with an unirradiated synthesis route.",
        ),
        citations=(
            "https://doi.org/10.1038/s41563-025-02211-z",
        ),
        preferred_oracles=("jouleweave_neb", "dipole_md", "phase_field", "experiment"),
    ),
    MechanismPrior(
        prior_id="crosslink_flattened_landscape_piezoelectricity",
        exposure="crosslink_density_fraction",
        mediators=(
            "torsional_angle_standard_deviation_deg",
            "torsional_barrier_eV",
            "helix_trans_energy_delta_eV",
        ),
        outcomes=("piezoelectric_d33_pC_N",),
        context_features=("phase_energy_gap_eV", "crystallinity_fraction"),
        expected_relation=(
            "An optimum window rather than a monotonic benefit: local conformational "
            "heterogeneity can flatten the rotation landscape near phase degeneracy."
        ),
        rationale=(
            "Intermolecular crosslinks can facilitate local bond rotation near "
            "crosslinking sites, but excessive crosslinking or the wrong ground-state "
            "phase can remove the benefit."
        ),
        falsification_tests=(
            "Scan torsional barriers at multiple distances from the crosslink.",
            "Repeat at matched crystallinity away from and near phase degeneracy.",
            "Test several crosslinker sizes at the same crosslink density.",
        ),
        citations=(
            "https://doi.org/10.1038/s41467-026-69998-6",
        ),
        preferred_oracles=("jouleweave_neb", "dft_conformer_scan", "experiment"),
    ),
    MechanismPrior(
        prior_id="functional_groups_band_edges_breakdown",
        exposure="functional_group_identity",
        mediators=(
            "homo_eV",
            "lumo_eV",
            "bandgap_eV",
            "electron_affinity_eV",
            "cohesive_energy_density_J_cm3",
        ),
        outcomes=(
            "leakage_current_density_A_m2",
            "breakdown_strength_MV_m",
            "recoverable_energy_density_J_cm3",
        ),
        context_features=(
            "conjugation_fraction",
            "crystallinity_fraction",
            "electric_field_MV_m",
        ),
        expected_relation=(
            "Group effects on HOMO and LUMO can be asymmetric and context-dependent; "
            "a classical interatomic potential cannot supply electronic band labels."
        ),
        rationale=(
            "Matched repeat-unit substitutions plus DFT/HSE or experimentally anchored "
            "electronic labels are required to distinguish a true gap effect from packing."
        ),
        falsification_tests=(
            "Construct matched scaffold pairs with one functional-group substitution.",
            "Calculate both band edges, not only the gap, using the same electronic method.",
            "Condition on packing density and crystallinity before linking gap to breakdown.",
        ),
        citations=(
            "https://doi.org/10.1038/sdata.2016.12",
            "https://doi.org/10.1021/acs.jpcc.8b02913",
        ),
        preferred_oracles=("dft_electronic", "experiment"),
    ),
    MechanismPrior(
        prior_id="packing_mobility_dielectric_tradeoff",
        exposure="cohesive_energy_density_J_cm3",
        mediators=(
            "fractional_free_volume",
            "segmental_relaxation_time_s",
            "dipole_variance_e2_A2",
        ),
        outcomes=(
            "dielectric_constant",
            "dielectric_loss_tangent",
            "glass_transition_temperature_K",
        ),
        context_features=("temperature_K", "log10_frequency_Hz"),
        expected_relation=(
            "Stronger packing may suppress segmental loss while also restricting "
            "orientational polarization; temperature and frequency determine the sign."
        ),
        rationale=(
            "The dielectric response is a dynamical, condition-dependent quantity. "
            "Static repeat-unit descriptors alone cannot identify the relaxation channel."
        ),
        falsification_tests=(
            "Measure or simulate the same polymer over a temperature-frequency grid.",
            "Separate electronic, vibrational, and orientational contributions.",
            "Compare matched structures with similar dipole moment but different packing.",
        ),
        citations=(
            "https://doi.org/10.1038/s41524-022-00906-4",
        ),
        preferred_oracles=("dipole_md", "dielectric_spectroscopy"),
    ),
)


def priors_for_target(target: str) -> tuple[MechanismPrior, ...]:
    return tuple(prior for prior in MECHANISM_PRIORS if target in prior.outcomes)


__all__ = ["MECHANISM_PRIORS", "MechanismPrior", "priors_for_target"]
