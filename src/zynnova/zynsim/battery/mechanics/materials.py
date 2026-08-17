"""Concentration-, temperature-, and orientation-dependent cathode laws."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

import numpy as np

from ...constants import GAS_CONSTANT


class PropertyProvider(Protocol):
    def __call__(
        self,
        name: str,
        *,
        soc: float,
        temperature_K: float,
        metadata: Mapping[str, object] | None = None,
    ) -> float | np.ndarray: ...


@dataclass(frozen=True, slots=True)
class NMCCathodeMaterial:
    """Physics parameters for layered Ni-rich cathodes.

    Defaults are literature-scale starting values, not a claim of calibration
    for a particular commercial powder.  Every state-dependent law can be
    replaced by a ZynForge/JouleWeave/DFT-backed ``property_provider``.
    """

    maximum_lithium_concentration_mol_m3: float = 49_200.0
    reference_temperature_K: float = 298.15
    regular_solution_parameter_J_mol: float = 4_500.0
    gradient_energy_J_m: float = 2.0e-10
    diffusivity_reference_m2_s: float = 2.0e-14
    diffusivity_activation_energy_J_mol: float = 28_000.0
    basal_to_c_axis_diffusivity_ratio: float = 1.0e-3
    diffusivity_high_soc_drop_decades: float = 2.0
    diffusivity_drop_center: float = 0.80
    diffusivity_drop_width: float = 0.08
    young_modulus_lithiated_Pa: float = 150.0e9
    young_modulus_delithiated_Pa: float = 105.0e9
    poisson_ratio: float = 0.30
    c_axis_modulus_ratio: float = 0.72
    basal_c_axis_shear_ratio: float = 0.82
    transverse_normal_coupling: float = 0.22
    residual_stiffness: float = 1.0e-6
    fracture_energy_J_m2: float = 5.0
    fracture_length_m: float = 1.0e-7
    damage_viscosity_Pa_s: float = 5.0e5
    grain_boundary_fracture_reduction: float = 0.55
    first_delithiation_fracture_reduction: float = 0.35
    fatigue_energy_scale_J_m3: float = 2.0e7
    fatigue_exponent: float = 1.8
    minimum_fatigue_fracture_fraction: float = 0.15
    critical_resolved_shear_Pa: float = 0.8e9
    dislocation_reference_time_s: float = 30.0
    dislocation_rate_exponent: float = 2.0
    oxygen_deficiency_shear_threshold: float = 0.12
    oxygen_deficiency_transition_width: float = 0.02
    high_voltage_transition_center_theta: float = 0.18
    high_voltage_transition_width: float = 0.035
    high_voltage_transition_time_s: float = 120.0
    transition_basal_mismatch_strain: float = -0.003
    transition_c_axis_mismatch_strain: float = -0.018
    oxygen_generation_per_transition: float = 0.25
    oxygen_migration_diffusivity_m2_s: float = 2.0e-16
    oxygen_trapping_time_s: float = 120.0
    oxygen_crack_trapping_gain: float = 5.0
    oxygen_toughness_reduction: float = 0.45
    oxygen_transport_penalty: float = 0.60
    crack_wetting_factor: float = 8.0
    crack_wetting_time_s: float = 20.0
    crack_wetting_diffusivity_m2_s: float = 5.0e-15
    crack_transport_factor: float = 4.0
    damage_transport_exponent: float = 2.0
    property_provider: PropertyProvider | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        positive = (
            self.maximum_lithium_concentration_mol_m3,
            self.reference_temperature_K,
            self.gradient_energy_J_m,
            self.diffusivity_reference_m2_s,
            self.young_modulus_lithiated_Pa,
            self.young_modulus_delithiated_Pa,
            self.fracture_energy_J_m2,
            self.fracture_length_m,
            self.damage_viscosity_Pa_s,
            self.fatigue_energy_scale_J_m3,
            self.critical_resolved_shear_Pa,
            self.dislocation_reference_time_s,
            self.high_voltage_transition_width,
            self.high_voltage_transition_time_s,
            self.oxygen_migration_diffusivity_m2_s,
            self.oxygen_trapping_time_s,
            self.crack_wetting_time_s,
            self.crack_wetting_diffusivity_m2_s,
        )
        if min(positive) <= 0.0:
            raise ValueError("cathode material positive parameters must be positive")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("Poisson ratio must lie in (-1, 0.5)")
        if min(self.c_axis_modulus_ratio, self.basal_c_axis_shear_ratio) <= 0.0:
            raise ValueError("anisotropic stiffness ratios must be positive")
        if not 0.0 <= self.transverse_normal_coupling < 1.0:
            raise ValueError("transverse_normal_coupling must lie in [0,1)")
        if not 0.0 <= self.grain_boundary_fracture_reduction < 1.0:
            raise ValueError("grain-boundary reduction must lie in [0,1)")
        if not 0.0 <= self.first_delithiation_fracture_reduction < 1.0:
            raise ValueError("first-delithiation reduction must lie in [0,1)")
        if not 0.0 < self.minimum_fatigue_fracture_fraction <= 1.0:
            raise ValueError("minimum fatigue fracture fraction must lie in (0,1]")
        if not 0.0 < self.high_voltage_transition_center_theta < 1.0:
            raise ValueError("high_voltage_transition_center_theta must lie in (0,1)")
        if min(
            self.oxygen_generation_per_transition,
            self.oxygen_crack_trapping_gain,
            self.oxygen_toughness_reduction,
            self.oxygen_transport_penalty,
        ) < 0.0:
            raise ValueError("oxygen/transition coupling factors cannot be negative")
        if self.oxygen_toughness_reduction >= 1.0 or self.oxygen_transport_penalty >= 1.0:
            raise ValueError("oxygen reductions must be smaller than one")

    def diffusivity(self, theta: np.ndarray, temperature_K: float) -> np.ndarray:
        supplied = self._provided("solid_diffusivity_m2_s", theta, temperature_K)
        if supplied is not None:
            return np.broadcast_to(np.asarray(supplied, dtype=float), np.shape(theta)).copy()
        occupancy = np.clip(np.asarray(theta, dtype=float), 0.0, 1.0)
        transition = 0.5 * (
            1.0 + np.tanh((occupancy - self.diffusivity_drop_center) / self.diffusivity_drop_width)
        )
        concentration_factor = np.power(10.0, -self.diffusivity_high_soc_drop_decades * transition)
        arrhenius = np.exp(
            -self.diffusivity_activation_energy_J_mol / GAS_CONSTANT
            * (1.0 / float(temperature_K) - 1.0 / self.reference_temperature_K)
        )
        return self.diffusivity_reference_m2_s * concentration_factor * arrhenius

    def young_modulus(self, theta: np.ndarray, temperature_K: float) -> np.ndarray:
        supplied = self._provided("young_modulus_Pa", theta, temperature_K)
        if supplied is not None:
            return np.broadcast_to(np.asarray(supplied, dtype=float), np.shape(theta)).copy()
        occupancy = np.clip(np.asarray(theta, dtype=float), 0.0, 1.0)
        return (
            self.young_modulus_delithiated_Pa
            + occupancy * (self.young_modulus_lithiated_Pa - self.young_modulus_delithiated_Pa)
        )

    def transversely_isotropic_stress(
        self,
        elastic_strain: np.ndarray,
        c_axes: np.ndarray,
        theta: np.ndarray,
        temperature_K: float,
        degradation: np.ndarray,
    ) -> np.ndarray:
        """Return a positive-definite transverse-isotropic Cauchy stress.

        With ``n`` the crystallographic c-axis and ``P=I-n⊗n``, the elastic
        energy density is

        ``1/2 Kb tr(Pe)^2 + Kbc tr(Pe)(n·e·n) + 1/2 Kc(n·e·n)^2
        + Gb dev(PeP):dev(PeP) + 2 Gbc |P e n|^2``.

        This captures basal/c-axis stiffness contrast without constructing a
        dense six-by-six tensor at every voxel.  The coupling is bounded so the
        local energy remains positive definite.
        """

        strain = np.asarray(elastic_strain, dtype=float)
        n = np.asarray(c_axes, dtype=float)
        identity = np.eye(3)
        nn = n[..., :, None] * n[..., None, :]
        projector = identity - nn
        basal_modulus = self.young_modulus(theta, temperature_K) * degradation
        c_modulus = basal_modulus * self.c_axis_modulus_ratio
        basal_bulk = basal_modulus / max(2.0 * (1.0 - self.poisson_ratio), 1e-8)
        basal_shear = basal_modulus / (2.0 * (1.0 + self.poisson_ratio))
        cross_shear = basal_shear * self.basal_c_axis_shear_ratio
        coupling = self.transverse_normal_coupling * np.sqrt(basal_bulk * c_modulus)
        eps_nn = np.einsum("...i,...ij,...j->...", n, strain, n, optimize=True)
        eps_pp = np.sum(projector * strain, axis=(-2, -1))
        projected = np.einsum("...ik,...kl,...lj->...ij", projector, strain, projector, optimize=True)
        dev_basal = projected - 0.5 * eps_pp[..., None, None] * projector
        shear_vector = np.einsum("...ij,...jk,...k->...i", projector, strain, n, optimize=True)
        shear_tensor = n[..., :, None] * shear_vector[..., None, :] + shear_vector[..., :, None] * n[..., None, :]
        stress = (
            (basal_bulk * eps_pp + coupling * eps_nn)[..., None, None] * projector
            + (coupling * eps_pp + c_modulus * eps_nn)[..., None, None] * nn
            + 2.0 * basal_shear[..., None, None] * dev_basal
            + 2.0 * cross_shear[..., None, None] * shear_tensor
        )
        return np.asarray(stress, dtype=float)

    def lattice_strains(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Basal and c-axis compositional strains relative to theta=0.

        The c-axis interpolation captures the measured non-monotonic expansion
        and collapse of Ni-rich NMC; users should replace it with operando-XRD
        data for their chemistry.
        """

        occupancy = np.clip(np.asarray(theta, dtype=float), 0.0, 1.0)
        basal = 0.020 * occupancy
        control_theta = np.asarray([0.0, 0.15, 0.37, 0.70, 1.0])
        control_c = np.asarray([0.0, 0.040, 0.035, 0.024, 0.0195])
        c_axis = np.interp(occupancy, control_theta, control_c)
        return basal, c_axis

    def lattice_strain_derivatives(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        occupancy = np.clip(np.asarray(theta, dtype=float), 0.0, 1.0)
        basal = np.full_like(occupancy, 0.020)
        control_theta = np.asarray([0.0, 0.15, 0.37, 0.70, 1.0])
        control_c = np.asarray([0.0, 0.040, 0.035, 0.024, 0.0195])
        slopes = np.diff(control_c) / np.diff(control_theta)
        indices = np.clip(np.searchsorted(control_theta[1:], occupancy, side="right"), 0, len(slopes) - 1)
        return basal, slopes[indices]

    def high_voltage_transition_target(
        self,
        theta: np.ndarray,
        temperature_K: float,
    ) -> np.ndarray:
        """Return a smooth transformed-phase target for deeply delithiated material.

        The field is a deliberately calibratable reduced description of the
        O3→O1/cation-mixed family of high-voltage transformations reported for
        layered oxides.  It is not a universal phase diagram.  A property
        provider can replace the default law with chemistry-specific data.
        """

        supplied = self._provided("high_voltage_transformed_fraction", theta, temperature_K)
        if supplied is not None:
            return np.clip(
                np.broadcast_to(np.asarray(supplied, dtype=float), np.shape(theta)),
                0.0,
                1.0,
            ).copy()
        occupancy = np.clip(np.asarray(theta, dtype=float), 0.0, 1.0)
        return 0.5 * (
            1.0
            + np.tanh(
                (self.high_voltage_transition_center_theta - occupancy)
                / self.high_voltage_transition_width
            )
        )

    def transition_mismatch_strains(
        self,
        transformed_fraction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        fraction = np.clip(np.asarray(transformed_fraction, dtype=float), 0.0, 1.0)
        return (
            self.transition_basal_mismatch_strain * fraction,
            self.transition_c_axis_mismatch_strain * fraction,
        )

    def chemical_potential_J_mol(
        self,
        theta: np.ndarray,
        temperature_K: float,
    ) -> np.ndarray:
        occupancy = np.clip(np.asarray(theta, dtype=float), 1e-8, 1.0 - 1e-8)
        return (
            GAS_CONSTANT * float(temperature_K) * np.log(occupancy / (1.0 - occupancy))
            + self.regular_solution_parameter_J_mol * (1.0 - 2.0 * occupancy)
        )

    def fracture_energy_field(
        self,
        *,
        grain_boundary_indicator: np.ndarray,
        fatigue: np.ndarray,
        minimum_theta_history: np.ndarray,
        oxygen_exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        gb = np.clip(np.asarray(grain_boundary_indicator, dtype=float), 0.0, 1.0)
        fatigue_values = np.maximum(np.asarray(fatigue, dtype=float), 0.0)
        min_theta = np.clip(np.asarray(minimum_theta_history, dtype=float), 0.0, 1.0)
        first_delithiation = 1.0 - self.first_delithiation_fracture_reduction * (1.0 - min_theta)
        fatigue_fraction = 1.0 / (
            1.0 + np.power(fatigue_values / self.fatigue_energy_scale_J_m3, self.fatigue_exponent)
        )
        fatigue_fraction = np.maximum(fatigue_fraction, self.minimum_fatigue_fracture_fraction)
        oxygen = (
            np.zeros_like(gb)
            if oxygen_exposure is None
            else np.clip(np.asarray(oxygen_exposure, dtype=float), 0.0, 1.0)
        )
        return (
            self.fracture_energy_J_m2
            * (1.0 - self.grain_boundary_fracture_reduction * gb)
            * first_delithiation
            * fatigue_fraction
            * (1.0 - self.oxygen_toughness_reduction * oxygen)
        )

    def _provided(
        self,
        name: str,
        theta: np.ndarray,
        temperature_K: float,
    ) -> float | np.ndarray | None:
        if self.property_provider is None:
            return None
        return self.property_provider(
            name,
            soc=float(np.mean(theta)),
            temperature_K=float(temperature_K),
            metadata={"field_shape": tuple(np.shape(theta)), "chemistry": "NMC"},
        )


__all__ = ["NMCCathodeMaterial", "PropertyProvider"]
