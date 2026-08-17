"""Production-oriented constitutive building blocks for battery mechanics.

These local integration kernels are solver-agnostic and can be used by the
existing Tet4 code, external FEM packages, or PETSc-based applications.  They
cover mechanisms absent from the original compact elastic-damage closure:
viscoelastic binder response, J2 plasticity, cohesive debonding/contact, and a
multi-reaction thermal-runaway source model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..constants import GAS_CONSTANT


@dataclass(frozen=True, slots=True)
class MaxwellBranch:
    modulus_Pa: float
    relaxation_time_s: float

    def __post_init__(self) -> None:
        if self.modulus_Pa <= 0.0 or self.relaxation_time_s <= 0.0:
            raise ValueError("Maxwell branch values must be positive")


@dataclass(slots=True)
class GeneralizedMaxwellState:
    viscous_strain: np.ndarray


class GeneralizedMaxwellModel:
    """Small-strain isotropic generalized Maxwell model in 6-vector notation."""

    def __init__(
        self,
        equilibrium_modulus_Pa: float,
        poisson_ratio: float,
        branches: Sequence[MaxwellBranch],
    ) -> None:
        if equilibrium_modulus_Pa <= 0.0 or not -1.0 < poisson_ratio < 0.5:
            raise ValueError("invalid equilibrium elastic constants")
        self.equilibrium_modulus_Pa = float(equilibrium_modulus_Pa)
        self.poisson_ratio = float(poisson_ratio)
        self.branches = tuple(branches)
        if not self.branches:
            raise ValueError("at least one Maxwell branch is required")

    def initialize(self, shape: tuple[int, ...] = ()) -> GeneralizedMaxwellState:
        return GeneralizedMaxwellState(np.zeros(shape + (len(self.branches), 6), dtype=float))

    def step(
        self,
        total_strain: np.ndarray,
        state: GeneralizedMaxwellState,
        dt_s: float,
    ) -> tuple[np.ndarray, GeneralizedMaxwellState, np.ndarray]:
        strain = np.asarray(total_strain, dtype=float)
        if strain.shape[-1] != 6 or dt_s <= 0.0:
            raise ValueError("strain must end in six components and dt must be positive")
        expected = strain.shape[:-1] + (len(self.branches), 6)
        if state.viscous_strain.shape != expected:
            raise ValueError("viscoelastic state shape mismatch")
        stress = _isotropic_stress(strain, self.equilibrium_modulus_Pa, self.poisson_ratio)
        updated = np.empty_like(state.viscous_strain)
        dissipation = np.zeros(strain.shape[:-1], dtype=float)
        for index, branch in enumerate(self.branches):
            decay = np.exp(-dt_s / branch.relaxation_time_s)
            old = state.viscous_strain[..., index, :]
            new = decay * old + (1.0 - decay) * strain
            branch_stress = _isotropic_stress(strain - new, branch.modulus_Pa, self.poisson_ratio)
            stress += branch_stress
            rate = (new - old) / dt_s
            dissipation += np.maximum(_double_contract(branch_stress, rate), 0.0)
            updated[..., index, :] = new
        return stress, GeneralizedMaxwellState(updated), dissipation


@dataclass(frozen=True, slots=True)
class J2PlasticityParameters:
    young_modulus_Pa: float
    poisson_ratio: float
    initial_yield_stress_Pa: float
    isotropic_hardening_Pa: float = 0.0

    def __post_init__(self) -> None:
        if min(self.young_modulus_Pa, self.initial_yield_stress_Pa) <= 0.0:
            raise ValueError("elastic modulus and yield stress must be positive")
        if not -1.0 < self.poisson_ratio < 0.5 or self.isotropic_hardening_Pa < 0.0:
            raise ValueError("invalid J2 plasticity parameters")


@dataclass(slots=True)
class J2PlasticityState:
    plastic_strain: np.ndarray
    equivalent_plastic_strain: np.ndarray


class J2PlasticityModel:
    """Backward-Euler radial return for von-Mises plasticity."""

    def __init__(self, parameters: J2PlasticityParameters) -> None:
        self.parameters = parameters

    def initialize(self, shape: tuple[int, ...] = ()) -> J2PlasticityState:
        return J2PlasticityState(
            plastic_strain=np.zeros(shape + (6,), dtype=float),
            equivalent_plastic_strain=np.zeros(shape, dtype=float),
        )

    def step(self, total_strain: np.ndarray, state: J2PlasticityState) -> tuple[np.ndarray, J2PlasticityState, np.ndarray]:
        strain = np.asarray(total_strain, dtype=float)
        if strain.shape[-1] != 6 or state.plastic_strain.shape != strain.shape:
            raise ValueError("J2 state/strain shape mismatch")
        p = self.parameters
        trial = _isotropic_stress(strain - state.plastic_strain, p.young_modulus_Pa, p.poisson_ratio)
        deviator = _deviator(trial)
        equivalent = np.sqrt(1.5 * _double_contract(deviator, deviator))
        yield_stress = p.initial_yield_stress_Pa + p.isotropic_hardening_Pa * state.equivalent_plastic_strain
        shear = p.young_modulus_Pa / (2.0 * (1.0 + p.poisson_ratio))
        increment = np.maximum(equivalent - yield_stress, 0.0) / max(3.0 * shear + p.isotropic_hardening_Pa, 1.0e-30)
        direction = 1.5 * deviator / np.maximum(equivalent[..., None], 1.0e-30)
        plastic_strain = state.plastic_strain + increment[..., None] * direction
        eq_plastic = state.equivalent_plastic_strain + increment
        stress = _isotropic_stress(strain - plastic_strain, p.young_modulus_Pa, p.poisson_ratio)
        plastic_work = increment * (yield_stress + 0.5 * p.isotropic_hardening_Pa * increment)
        return stress, J2PlasticityState(plastic_strain, eq_plastic), plastic_work


@dataclass(frozen=True, slots=True)
class CohesiveZoneParameters:
    normal_stiffness_Pa_m: float = 1.0e15
    tangential_stiffness_Pa_m: float = 5.0e14
    peak_normal_traction_Pa: float = 100.0e6
    peak_shear_traction_Pa: float = 80.0e6
    mode_I_energy_J_m2: float = 5.0
    mode_II_energy_J_m2: float = 10.0
    friction_coefficient: float = 0.2
    penalty_contact_Pa_m: float = 2.0e15

    def __post_init__(self) -> None:
        if min(
            self.normal_stiffness_Pa_m,
            self.tangential_stiffness_Pa_m,
            self.peak_normal_traction_Pa,
            self.peak_shear_traction_Pa,
            self.mode_I_energy_J_m2,
            self.mode_II_energy_J_m2,
            self.penalty_contact_Pa_m,
        ) <= 0.0:
            raise ValueError("cohesive parameters must be positive")
        if self.friction_coefficient < 0.0:
            raise ValueError("friction coefficient cannot be negative")


@dataclass(slots=True)
class CohesiveZoneState:
    damage: np.ndarray
    maximum_effective_separation: np.ndarray


class MixedModeCohesiveZone:
    """Benzeggagh-Kenane-like irreversible cohesive/contact law."""

    def __init__(self, parameters: CohesiveZoneParameters | None = None) -> None:
        self.parameters = parameters or CohesiveZoneParameters()

    def initialize(self, shape: tuple[int, ...] = ()) -> CohesiveZoneState:
        return CohesiveZoneState(np.zeros(shape), np.zeros(shape))

    def response(
        self,
        separation_m: np.ndarray,
        state: CohesiveZoneState,
    ) -> tuple[np.ndarray, CohesiveZoneState, np.ndarray]:
        jump = np.asarray(separation_m, dtype=float)
        if jump.shape[-1] != 3 or state.damage.shape != jump.shape[:-1]:
            raise ValueError("cohesive separation/state shape mismatch")
        p = self.parameters
        opening = np.maximum(jump[..., 0], 0.0)
        closing = np.minimum(jump[..., 0], 0.0)
        shear = np.linalg.norm(jump[..., 1:], axis=-1)
        onset_n = p.peak_normal_traction_Pa / p.normal_stiffness_Pa_m
        onset_s = p.peak_shear_traction_Pa / p.tangential_stiffness_Pa_m
        effective = np.sqrt((opening / onset_n) ** 2 + (shear / onset_s) ** 2)
        maximum = np.maximum(state.maximum_effective_separation, effective)
        mix = shear * shear / np.maximum(opening * opening + shear * shear, 1.0e-30)
        critical_energy = p.mode_I_energy_J_m2 + (p.mode_II_energy_J_m2 - p.mode_I_energy_J_m2) * mix**1.5
        peak = np.sqrt(p.peak_normal_traction_Pa**2 + p.peak_shear_traction_Pa**2)
        final = np.maximum(2.0 * critical_energy / np.maximum(peak * np.maximum(onset_n, onset_s), 1.0e-30), 1.0 + 1.0e-8)
        damage_trial = np.clip((maximum - 1.0) / (final - 1.0), 0.0, 1.0)
        damage = np.maximum(state.damage, damage_trial)
        traction = np.zeros_like(jump)
        traction[..., 0] = (1.0 - damage) * p.normal_stiffness_Pa_m * opening + p.penalty_contact_Pa_m * closing
        tangential_trial = (1.0 - damage)[..., None] * p.tangential_stiffness_Pa_m * jump[..., 1:]
        normal_compression = np.maximum(-p.penalty_contact_Pa_m * closing, 0.0)
        friction_limit = p.friction_coefficient * normal_compression
        magnitude = np.linalg.norm(tangential_trial, axis=-1)
        scale = np.minimum(1.0, friction_limit / np.maximum(magnitude, 1.0e-30))
        # In opening, retain cohesive shear; in compression, enforce Coulomb cap.
        scale = np.where(closing < 0.0, scale, 1.0)
        traction[..., 1:] = tangential_trial * scale[..., None]
        dissipation = np.maximum(damage - state.damage, 0.0) * critical_energy
        return traction, CohesiveZoneState(damage, maximum), dissipation


@dataclass(frozen=True, slots=True)
class ThermalReaction:
    name: str
    pre_exponential_s: float
    activation_energy_J_mol: float
    enthalpy_J_kg: float
    onset_temperature_K: float
    order: float = 1.0

    def __post_init__(self) -> None:
        if min(self.pre_exponential_s, self.activation_energy_J_mol, self.enthalpy_J_kg, self.onset_temperature_K, self.order) <= 0.0:
            raise ValueError("thermal reaction parameters must be positive")


@dataclass(slots=True)
class ThermalRunawayState:
    conversion: np.ndarray


class ThermalRunawayKinetics:
    """Parallel Arrhenius reaction network with reactant depletion."""

    def __init__(self, reactions: Sequence[ThermalReaction]) -> None:
        self.reactions = tuple(reactions)
        if not self.reactions:
            raise ValueError("at least one thermal reaction is required")

    def initialize(self, shape: tuple[int, ...] = ()) -> ThermalRunawayState:
        return ThermalRunawayState(np.zeros(shape + (len(self.reactions),), dtype=float))

    def step(
        self,
        temperature_K: np.ndarray,
        state: ThermalRunawayState,
        dt_s: float,
        *,
        density_kg_m3: float,
    ) -> tuple[ThermalRunawayState, np.ndarray]:
        temperature = np.asarray(temperature_K, dtype=float)
        if state.conversion.shape != temperature.shape + (len(self.reactions),) or dt_s <= 0.0 or density_kg_m3 <= 0.0:
            raise ValueError("thermal-runaway state/input mismatch")
        updated = state.conversion.copy()
        heat = np.zeros_like(temperature)
        for index, reaction in enumerate(self.reactions):
            alpha = state.conversion[..., index]
            gate = 0.5 * (1.0 + np.tanh((temperature - reaction.onset_temperature_K) / 3.0))
            rate = gate * reaction.pre_exponential_s * np.exp(-reaction.activation_energy_J_mol / (GAS_CONSTANT * temperature)) * np.power(np.maximum(1.0 - alpha, 0.0), reaction.order)
            increment = np.minimum(dt_s * rate, 1.0 - alpha)
            updated[..., index] = alpha + increment
            heat += density_kg_m3 * reaction.enthalpy_J_kg * increment / dt_s
        return ThermalRunawayState(updated), heat


def _isotropic_stress(strain: np.ndarray, young: float, poisson: float) -> np.ndarray:
    shear = young / (2.0 * (1.0 + poisson))
    lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    result = np.empty_like(strain)
    trace = strain[..., 0] + strain[..., 1] + strain[..., 2]
    result[..., :3] = 2.0 * shear * strain[..., :3] + lame * trace[..., None]
    result[..., 3:] = shear * strain[..., 3:]
    return result


def _deviator(stress: np.ndarray) -> np.ndarray:
    result = stress.copy()
    mean = np.mean(stress[..., :3], axis=-1)
    result[..., :3] -= mean[..., None]
    return result


def _double_contract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a[..., :3] * b[..., :3], axis=-1) + 2.0 * np.sum(a[..., 3:] * b[..., 3:], axis=-1)


__all__ = [
    "CohesiveZoneParameters",
    "CohesiveZoneState",
    "GeneralizedMaxwellModel",
    "GeneralizedMaxwellState",
    "J2PlasticityModel",
    "J2PlasticityParameters",
    "J2PlasticityState",
    "MaxwellBranch",
    "MixedModeCohesiveZone",
    "ThermalReaction",
    "ThermalRunawayKinetics",
    "ThermalRunawayState",
]
