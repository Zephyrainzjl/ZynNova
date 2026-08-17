"""Cathode particle-to-electrode-to-cell mechanical degradation coupling.

The module maps high-resolution chemo-mechanical RVE fields onto effective
continuum properties without hiding the scale transition.  It is deliberately
stateful: an RVE can be advanced sparsely on a slower clock than the P2D model,
while its connectivity, crack area, fatigue, and stress statistics feed back to
transport, reaction area, electronic conduction, and available capacity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..properties import evaluate_property
from ..p2d.parameters import P2DParameters
from .spectral import (
    CathodeDegradationState,
    CathodeStepDiagnostics,
    SpectralCathodeDegradationSolver,
)


@dataclass(frozen=True, slots=True)
class StackPressureContactModel:
    """Smooth force-chain closure for porous composite cathodes.

    The coordination number increases and saturates with stack pressure.  A
    Hertz-like concentration factor ``(Z_ref/Z)^(1/3)`` amplifies local tensile
    and shear driving forces when only sparse particle contacts carry load.
    This closure is a reduced constitutive law and must be calibrated against
    tomography/pressure data for a specific electrode.
    """

    reference_pressure_Pa: float = 3.0e6
    minimum_coordination: float = 2.2
    maximum_coordination: float = 6.0
    pressure_exponent: float = 0.65
    reference_coordination: float = 6.0
    binder_relaxation_fraction: float = 0.25
    optimum_pressure_Pa: float = 1.25e6
    porosity_compression_at_reference: float = 0.08
    high_pressure_plating_gain: float = 1.5

    def __post_init__(self) -> None:
        if min(self.reference_pressure_Pa, self.minimum_coordination, self.maximum_coordination) <= 0.0:
            raise ValueError("pressure/contact parameters must be positive")
        if self.minimum_coordination > self.maximum_coordination:
            raise ValueError("minimum coordination cannot exceed maximum coordination")
        if self.pressure_exponent <= 0.0:
            raise ValueError("pressure_exponent must be positive")
        if not 0.0 <= self.binder_relaxation_fraction < 1.0:
            raise ValueError("binder_relaxation_fraction must lie in [0,1)")
        if self.optimum_pressure_Pa < 0.0 or self.porosity_compression_at_reference < 0.0:
            raise ValueError("pressure optimum/compression parameters cannot be negative")
        if self.high_pressure_plating_gain < 0.0:
            raise ValueError("high_pressure_plating_gain cannot be negative")

    def coordination(self, pressure_Pa: float) -> float:
        pressure = max(float(pressure_Pa), 0.0)
        scaled = np.power(pressure / self.reference_pressure_Pa, self.pressure_exponent)
        fraction = scaled / (1.0 + scaled)
        return float(
            self.minimum_coordination
            + (self.maximum_coordination - self.minimum_coordination) * fraction
        )

    def local_stress_factor(self, pressure_Pa: float) -> float:
        coordination = max(self.coordination(pressure_Pa), 1.0e-12)
        factor = np.power(self.reference_coordination / coordination, 1.0 / 3.0)
        return float((1.0 - self.binder_relaxation_fraction) * factor + self.binder_relaxation_fraction)

    def porosity_multiplier(self, pressure_Pa: float) -> float:
        pressure = max(float(pressure_Pa), 0.0)
        compression = self.porosity_compression_at_reference * pressure / max(
            pressure + self.reference_pressure_Pa, 1.0e-30
        )
        return float(np.clip(1.0 - compression, 0.5, 1.0))

    def plating_risk_multiplier(self, pressure_Pa: float) -> float:
        pressure = max(float(pressure_Pa), 0.0)
        excess = max(pressure - self.optimum_pressure_Pa, 0.0) / max(
            self.reference_pressure_Pa, 1.0e-30
        )
        return float(1.0 + self.high_pressure_plating_gain * excess)


@dataclass(frozen=True, slots=True)
class CathodeScaleFeedback:
    time_s: float
    mean_theta: float
    damage_fraction: float
    crack_surface_density_m_inv: float
    connected_active_fraction: float
    oxygen_deficient_fraction: float
    maximum_principal_stress_Pa: float
    contact_stress_factor: float
    active_material_multiplier: float
    diffusivity_multiplier: float
    reaction_area_multiplier: float
    electronic_conductivity_multiplier: float
    capacity_multiplier: float
    porosity_multiplier: float
    electrolyte_transport_multiplier: float
    plating_risk_multiplier: float
    metadata: Mapping[str, object] = field(default_factory=dict)
    transformed_phase_fraction: float = 0.0
    trapped_oxygen_fraction: float = 0.0


@dataclass(frozen=True, slots=True)
class CathodeScaleConfig:
    stack_pressure_Pa: float = 3.0e6
    mechanics_interval_s: float = 10.0
    minimum_property_multiplier: float = 1.0e-4
    crack_wetting_reaction_gain: float = 0.15
    disconnected_capacity_exponent: float = 1.0
    diffusivity_damage_exponent: float = 2.0
    conductivity_percolation_exponent: float = 2.0
    oxygen_deficiency_transport_penalty: float = 0.70
    pressure_model: StackPressureContactModel = field(default_factory=StackPressureContactModel)
    parallel_workers: int = 0

    def __post_init__(self) -> None:
        if self.stack_pressure_Pa < 0.0 or self.mechanics_interval_s <= 0.0:
            raise ValueError("stack pressure must be non-negative and interval positive")
        if not 0.0 < self.minimum_property_multiplier <= 1.0:
            raise ValueError("minimum_property_multiplier must lie in (0,1]")
        if self.parallel_workers < 0:
            raise ValueError("parallel_workers cannot be negative")
        if min(
            self.crack_wetting_reaction_gain,
            self.disconnected_capacity_exponent,
            self.diffusivity_damage_exponent,
            self.conductivity_percolation_exponent,
            self.oxygen_deficiency_transport_penalty,
        ) < 0.0:
            raise ValueError("feedback exponents/penalties cannot be negative")


class CathodeMechanicalMultiscaleModel:
    """Advance one or more cathode RVEs and close them into cell parameters."""

    def __init__(
        self,
        solvers: SpectralCathodeDegradationSolver | Sequence[SpectralCathodeDegradationSolver],
        *,
        config: CathodeScaleConfig | None = None,
        weights: Sequence[float] | None = None,
    ) -> None:
        self.solvers = (solvers,) if isinstance(solvers, SpectralCathodeDegradationSolver) else tuple(solvers)
        if not self.solvers:
            raise ValueError("at least one cathode RVE solver is required")
        self.config = config or CathodeScaleConfig()
        raw_weights = np.ones(len(self.solvers), dtype=float) if weights is None else np.asarray(weights, dtype=float)
        if raw_weights.shape != (len(self.solvers),) or np.any(raw_weights < 0.0) or raw_weights.sum() <= 0.0:
            raise ValueError("weights must be non-negative with one value per solver")
        self.weights = raw_weights / raw_weights.sum()
        self.states: list[CathodeDegradationState] = [solver.initialize() for solver in self.solvers]
        self.last_update_time_s = 0.0
        self.history: list[CathodeScaleFeedback] = []
        self._parameter_baselines: dict[int, dict[str, Any]] = {}

    def synchronize_theta(self, mean_theta: float) -> None:
        """Shift every RVE occupancy to a supplied electrode mean conservatively."""

        target = float(np.clip(mean_theta, 1.0e-7, 1.0 - 1.0e-7))
        for solver, state in zip(self.solvers, self.states, strict=True):
            mask = solver.active_mask
            current = float(np.mean(state.theta[mask]))
            shifted = np.clip(state.theta + (target - current), 1.0e-7, 1.0 - 1.0e-7)
            state.theta = np.where(mask, shifted, 0.0)

    def advance(
        self,
        *,
        time_s: float,
        mean_theta: float,
        mean_c_rate: float,
        temperature_K: float,
        force: bool = False,
    ) -> CathodeScaleFeedback:
        target_time = float(time_s)
        if target_time < self.last_update_time_s:
            raise ValueError("mechanics time cannot move backwards")
        if self.history and not force and target_time - self.last_update_time_s < self.config.mechanics_interval_s:
            return self.history[-1]
        elapsed = max(target_time - self.last_update_time_s, 1.0e-12)
        self.synchronize_theta(mean_theta)
        diagnostics: list[CathodeStepDiagnostics] = []
        contact = self.config.pressure_model.local_stress_factor(self.config.stack_pressure_Pa)

        def run_one(item: tuple[int, SpectralCathodeDegradationSolver, CathodeDegradationState]):
            index, solver, state = item
            updated, diagnostic = solver.step(
                state,
                mean_c_rate=float(mean_c_rate),
                temperature_K=float(temperature_K),
                dt_s=elapsed,
                mechanical_load_factor=contact,
            )
            return index, updated, diagnostic

        items = [(index, solver, state) for index, (solver, state) in enumerate(zip(self.solvers, self.states, strict=True))]
        workers = self.config.parallel_workers
        if workers > 1 and len(items) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(workers, len(items)), thread_name_prefix="zynsim-rve") as pool:
                completed = list(pool.map(run_one, items))
        else:
            completed = [run_one(item) for item in items]
        completed.sort(key=lambda value: value[0])
        for index, updated, diagnostic in completed:
            self.states[index] = updated
            diagnostics.append(diagnostic)
        feedback = self._aggregate(diagnostics, contact)
        self.last_update_time_s = target_time
        self.history.append(feedback)
        return feedback

    def _aggregate(
        self,
        diagnostics: Sequence[CathodeStepDiagnostics],
        contact_factor: float,
    ) -> CathodeScaleFeedback:
        def weighted(name: str) -> float:
            return float(sum(weight * float(getattr(diag, name)) for weight, diag in zip(self.weights, diagnostics, strict=True)))

        damage = float(np.clip(weighted("damage_fraction"), 0.0, 1.0))
        connected = float(np.clip(weighted("connected_active_fraction"), 0.0, 1.0))
        oxygen = float(np.clip(weighted("oxygen_deficient_fraction"), 0.0, 1.0))
        transformed = float(np.clip(weighted("transformed_phase_fraction"), 0.0, 1.0))
        trapped_oxygen = float(np.clip(weighted("trapped_oxygen_fraction"), 0.0, 1.0))
        crack_density = max(weighted("crack_surface_density_m_inv"), 0.0)
        minimum = self.config.minimum_property_multiplier
        active = max(minimum, connected * (1.0 - damage))
        capacity = max(minimum, np.power(connected, self.config.disconnected_capacity_exponent) * (1.0 - damage))
        diffusivity = max(
            minimum,
            np.power(max(1.0 - damage, 0.0), self.config.diffusivity_damage_exponent)
            * (1.0 - self.config.oxygen_deficiency_transport_penalty * oxygen),
        )
        conductivity = max(
            minimum,
            np.power(connected, self.config.conductivity_percolation_exponent)
            * np.power(max(1.0 - damage, 0.0), 2.0),
        )
        wetting_gain = 1.0 + self.config.crack_wetting_reaction_gain * np.tanh(
            crack_density * min(self.solvers[0].config.spacing_m)
        )
        reaction = max(minimum, active * wetting_gain)
        porosity = self.config.pressure_model.porosity_multiplier(self.config.stack_pressure_Pa)
        electrolyte_transport = max(minimum, porosity ** 1.5)
        plating_risk = self.config.pressure_model.plating_risk_multiplier(self.config.stack_pressure_Pa)
        return CathodeScaleFeedback(
            time_s=float(max(diag.time_s for diag in diagnostics)),
            mean_theta=weighted("mean_theta"),
            damage_fraction=damage,
            crack_surface_density_m_inv=crack_density,
            connected_active_fraction=connected,
            oxygen_deficient_fraction=oxygen,
            maximum_principal_stress_Pa=max(float(diag.maximum_principal_stress_Pa) for diag in diagnostics),
            contact_stress_factor=float(contact_factor),
            active_material_multiplier=float(active),
            diffusivity_multiplier=float(diffusivity),
            reaction_area_multiplier=float(reaction),
            electronic_conductivity_multiplier=float(conductivity),
            capacity_multiplier=float(capacity),
            porosity_multiplier=float(porosity),
            electrolyte_transport_multiplier=float(electrolyte_transport),
            plating_risk_multiplier=float(plating_risk),
            metadata={
                "rve_count": len(self.solvers),
                "stack_pressure_Pa": self.config.stack_pressure_Pa,
                "weights": tuple(map(float, self.weights)),
            },
            transformed_phase_fraction=transformed,
            trapped_oxygen_fraction=trapped_oxygen,
        )

    def apply_to_p2d(self, parameters: P2DParameters, feedback: CathodeScaleFeedback | None = None) -> P2DParameters:
        """Apply RVE feedback to the positive electrode without cumulative drift.

        The object is updated in-place to preserve existing solver references.
        Original pristine values are cached once per parameter object.
        """

        if feedback is None:
            if not self.history:
                raise RuntimeError("no cathode feedback is available")
            feedback = self.history[-1]
        key = id(parameters)
        baseline = self._parameter_baselines.setdefault(
            key,
            {
                "active_volume_fraction": float(parameters.positive.active_volume_fraction),
                "solid_diffusivity_m2_s": parameters.positive.solid_diffusivity_m2_s,
                "reaction_rate": parameters.positive.reaction_rate_m2p5_mol_m0p5_s,
                "conductivity": parameters.positive.electronic_conductivity_S_m,
                "porosity": float(parameters.positive.porosity),
            },
        )
        parameters.positive.porosity = float(
            np.clip(baseline["porosity"] * feedback.porosity_multiplier, 1.0e-6, 1.0 - 1.0e-6)
        )
        parameters.positive.active_volume_fraction = float(
            np.clip(
                baseline["active_volume_fraction"] * feedback.active_material_multiplier,
                1.0e-8,
                1.0 - parameters.positive.porosity - 1.0e-8,
            )
        )

        def scaled(base: Any, multiplier: float, name: str):
            def property_law(soc: float, temperature_K: float) -> float:
                return multiplier * evaluate_property(base, soc, temperature_K, name=name)
            return property_law

        parameters.positive.solid_diffusivity_m2_s = scaled(
            baseline["solid_diffusivity_m2_s"], feedback.diffusivity_multiplier,
            "positive.solid_diffusivity_m2_s",
        )
        parameters.positive.reaction_rate_m2p5_mol_m0p5_s = scaled(
            baseline["reaction_rate"], feedback.reaction_area_multiplier,
            "positive.reaction_rate",
        )
        parameters.positive.electronic_conductivity_S_m = scaled(
            baseline["conductivity"], feedback.electronic_conductivity_multiplier,
            "positive.electronic_conductivity_S_m",
        )
        return parameters

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_update_time_s": self.last_update_time_s,
            "states": [state.copy() for state in self.states],
            "history": list(self.history),
        }


__all__ = [
    "CathodeMechanicalMultiscaleModel",
    "CathodeScaleConfig",
    "CathodeScaleFeedback",
    "StackPressureContactModel",
]
