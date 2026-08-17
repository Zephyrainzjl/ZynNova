"""Three-dimensional FFT-accelerated cathode chemo-mechanical degradation.

Governing fields are lithium occupancy, mechanical equilibrium, AT2-like
fracture damage, fatigue, a dislocation/shear proxy, and oxygen-deficient
material.  The implementation uses periodic spectral derivatives and an
iterative heterogeneous-elasticity equilibrium correction, making it suitable
for large voxel RVEs and GPU-independent high-throughput studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

import numpy as np

from .materials import NMCCathodeMaterial


@dataclass(frozen=True, slots=True)
class CathodeSpectralConfig:
    spacing_m: float | tuple[float, float, float] = 1.0e-7
    active_phase_labels: tuple[int, ...] = (1,)
    time_step_s: float = 0.1
    mechanical_iterations: int = 24
    mechanical_tolerance: float = 1.0e-5
    mechanical_relaxation: float = 0.75
    diffusion_substeps: int = 1
    damage_substeps: int = 1
    maximum_occupancy_increment: float = 0.02
    maximum_damage_increment: float = 0.03
    fft_workers: int = -1
    external_stack_pressure_Pa: float = 0.0
    constrained_macroscopic_strain: bool = False
    re_equilibrate_after_damage: bool = True
    connectivity_mode: Literal["largest_component", "collector_boundary"] = "largest_component"
    connectivity_axis: int = 0
    dtype: str = "float64"

    def __post_init__(self) -> None:
        spacing = _triple(self.spacing_m)
        object.__setattr__(self, "spacing_m", spacing)
        if self.time_step_s <= 0.0:
            raise ValueError("time_step_s must be positive")
        if self.mechanical_iterations < 1 or self.diffusion_substeps < 1 or self.damage_substeps < 1:
            raise ValueError("iteration/substep counts must be positive")
        if not 0.0 < self.mechanical_relaxation <= 1.0:
            raise ValueError("mechanical_relaxation must lie in (0,1]")
        if self.mechanical_tolerance <= 0.0:
            raise ValueError("mechanical_tolerance must be positive")
        if self.connectivity_mode not in {"largest_component", "collector_boundary"}:
            raise ValueError("connectivity_mode is invalid")
        if self.connectivity_axis not in (0, 1, 2):
            raise ValueError("connectivity_axis must be 0, 1, or 2")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")


@dataclass(slots=True)
class CathodeDegradationState:
    time_s: float
    theta: np.ndarray
    damage: np.ndarray
    history_energy_J_m3: np.ndarray
    fatigue_energy_J_m3: np.ndarray
    plastic_shear: np.ndarray
    oxygen_deficiency: np.ndarray
    minimum_theta_history: np.ndarray
    previous_tensile_energy_J_m3: np.ndarray
    wetting_fraction: np.ndarray
    strain: np.ndarray
    stress_Pa: np.ndarray
    chemical_potential_J_mol: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)
    transformed_phase_fraction: np.ndarray | None = None
    mobile_oxygen_fraction: np.ndarray | None = None
    trapped_oxygen_fraction: np.ndarray | None = None

    def copy(self) -> "CathodeDegradationState":
        return CathodeDegradationState(
            time_s=float(self.time_s),
            theta=self.theta.copy(),
            damage=self.damage.copy(),
            history_energy_J_m3=self.history_energy_J_m3.copy(),
            fatigue_energy_J_m3=self.fatigue_energy_J_m3.copy(),
            plastic_shear=self.plastic_shear.copy(),
            oxygen_deficiency=self.oxygen_deficiency.copy(),
            minimum_theta_history=self.minimum_theta_history.copy(),
            previous_tensile_energy_J_m3=self.previous_tensile_energy_J_m3.copy(),
            wetting_fraction=self.wetting_fraction.copy(),
            strain=self.strain.copy(),
            stress_Pa=self.stress_Pa.copy(),
            chemical_potential_J_mol=self.chemical_potential_J_mol.copy(),
            metadata=dict(self.metadata),
            transformed_phase_fraction=(
                None
                if self.transformed_phase_fraction is None
                else self.transformed_phase_fraction.copy()
            ),
            mobile_oxygen_fraction=(
                None if self.mobile_oxygen_fraction is None else self.mobile_oxygen_fraction.copy()
            ),
            trapped_oxygen_fraction=(
                None if self.trapped_oxygen_fraction is None else self.trapped_oxygen_fraction.copy()
            ),
        )


@dataclass(frozen=True, slots=True)
class CathodeStepDiagnostics:
    time_s: float
    mean_theta: float
    maximum_principal_stress_Pa: float
    damage_fraction: float
    crack_surface_density_m_inv: float
    oxygen_deficient_fraction: float
    connected_active_fraction: float
    mechanical_iterations: int
    mechanical_residual: float
    lithium_balance_error: float
    transformed_phase_fraction: float = 0.0
    trapped_oxygen_fraction: float = 0.0


class SpectralCathodeDegradationSolver:
    """FFT-accelerated 3-D solver on a voxel microstructure."""

    def __init__(
        self,
        phase_labels: np.ndarray,
        *,
        grain_labels: np.ndarray | None = None,
        crystal_c_axes: Mapping[int, Sequence[float]] | np.ndarray | None = None,
        material: NMCCathodeMaterial | None = None,
        config: CathodeSpectralConfig | None = None,
    ) -> None:
        self.config = config or CathodeSpectralConfig()
        self.material = material or NMCCathodeMaterial(
            fracture_length_m=2.0 * min(self.config.spacing_m)
        )
        self.phase_labels = np.ascontiguousarray(phase_labels, dtype=np.int32)
        if self.phase_labels.ndim != 3 or min(self.phase_labels.shape) < 3:
            raise ValueError("phase_labels must be a 3-D array with each extent >= 3")
        self.active_mask = np.isin(self.phase_labels, np.asarray(self.config.active_phase_labels, dtype=np.int32))
        if not np.any(self.active_mask):
            raise ValueError("no active cathode voxels were found")
        if grain_labels is None:
            self.grain_labels = np.where(self.active_mask, 1, 0).astype(np.int32)
        else:
            self.grain_labels = np.ascontiguousarray(grain_labels, dtype=np.int32)
            if self.grain_labels.shape != self.phase_labels.shape:
                raise ValueError("grain_labels shape mismatch")
            self.grain_labels = np.where(self.active_mask, self.grain_labels, 0)
        self.c_axes = _orientation_field(self.grain_labels, crystal_c_axes)
        self.grain_boundary = _grain_boundary_indicator(self.grain_labels, self.active_mask)
        self.dtype = np.float32 if self.config.dtype == "float32" else np.float64
        self._wavevectors, self._k2 = _wavevectors(self.phase_labels.shape, self.config.spacing_m)
        self._active_volume_fraction = float(np.mean(self.active_mask))

    def initialize(
        self,
        theta: float | np.ndarray = 0.95,
        *,
        initial_damage: np.ndarray | None = None,
    ) -> CathodeDegradationState:
        occupancy = np.broadcast_to(np.asarray(theta, dtype=self.dtype), self.phase_labels.shape).copy()
        occupancy = np.where(self.active_mask, np.clip(occupancy, 1e-7, 1.0 - 1e-7), 0.0)
        damage = np.zeros_like(occupancy) if initial_damage is None else np.asarray(initial_damage, dtype=self.dtype).copy()
        if damage.shape != occupancy.shape:
            raise ValueError("initial_damage shape mismatch")
        damage = np.where(self.active_mask, np.clip(damage, 0.0, 1.0), 0.0)
        zeros = np.zeros_like(occupancy)
        transformed = np.zeros_like(occupancy)
        mobile_oxygen = np.zeros_like(occupancy)
        trapped_oxygen = np.zeros_like(occupancy)
        strain, stress, iterations, residual = self._mechanical_equilibrium(
            occupancy,
            damage,
            transformed,
            np.zeros(occupancy.shape + (3, 3), dtype=self.dtype),
            self.material.reference_temperature_K,
        )
        potential = self._chemical_potential(occupancy, stress, 298.15)
        return CathodeDegradationState(
            time_s=0.0,
            theta=occupancy,
            damage=damage,
            history_energy_J_m3=zeros.copy(),
            fatigue_energy_J_m3=zeros.copy(),
            plastic_shear=zeros.copy(),
            oxygen_deficiency=zeros.copy(),
            minimum_theta_history=occupancy.copy(),
            previous_tensile_energy_J_m3=zeros.copy(),
            wetting_fraction=zeros.copy(),
            strain=strain,
            stress_Pa=stress,
            chemical_potential_J_mol=potential,
            metadata={"mechanical_iterations": iterations, "mechanical_residual": residual},
            transformed_phase_fraction=transformed,
            mobile_oxygen_fraction=mobile_oxygen,
            trapped_oxygen_fraction=trapped_oxygen,
        )

    def step(
        self,
        state: CathodeDegradationState,
        *,
        mean_c_rate: float,
        temperature_K: float = 298.15,
        dt_s: float | None = None,
        mechanical_load_factor: float = 1.0,
    ) -> tuple[CathodeDegradationState, CathodeStepDiagnostics]:
        dt = float(self.config.time_step_s if dt_s is None else dt_s)
        if (
            dt <= 0.0
            or not np.isfinite(mean_c_rate)
            or temperature_K <= 0.0
            or not np.isfinite(mechanical_load_factor)
            or mechanical_load_factor <= 0.0
        ):
            raise ValueError("invalid time step, C-rate, temperature, or mechanical load factor")
        previous_mean = _active_mean(state.theta, self.active_mask)
        theta = state.theta.copy()
        stress = state.stress_Pa
        transformed = _state_field(state.transformed_phase_fraction, state.theta)
        mobile_oxygen = _state_field(state.mobile_oxygen_fraction, state.theta)
        trapped_oxygen = _state_field(state.trapped_oxygen_fraction, state.theta)
        for _ in range(self.config.diffusion_substeps):
            sub_dt = dt / self.config.diffusion_substeps
            potential = self._chemical_potential(theta, stress, temperature_K)
            source = self._reaction_source(theta, state.damage, state.wetting_fraction, mean_c_rate)
            theta = self._diffusion_step(
                theta,
                potential,
                state.damage,
                np.maximum(state.oxygen_deficiency, trapped_oxygen),
                source,
                temperature_K,
                sub_dt,
            )
        transformed, mobile_oxygen, trapped_oxygen = self._phase_oxygen_step(
            transformed,
            mobile_oxygen,
            trapped_oxygen,
            theta,
            state.damage,
            state.wetting_fraction,
            temperature_K,
            dt,
        )
        minimum_theta = np.minimum(state.minimum_theta_history, theta)
        strain, stress, mechanical_iterations, mechanical_residual = self._mechanical_equilibrium(
            theta, state.damage, transformed, state.strain, temperature_K
        )
        energy_plus, max_principal, max_shear = _tensile_energy_and_stress(
            strain - self._eigenstrain(theta, transformed), stress, self.material.poisson_ratio
        )
        # Sparse force chains amplify local mechanical driving, not lithium
        # flux.  Energy scales quadratically with the stress concentration.
        load_factor = float(mechanical_load_factor)
        energy_plus = energy_plus * load_factor**2
        max_principal = max_principal * load_factor
        max_shear = max_shear * load_factor
        history = np.maximum(state.history_energy_J_m3, energy_plus)
        fatigue = state.fatigue_energy_J_m3 + 0.5 * np.abs(
            energy_plus - state.previous_tensile_energy_J_m3
        )
        plastic_shear = self._plastic_shear_step(state.plastic_shear, max_shear, dt)
        shear_oxygen = _smooth_indicator(
            plastic_shear,
            self.material.oxygen_deficiency_shear_threshold,
            self.material.oxygen_deficiency_transition_width,
        ) * self.active_mask
        oxygen = np.clip(
            shear_oxygen + trapped_oxygen - shear_oxygen * trapped_oxygen,
            0.0,
            1.0,
        ) * self.active_mask
        damage = state.damage.copy()
        for _ in range(self.config.damage_substeps):
            damage = self._damage_step(
                damage,
                history,
                fatigue,
                minimum_theta,
                oxygen,
                dt / self.config.damage_substeps,
            )
        wetting = self._wetting_step(state.wetting_fraction, damage, dt)
        if self.config.re_equilibrate_after_damage and float(np.max(damage - state.damage)) > 1.0e-10:
            strain, stress, post_iterations, post_residual = self._mechanical_equilibrium(
                theta, damage, transformed, strain, temperature_K
            )
            mechanical_iterations += post_iterations
            mechanical_residual = post_residual
            _, max_principal, _ = _tensile_energy_and_stress(
                strain - self._eigenstrain(theta, transformed), stress, self.material.poisson_ratio
            )
            max_principal *= load_factor
        potential = self._chemical_potential(theta, stress, temperature_K)
        new_state = CathodeDegradationState(
            time_s=state.time_s + dt,
            theta=theta,
            damage=damage,
            history_energy_J_m3=history,
            fatigue_energy_J_m3=fatigue,
            plastic_shear=plastic_shear,
            oxygen_deficiency=oxygen,
            minimum_theta_history=minimum_theta,
            previous_tensile_energy_J_m3=energy_plus,
            wetting_fraction=wetting,
            strain=strain,
            stress_Pa=stress,
            chemical_potential_J_mol=potential,
            metadata={
                **state.metadata,
                "mechanical_iterations": mechanical_iterations,
                "mechanical_residual": mechanical_residual,
                "temperature_K": float(temperature_K),
                "mean_c_rate": float(mean_c_rate),
                "mechanical_load_factor": load_factor,
            },
            transformed_phase_fraction=transformed,
            mobile_oxygen_fraction=mobile_oxygen,
            trapped_oxygen_fraction=trapped_oxygen,
        )
        target_mean = previous_mean + mean_c_rate * dt / 3600.0
        actual_mean = _active_mean(theta, self.active_mask)
        balance_error = actual_mean - target_mean
        crack_density = _crack_surface_density(damage, self.config.spacing_m, self.active_mask)
        connected = _connected_active_fraction(
            self.active_mask,
            damage < 0.95,
            mode=self.config.connectivity_mode,
            axis=self.config.connectivity_axis,
        )
        diagnostics = CathodeStepDiagnostics(
            time_s=new_state.time_s,
            mean_theta=actual_mean,
            maximum_principal_stress_Pa=float(np.max(max_principal[self.active_mask])),
            damage_fraction=float(np.mean(damage[self.active_mask] > 0.95)),
            crack_surface_density_m_inv=crack_density,
            oxygen_deficient_fraction=float(np.mean(oxygen[self.active_mask])),
            connected_active_fraction=connected,
            mechanical_iterations=mechanical_iterations,
            mechanical_residual=mechanical_residual,
            lithium_balance_error=float(balance_error),
            transformed_phase_fraction=float(np.mean(transformed[self.active_mask])),
            trapped_oxygen_fraction=float(np.mean(trapped_oxygen[self.active_mask])),
        )
        return new_state, diagnostics

    def run(
        self,
        initial_state: CathodeDegradationState,
        c_rate: Sequence[float] | np.ndarray,
        *,
        temperature_K: float | Sequence[float] = 298.15,
        dt_s: float | None = None,
        store_every: int = 1,
    ) -> tuple[list[CathodeDegradationState], list[CathodeStepDiagnostics]]:
        rates = np.asarray(c_rate, dtype=float).reshape(-1)
        temperatures = np.broadcast_to(np.asarray(temperature_K, dtype=float), rates.shape)
        state = initial_state.copy()
        states = [state.copy()]
        diagnostics: list[CathodeStepDiagnostics] = []
        for index, (rate, temperature) in enumerate(zip(rates, temperatures, strict=True), start=1):
            state, report = self.step(state, mean_c_rate=float(rate), temperature_K=float(temperature), dt_s=dt_s)
            diagnostics.append(report)
            if index % max(int(store_every), 1) == 0:
                states.append(state.copy())
        if states[-1].time_s != state.time_s:
            states.append(state.copy())
        return states, diagnostics

    def _eigenstrain(
        self,
        theta: np.ndarray,
        transformed_phase_fraction: np.ndarray | None = None,
    ) -> np.ndarray:
        basal, c_axis = self.material.lattice_strains(theta)
        if transformed_phase_fraction is not None:
            transition_basal, transition_c = self.material.transition_mismatch_strains(
                transformed_phase_fraction
            )
            basal = basal + transition_basal
            c_axis = c_axis + transition_c
        identity = np.eye(3, dtype=self.dtype)
        outer = self.c_axes[..., :, None] * self.c_axes[..., None, :]
        eigen = basal[..., None, None] * identity + (c_axis - basal)[..., None, None] * outer
        return eigen * self.active_mask[..., None, None]

    def _eigenstrain_derivative(self, theta: np.ndarray) -> np.ndarray:
        basal, c_axis = self.material.lattice_strain_derivatives(theta)
        identity = np.eye(3, dtype=self.dtype)
        outer = self.c_axes[..., :, None] * self.c_axes[..., None, :]
        return (
            basal[..., None, None] * identity
            + (c_axis - basal)[..., None, None] * outer
        ) * self.active_mask[..., None, None]

    def _chemical_potential(
        self,
        theta: np.ndarray,
        stress: np.ndarray,
        temperature_K: float,
    ) -> np.ndarray:
        chemical = self.material.chemical_potential_J_mol(theta, temperature_K)
        gradient = -(
            self.material.gradient_energy_J_m
            / self.material.maximum_lithium_concentration_mol_m3
        ) * _laplacian(theta, self.config.spacing_m)
        derivative = self._eigenstrain_derivative(theta)
        mechanical = -np.sum(stress * derivative, axis=(-2, -1)) / self.material.maximum_lithium_concentration_mol_m3
        potential = chemical + gradient + mechanical
        return np.where(self.active_mask, potential, 0.0).astype(self.dtype, copy=False)

    def _diffusion_step(
        self,
        theta: np.ndarray,
        potential: np.ndarray,
        damage: np.ndarray,
        oxygen_exposure: np.ndarray,
        source: np.ndarray,
        temperature_K: float,
        dt: float,
    ) -> np.ndarray:
        gradients = _gradient(potential, self.config.spacing_m)
        diffusivity = self.material.diffusivity(theta, temperature_K)
        mobility_factor = np.clip(theta * (1.0 - theta), 1e-8, None) / (
            8.31446261815324 * temperature_K
        )
        n = self.c_axes
        directional = sum(n[..., axis] * gradients[axis] for axis in range(3))
        flux: list[np.ndarray] = []
        damage_transport = (
            np.power(np.maximum(1.0 - damage, 0.0), self.material.damage_transport_exponent)
            + self.material.crack_transport_factor * damage * (1.0 - damage)
        )
        oxygen_transport = np.maximum(
            1.0
            - self.material.oxygen_transport_penalty
            * np.clip(np.asarray(oxygen_exposure, dtype=float), 0.0, 1.0),
            1.0e-4,
        )
        for axis in range(3):
            basal_gradient = gradients[axis] - n[..., axis] * directional
            c_gradient = n[..., axis] * directional
            component = -diffusivity * mobility_factor * damage_transport * oxygen_transport * (
                basal_gradient + self.material.basal_to_c_axis_diffusivity_ratio * c_gradient
            )
            flux.append(np.where(self.active_mask, component, 0.0))
        derivative = -_divergence(flux, self.config.spacing_m) + source
        maximum = self.config.maximum_occupancy_increment / max(dt, 1e-30)
        derivative = np.clip(derivative, -maximum, maximum)
        updated = theta + dt * derivative
        updated = np.where(self.active_mask, np.clip(updated, 1e-7, 1.0 - 1e-7), 0.0)
        return updated.astype(self.dtype, copy=False)

    def _reaction_source(
        self,
        theta: np.ndarray,
        damage: np.ndarray,
        wetting: np.ndarray,
        mean_c_rate: float,
    ) -> np.ndarray:
        surface = _active_surface_density(self.active_mask, self.config.spacing_m)
        crack = np.sqrt(sum(component * component for component in _gradient(damage, self.config.spacing_m)))
        weights = surface + self.material.crack_wetting_factor * wetting * crack
        weights = np.where(self.active_mask, weights, 0.0)
        mean_weight = float(np.mean(weights[self.active_mask]))
        if mean_weight <= 0.0:
            weights = self.active_mask.astype(float)
            mean_weight = 1.0
        # Mean occupancy rate over active material equals C-rate / 3600.
        return (float(mean_c_rate) / 3600.0) * weights / mean_weight

    def _wetting_step(
        self,
        wetting: np.ndarray,
        damage: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        crack_gradient = np.sqrt(
            sum(component * component for component in _gradient(damage, self.config.spacing_m))
        )
        surface = _active_surface_density(self.active_mask, self.config.spacing_m)
        driving = crack_gradient + surface
        scale = float(np.quantile(driving[self.active_mask], 0.95)) if np.any(self.active_mask) else 1.0
        target = np.clip(driving / max(scale, 1.0e-30), 0.0, 1.0)
        diffusivity = self.material.crack_wetting_diffusivity_m2_s
        updated = wetting + dt * (
            (target - wetting) / self.material.crack_wetting_time_s
            + diffusivity * _laplacian(wetting, self.config.spacing_m)
        )
        return np.where(self.active_mask, np.clip(updated, 0.0, 1.0), 0.0).astype(self.dtype, copy=False)

    def _phase_oxygen_step(
        self,
        transformed: np.ndarray,
        mobile_oxygen: np.ndarray,
        trapped_oxygen: np.ndarray,
        theta: np.ndarray,
        damage: np.ndarray,
        wetting: np.ndarray,
        temperature_K: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance high-voltage phase mismatch and crack-directed oxygen.

        This is a bounded reduced mechanism, not an atomistically resolved
        oxygen-redox model.  The transformed fraction relaxes toward a
        chemistry-specific target and is irreversible by default.  Newly
        transformed material releases a configurable mobile-oxygen fraction;
        spectral diffusion and a crack/wetting-dependent sink then move it
        toward internal free surfaces where it is stored as trapped oxygen.
        """

        target = self.material.high_voltage_transition_target(theta, temperature_K)
        ratio = dt / self.material.high_voltage_transition_time_s
        relaxed = (transformed + ratio * target) / (1.0 + ratio)
        updated_transformed = np.maximum(transformed, relaxed)
        updated_transformed = np.where(
            self.active_mask,
            np.clip(updated_transformed, 0.0, 1.0),
            0.0,
        )
        increment = np.maximum(updated_transformed - transformed, 0.0)
        generated = np.clip(
            mobile_oxygen + self.material.oxygen_generation_per_transition * increment,
            0.0,
            1.0,
        )
        transformed_hat = _fftn(generated, self.config.fft_workers)
        denominator = (
            1.0
            + dt * self.material.oxygen_migration_diffusivity_m2_s * self._k2
        )
        diffused = _ifftn(transformed_hat / denominator, self.config.fft_workers).real
        crack_gradient = np.sqrt(
            sum(component * component for component in _gradient(damage, self.config.spacing_m))
        )
        surface = _active_surface_density(self.active_mask, self.config.spacing_m)
        sink_driver = crack_gradient + wetting * (crack_gradient + surface)
        if np.any(self.active_mask):
            scale = float(np.quantile(sink_driver[self.active_mask], 0.95))
        else:
            scale = 1.0
        normalized = np.clip(sink_driver / max(scale, 1.0e-30), 0.0, 1.0)
        trap_rate = (
            1.0
            + self.material.oxygen_crack_trapping_gain * normalized
        ) / self.material.oxygen_trapping_time_s
        trapped_increment = np.clip(diffused, 0.0, 1.0) * (
            1.0 - np.exp(-dt * trap_rate)
        )
        updated_mobile = np.clip(diffused - trapped_increment, 0.0, 1.0)
        updated_trapped = np.clip(trapped_oxygen + trapped_increment, 0.0, 1.0)
        updated_mobile = np.where(self.active_mask, updated_mobile, 0.0)
        updated_trapped = np.where(self.active_mask, updated_trapped, 0.0)
        return (
            updated_transformed.astype(self.dtype, copy=False),
            updated_mobile.astype(self.dtype, copy=False),
            updated_trapped.astype(self.dtype, copy=False),
        )

    def _mechanical_equilibrium(
        self,
        theta: np.ndarray,
        damage: np.ndarray,
        transformed_phase_fraction: np.ndarray,
        initial_strain: np.ndarray,
        temperature_K: float,
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        eigen = self._eigenstrain(theta, transformed_phase_fraction)
        modulus = self.material.young_modulus(theta, temperature_K)
        degradation = np.power(1.0 - np.clip(damage, 0.0, 1.0), 2) + self.material.residual_stiffness
        modulus = np.where(self.active_mask, modulus * degradation, np.maximum(np.mean(modulus[self.active_mask]) * 1e-4, 1e6))
        nu = self.material.poisson_ratio
        shear = modulus / (2.0 * (1.0 + nu))
        lame = modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        reference_E = float(np.mean(modulus[self.active_mask]))
        reference_mu = reference_E / (2.0 * (1.0 + nu))
        reference_lame = reference_E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        if initial_strain.shape != theta.shape + (3, 3):
            strain = np.zeros(theta.shape + (3, 3), dtype=self.dtype)
        else:
            strain = np.asarray(initial_strain, dtype=self.dtype).copy()
        mean_eigen = np.mean(eigen[self.active_mask], axis=0)
        if not self.config.constrained_macroscopic_strain:
            strain += mean_eigen - np.mean(strain, axis=(0, 1, 2))
        if self.config.external_stack_pressure_Pa:
            bulk = reference_E / (3.0 * (1.0 - 2.0 * nu))
            strain += -self.config.external_stack_pressure_Pa / (3.0 * bulk) * np.eye(3)
        residual = np.inf
        stress = np.zeros_like(strain)
        for iteration in range(1, self.config.mechanical_iterations + 1):
            elastic = strain - eigen
            stress = self.material.transversely_isotropic_stress(
                elastic,
                self.c_axes,
                theta,
                temperature_K,
                degradation,
            )
            stress = np.where(
                self.active_mask[..., None, None],
                stress,
                2.0 * shear[..., None, None] * elastic
                + lame[..., None, None] * np.trace(elastic, axis1=-2, axis2=-1)[..., None, None] * np.eye(3),
            )
            if self.config.external_stack_pressure_Pa:
                stress -= self.config.external_stack_pressure_Pa * np.eye(3)
            correction, residual = _equilibrium_correction(
                stress,
                self._wavevectors,
                self._k2,
                reference_lame,
                reference_mu,
                self.config.fft_workers,
            )
            strain += self.config.mechanical_relaxation * correction
            strain = 0.5 * (strain + np.swapaxes(strain, -1, -2))
            if residual <= self.config.mechanical_tolerance:
                return strain, stress, iteration, residual
        return strain, stress, self.config.mechanical_iterations, residual

    def _damage_step(
        self,
        damage: np.ndarray,
        history: np.ndarray,
        fatigue: np.ndarray,
        minimum_theta: np.ndarray,
        oxygen_exposure: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        gc = self.material.fracture_energy_field(
            grain_boundary_indicator=self.grain_boundary,
            fatigue=fatigue,
            minimum_theta_history=minimum_theta,
            oxygen_exposure=oxygen_exposure,
        )
        ell = self.material.fracture_length_m
        eta = self.material.damage_viscosity_Pa_s
        reaction = (
            damage + dt / eta * 2.0 * history
        ) / np.maximum(
            1.0 + dt / eta * (2.0 * history + gc / ell),
            1e-30,
        )
        gc_reference = float(np.mean(gc[self.active_mask]))
        transformed = _fftn(reaction, self.config.fft_workers)
        denominator = 1.0 + dt / eta * gc_reference * ell * self._k2
        smoothed = _ifftn(transformed / denominator, self.config.fft_workers).real
        increment = np.clip(smoothed - damage, 0.0, self.config.maximum_damage_increment)
        updated = np.maximum(damage, damage + increment)
        return np.where(self.active_mask, np.clip(updated, 0.0, 1.0), 0.0).astype(self.dtype, copy=False)

    def _plastic_shear_step(self, plastic_shear: np.ndarray, max_shear: np.ndarray, dt: float) -> np.ndarray:
        driving = np.maximum(max_shear / self.material.critical_resolved_shear_Pa - 1.0, 0.0)
        rate = np.power(driving, self.material.dislocation_rate_exponent) / self.material.dislocation_reference_time_s
        return np.where(self.active_mask, plastic_shear + dt * rate, 0.0)


def _equilibrium_correction(
    stress: np.ndarray,
    wavevectors: tuple[np.ndarray, np.ndarray, np.ndarray],
    k2: np.ndarray,
    lame: float,
    shear: float,
    workers: int,
) -> tuple[np.ndarray, float]:
    sigma_hat = [[_fftn(stress[..., i, j], workers) for j in range(3)] for i in range(3)]
    residual_hat = []
    for i in range(3):
        residual_hat.append(sum(wavevectors[j] * sigma_hat[i][j] for j in range(3)))
    residual_norm = float(np.sqrt(sum(np.mean(np.abs(value) ** 2) for value in residual_hat)))
    stress_norm = float(np.sqrt(sum(np.mean(np.abs(value) ** 2) for row in sigma_hat for value in row)))
    relative = residual_norm / max(stress_norm * np.sqrt(float(np.max(k2))), 1e-30)
    safe_k2 = np.where(k2 > 0.0, k2, 1.0)
    dot = sum(wavevectors[i] * residual_hat[i] for i in range(3))
    coefficient = (lame + shear) / (shear * (lame + 2.0 * shear))
    velocity = [
        residual_hat[i] / (shear * safe_k2)
        - coefficient * wavevectors[i] * dot / (safe_k2 * safe_k2)
        for i in range(3)
    ]
    correction = np.empty(stress.shape, dtype=float)
    for i in range(3):
        for j in range(3):
            value_hat = -0.5 * (wavevectors[i] * velocity[j] + wavevectors[j] * velocity[i])
            value_hat.flat[0] = 0.0
            correction[..., i, j] = _ifftn(value_hat, workers).real
    return correction, relative


def _tensile_energy_and_stress(
    elastic_strain: np.ndarray,
    stress: np.ndarray,
    poisson_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strain_values, strain_vectors = np.linalg.eigh(0.5 * (elastic_strain + np.swapaxes(elastic_strain, -1, -2)))
    positive = np.maximum(strain_values, 0.0)
    eps_plus = np.einsum("...ik,...k,...jk->...ij", strain_vectors, positive, strain_vectors, optimize=True)
    # Local Lamé parameters can be reconstructed from stress/strain only
    # approximately; use tensile stress work, guaranteed non-negative.
    energy = np.maximum(0.5 * np.sum(stress * eps_plus, axis=(-2, -1)), 0.0)
    principal = np.linalg.eigvalsh(0.5 * (stress + np.swapaxes(stress, -1, -2)))
    maximum = principal[..., -1]
    max_shear = 0.5 * (principal[..., -1] - principal[..., 0])
    return energy, maximum, max_shear


def _orientation_field(
    grain_labels: np.ndarray,
    supplied: Mapping[int, Sequence[float]] | np.ndarray | None,
) -> np.ndarray:
    shape = grain_labels.shape
    field = np.zeros(shape + (3,), dtype=float)
    if isinstance(supplied, np.ndarray):
        array = np.asarray(supplied, dtype=float)
        if array.shape == shape + (3,):
            field = array.copy()
        else:
            raise ValueError("crystal_c_axes array must have shape labels.shape + (3,)")
    else:
        mapping = dict(supplied or {})
        for grain in np.unique(grain_labels):
            if grain <= 0:
                continue
            if int(grain) in mapping:
                axis = np.asarray(mapping[int(grain)], dtype=float)
            else:
                # Deterministic quasi-random orientation from grain id.
                rng = np.random.default_rng(int(grain) * 104729)
                axis = rng.normal(size=3)
            norm = float(np.linalg.norm(axis))
            if norm <= 0.0:
                raise ValueError("crystal c-axis cannot be zero")
            field[grain_labels == grain] = axis / norm
    norms = np.linalg.norm(field, axis=-1)
    active = grain_labels > 0
    field[active] /= np.maximum(norms[active, None], 1e-30)
    field[~active] = np.asarray([0.0, 0.0, 1.0])
    return field


def _grain_boundary_indicator(grains: np.ndarray, active: np.ndarray) -> np.ndarray:
    indicator = np.zeros(grains.shape, dtype=float)
    for axis in range(3):
        different = (grains != np.roll(grains, 1, axis=axis)) & active
        indicator = np.maximum(indicator, different.astype(float))
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return indicator
    return np.clip(gaussian_filter(indicator, sigma=0.65, mode="nearest"), 0.0, 1.0)


def _wavevectors(shape: tuple[int, int, int], spacing: tuple[float, float, float]) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    vectors = tuple(
        2.0 * np.pi * np.fft.fftfreq(shape[axis], d=spacing[axis]).reshape(
            tuple(shape[axis] if i == axis else 1 for i in range(3))
        )
        for axis in range(3)
    )
    k2 = sum(value * value for value in vectors)
    return vectors, k2


def _gradient(values: np.ndarray, spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / (2.0 * spacing[axis])
        for axis in range(3)
    )  # type: ignore[return-value]


def _divergence(values: Sequence[np.ndarray], spacing: tuple[float, float, float]) -> np.ndarray:
    return sum(
        (np.roll(values[axis], -1, axis=axis) - np.roll(values[axis], 1, axis=axis)) / (2.0 * spacing[axis])
        for axis in range(3)
    )


def _laplacian(values: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    return sum(
        (np.roll(values, -1, axis=axis) - 2.0 * values + np.roll(values, 1, axis=axis)) / spacing[axis] ** 2
        for axis in range(3)
    )


def _active_surface_density(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    density = np.zeros(mask.shape, dtype=float)
    for axis in range(3):
        density += ((mask != np.roll(mask, 1, axis=axis)) | (mask != np.roll(mask, -1, axis=axis))) / spacing[axis]
    return density * mask


def _crack_surface_density(damage: np.ndarray, spacing: tuple[float, float, float], active: np.ndarray) -> float:
    gradient = _gradient(damage, spacing)
    density = np.sqrt(sum(value * value for value in gradient))
    return float(np.mean(density[active]))


def _connected_active_fraction(
    active: np.ndarray,
    intact: np.ndarray,
    *,
    mode: str,
    axis: int,
) -> float:
    """Return surviving connected active material under an explicit topology model.

    Particle/grain RVEs normally use ``largest_component`` because they need
    not touch the numerical box boundary.  Electrode-scale subvolumes can use
    ``collector_boundary`` to retain only components connected to either face
    normal to the selected current-collector axis.
    """

    mask = active & intact
    active_count = max(np.count_nonzero(active), 1)
    try:
        from scipy import ndimage
    except ImportError:
        return float(np.count_nonzero(mask) / active_count)
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    if count == 0:
        return 0.0
    if mode == "largest_component":
        sizes = np.bincount(labels.ravel())
        return float(np.max(sizes[1:], initial=0) / active_count)
    lower = np.take(labels, 0, axis=axis)
    upper = np.take(labels, -1, axis=axis)
    connected_ids = np.unique(np.concatenate((lower.ravel(), upper.ravel())))
    connected_ids = connected_ids[connected_ids > 0]
    connected = np.isin(labels, connected_ids)
    return float(np.count_nonzero(connected & active) / active_count)


def _active_mean(values: np.ndarray, active: np.ndarray) -> float:
    return float(np.mean(values[active]))


def _state_field(values: np.ndarray | None, template: np.ndarray) -> np.ndarray:
    if values is None:
        return np.zeros_like(template)
    array = np.asarray(values, dtype=template.dtype)
    if array.shape != template.shape:
        raise ValueError("cathode state field shape mismatch")
    return array.copy()


def _smooth_indicator(values: np.ndarray, threshold: float, width: float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh((values - threshold) / max(width, 1e-12)))


def _fftn(values: np.ndarray, workers: int) -> np.ndarray:
    try:
        from scipy import fft
    except ImportError:
        return np.fft.fftn(values)
    return fft.fftn(values, workers=workers)


def _ifftn(values: np.ndarray, workers: int) -> np.ndarray:
    try:
        from scipy import fft
    except ImportError:
        return np.fft.ifftn(values)
    return fft.ifftn(values, workers=workers)


def _triple(value: float | Sequence[float]) -> tuple[float, float, float]:
    result = (float(value),) * 3 if np.isscalar(value) else tuple(map(float, value))
    if len(result) != 3 or min(result) <= 0.0:
        raise ValueError("spacing must contain three positive values")
    return result  # type: ignore[return-value]


__all__ = [
    "CathodeDegradationState",
    "CathodeSpectralConfig",
    "CathodeStepDiagnostics",
    "SpectralCathodeDegradationSolver",
]
