"""Conservative, thermally coupled Doyle–Fuller–Newman/P2D solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ...constants import FARADAY, GAS_CONSTANT
from ...exceptions import ConvergenceError
from ..properties import evaluate_property
from .numerics import (
    ThroughCellGrid,
    conservative_diffusion_step,
    harmonic_face_coefficients,
    spherical_inventory,
    spherical_particle_step,
    through_cell_grid,
)
from .parameters import ElectrodeParameters, P2DParameters
from .protocol import CurrentSegment, P2DTrajectory
from .state import P2DState, P2DStepDiagnostics


MaterialUpdate = Callable[[P2DParameters, float, float], None]


@dataclass(slots=True)
class _PotentialResult:
    phi_s_negative: np.ndarray
    phi_s_positive: np.ndarray
    phi_electrolyte: np.ndarray
    j_negative: np.ndarray
    j_positive: np.ndarray
    evaluations: int
    residual_norm: float


class P2DModel:
    """A finite-volume DFN model with spherical active-material particles.

    Sign convention: positive cell current is discharge. Interfacial current is
    positive for oxidation (outward lithium flux from the negative particle) and
    negative for reduction in the positive electrode.
    """

    def __init__(
        self,
        parameters: P2DParameters,
        *,
        material_update: MaterialUpdate | None = None,
    ) -> None:
        self.parameters = parameters
        self.material_update = material_update
        grid = parameters.discretization
        self.grid = through_cell_grid(
            parameters.negative.thickness_m,
            parameters.separator.thickness_m,
            parameters.positive.thickness_m,
            grid.negative_cells,
            grid.separator_cells,
            grid.positive_cells,
        )

    def initialize(
        self,
        soc: float = 1.0,
        *,
        temperature_K: float | None = None,
    ) -> P2DState:
        if not 0.0 <= soc <= 1.0:
            raise ValueError("initial SOC must lie in [0, 1]")
        parameters = self.parameters
        temperature = (
            parameters.initial_temperature_K
            if temperature_K is None
            else float(temperature_K)
        )
        if temperature <= 0.0:
            raise ValueError("initial temperature must be positive")
        if self.material_update is not None:
            self.material_update(parameters, float(soc), temperature)
        discretization = parameters.discretization
        theta_n = parameters.negative.stoichiometry(soc)
        theta_p = parameters.positive.stoichiometry(soc)
        c_n = np.full(
            (
                discretization.negative_cells,
                discretization.negative_particle_cells,
            ),
            theta_n * parameters.negative.maximum_concentration_mol_m3,
        )
        c_p = np.full(
            (
                discretization.positive_cells,
                discretization.positive_particle_cells,
            ),
            theta_p * parameters.positive.maximum_concentration_mol_m3,
        )
        c_e = np.full(
            len(self.grid.centers_m),
            parameters.electrolyte.initial_concentration_mol_m3,
        )
        phi_e = np.zeros_like(c_e)
        phi_s_n = np.full(
            discretization.negative_cells,
            float(parameters.negative.ocp_V(theta_n, temperature)),
        )
        phi_s_p = np.full(
            discretization.positive_cells,
            float(parameters.positive.ocp_V(theta_p, temperature)),
        )
        voltage = float(phi_s_p[-1] - phi_s_n[0])
        state = P2DState(
            time_s=0.0,
            electrolyte_concentration_mol_m3=c_e,
            negative_particle_concentration_mol_m3=c_n,
            positive_particle_concentration_mol_m3=c_p,
            electrolyte_potential_V=phi_e,
            negative_solid_potential_V=phi_s_n,
            positive_solid_potential_V=phi_s_p,
            negative_interfacial_current_A_m2=np.zeros(discretization.negative_cells),
            positive_interfacial_current_A_m2=np.zeros(discretization.positive_cells),
            temperature_K=temperature,
            terminal_voltage_V=voltage,
        )
        state.validate(parameters)
        return state

    def step(self, state: P2DState, current_A: float, dt_s: float) -> P2DState:
        parameters = self.parameters
        state.validate(parameters)
        if dt_s <= 0.0 or not np.isfinite(current_A):
            raise ValueError("P2D time step must be positive and current must be finite")
        soc = state.soc(parameters)
        if self.material_update is not None:
            self.material_update(parameters, soc, state.temperature_K)

        old_electrolyte = state.electrolyte_concentration_mol_m3
        old_negative = state.negative_particle_concentration_mol_m3
        old_positive = state.positive_particle_concentration_mol_m3
        iterate_electrolyte = old_electrolyte.copy()
        iterate_negative = old_negative.copy()
        iterate_positive = old_positive.copy()
        initial_potential = np.concatenate(
            (
                state.negative_solid_potential_V,
                state.positive_solid_potential_V,
                state.electrolyte_potential_V,
            )
        )
        potential: _PotentialResult | None = None
        coupling_error = float("inf")

        for coupling_iteration in range(1, parameters.coupling_max_iterations + 1):
            potential = self._solve_potentials(
                iterate_electrolyte,
                iterate_negative,
                iterate_positive,
                state.temperature_K,
                current_A,
                initial_potential,
                soc,
            )
            initial_potential = np.concatenate(
                (
                    potential.phi_s_negative,
                    potential.phi_s_positive,
                    potential.phi_electrolyte,
                )
            )
            updated_negative = spherical_particle_step(
                old_negative,
                dt_s,
                parameters.negative.particle_radius_m,
                parameters.negative.diffusivity(soc, state.temperature_K),
                potential.j_negative,
                FARADAY,
            )
            updated_positive = spherical_particle_step(
                old_positive,
                dt_s,
                parameters.positive.particle_radius_m,
                parameters.positive.diffusivity(soc, state.temperature_K),
                potential.j_positive,
                FARADAY,
            )
            updated_electrolyte = self._electrolyte_step(
                old_electrolyte,
                potential.j_negative,
                potential.j_positive,
                dt_s,
                soc,
                state.temperature_K,
            )
            self._validate_concentrations(
                updated_electrolyte, updated_negative, updated_positive
            )
            errors = (
                _relative_change(updated_electrolyte, iterate_electrolyte),
                _relative_change(updated_negative, iterate_negative),
                _relative_change(updated_positive, iterate_positive),
            )
            coupling_error = max(errors)
            relaxation = parameters.coupling_relaxation
            iterate_electrolyte = (
                relaxation * updated_electrolyte
                + (1.0 - relaxation) * iterate_electrolyte
            )
            iterate_negative = (
                relaxation * updated_negative + (1.0 - relaxation) * iterate_negative
            )
            iterate_positive = (
                relaxation * updated_positive + (1.0 - relaxation) * iterate_positive
            )
            if coupling_error <= parameters.coupling_tolerance:
                iterate_electrolyte = updated_electrolyte
                iterate_negative = updated_negative
                iterate_positive = updated_positive
                break
        else:
            raise ConvergenceError(
                "P2D electrochemical coupling did not converge: "
                f"relative update={coupling_error:.3e}"
            )
        assert potential is not None

        voltage = float(
            potential.phi_s_positive[-1] - potential.phi_s_negative[0]
        )
        temperature = self._temperature_step(
            state,
            current_A,
            voltage,
            iterate_negative,
            iterate_positive,
            dt_s,
        )
        inventory_errors = self._inventory_errors(
            old_electrolyte,
            iterate_electrolyte,
            old_negative,
            iterate_negative,
            old_positive,
            iterate_positive,
            potential.j_negative,
            potential.j_positive,
            dt_s,
        )
        result = P2DState(
            time_s=state.time_s + dt_s,
            electrolyte_concentration_mol_m3=iterate_electrolyte,
            negative_particle_concentration_mol_m3=iterate_negative,
            positive_particle_concentration_mol_m3=iterate_positive,
            electrolyte_potential_V=potential.phi_electrolyte,
            negative_solid_potential_V=potential.phi_s_negative,
            positive_solid_potential_V=potential.phi_s_positive,
            negative_interfacial_current_A_m2=potential.j_negative,
            positive_interfacial_current_A_m2=potential.j_positive,
            temperature_K=temperature,
            terminal_voltage_V=voltage,
            current_A=float(current_A),
            diagnostics=P2DStepDiagnostics(
                converged=True,
                potential_evaluations=potential.evaluations,
                potential_residual_norm=potential.residual_norm,
                coupling_iterations=coupling_iteration,
                coupling_error=coupling_error,
                electrolyte_inventory_error=inventory_errors[0],
                negative_inventory_error=inventory_errors[1],
                positive_inventory_error=inventory_errors[2],
                message="fully coupled fixed-point/implicit finite-volume step",
            ),
            metadata={
                "soc_at_step_start": soc,
                "sign_convention": "positive current is discharge",
            },
        )
        result.validate(parameters)
        return result

    def run(
        self,
        protocol: list[CurrentSegment] | tuple[CurrentSegment, ...],
        *,
        initial_state: P2DState | None = None,
        initial_soc: float = 1.0,
    ) -> P2DTrajectory:
        state = self.initialize(initial_soc) if initial_state is None else initial_state.copy()
        trajectory = P2DTrajectory(states=[state.copy()], segment_labels=["initial"])
        for segment_index, segment in enumerate(protocol):
            local_time = 0.0
            label = segment.label or f"segment-{segment_index}"
            while local_time < segment.duration_s - 1.0e-14:
                dt_s = min(segment.time_step_s, segment.duration_s - local_time)
                current = segment.current(local_time, state)
                state = self.step(state, current, dt_s)
                trajectory.states.append(state.copy())
                trajectory.segment_labels.append(label)
                local_time += dt_s
                if (
                    segment.minimum_voltage_V is not None
                    and state.terminal_voltage_V <= segment.minimum_voltage_V
                ):
                    trajectory.termination_reason = (
                        f"{label}: minimum voltage {segment.minimum_voltage_V:g} V reached"
                    )
                    return trajectory
                if (
                    segment.maximum_voltage_V is not None
                    and state.terminal_voltage_V >= segment.maximum_voltage_V
                ):
                    trajectory.termination_reason = (
                        f"{label}: maximum voltage {segment.maximum_voltage_V:g} V reached"
                    )
                    return trajectory
        return trajectory

    def _solve_potentials(
        self,
        electrolyte_concentration: np.ndarray,
        negative_particles: np.ndarray,
        positive_particles: np.ndarray,
        temperature_K: float,
        current_A: float,
        initial: np.ndarray,
        soc: float,
    ) -> _PotentialResult:
        try:
            from scipy.optimize import least_squares
        except ImportError as exc:
            raise ImportError("P2D potential solve requires SciPy") from exc
        p = self.parameters
        grid = self.grid
        n_negative = p.discretization.negative_cells
        n_positive = p.discretization.positive_cells
        total = len(grid.widths_m)
        current_density = current_A / p.area_m2
        current_scale = max(abs(current_density), 1.0)
        concentration = np.maximum(
            np.asarray(electrolyte_concentration, dtype=float),
            p.minimum_concentration_mol_m3,
        )
        theta_n = np.clip(
            negative_particles[:, -1] / p.negative.maximum_concentration_mol_m3,
            1.0e-8,
            1.0 - 1.0e-8,
        )
        theta_p = np.clip(
            positive_particles[:, -1] / p.positive.maximum_concentration_mol_m3,
            1.0e-8,
            1.0 - 1.0e-8,
        )
        c_surface_n = theta_n * p.negative.maximum_concentration_mol_m3
        c_surface_p = theta_p * p.positive.maximum_concentration_mol_m3
        ocp_n = np.asarray(p.negative.ocp_V(theta_n, temperature_K), dtype=float)
        ocp_p = np.asarray(p.positive.ocp_V(theta_p, temperature_K), dtype=float)
        i0_n = self._exchange_current(
            p.negative,
            concentration[grid.negative],
            c_surface_n,
            soc,
            temperature_K,
        )
        i0_p = self._exchange_current(
            p.positive,
            concentration[grid.positive],
            c_surface_p,
            soc,
            temperature_K,
        )
        sigma_n = (
            p.negative.conductivity(soc, temperature_K)
            * p.negative.active_volume_fraction ** p.negative.bruggeman
        )
        sigma_p = (
            p.positive.conductivity(soc, temperature_K)
            * p.positive.active_volume_fraction ** p.positive.bruggeman
        )
        kappa = evaluate_property(
            p.electrolyte.ionic_conductivity_S_m,
            soc,
            temperature_K,
            name="electrolyte.ionic_conductivity_S_m",
        )
        porosity, bruggeman = self._electrolyte_structure()
        kappa_cells = kappa * porosity**bruggeman
        thermodynamic_factor = evaluate_property(
            p.electrolyte.thermodynamic_factor,
            soc,
            temperature_K,
            name="electrolyte.thermodynamic_factor",
        )
        beta = (
            2.0
            * GAS_CONSTANT
            * temperature_K
            * (1.0 - p.electrolyte.transference_number)
            * thermodynamic_factor
            / FARADAY
        )
        a_n = p.negative.specific_surface_area_m2_m3
        a_p = p.positive.specific_surface_area_m2_m3

        def residual(unknown: np.ndarray) -> np.ndarray:
            phi_n = unknown[:n_negative]
            phi_p = unknown[n_negative : n_negative + n_positive]
            phi_e = unknown[n_negative + n_positive :]
            eta_n = phi_n - phi_e[grid.negative] - ocp_n
            eta_p = phi_p - phi_e[grid.positive] - ocp_p
            j_n = self._butler_volmer(
                i0_n, eta_n, p.negative.charge_transfer_coefficient, temperature_K
            )
            j_p = self._butler_volmer(
                i0_p, eta_p, p.positive.charge_transfer_coefficient, temperature_K
            )
            i_s_n = _phase_current_faces(
                phi_n,
                np.full(n_negative, sigma_n),
                grid.widths_m[grid.negative],
                current_density,
                0.0,
            )
            i_s_p = _phase_current_faces(
                phi_p,
                np.full(n_positive, sigma_p),
                grid.widths_m[grid.positive],
                0.0,
                current_density,
            )
            solid_n = np.diff(i_s_n) + a_n * j_n * grid.widths_m[grid.negative]
            solid_p = np.diff(i_s_p) + a_p * j_p * grid.widths_m[grid.positive]
            ionic_faces = np.zeros(total + 1)
            face_kappa = harmonic_face_coefficients(kappa_cells, grid.widths_m)
            delta_phi = np.diff(phi_e)
            delta_log_c = np.diff(np.log(concentration))
            ionic_faces[1:-1] = face_kappa * (-delta_phi + beta * delta_log_c)
            reaction = np.zeros(total)
            reaction[grid.negative] = a_n * j_n
            reaction[grid.positive] = a_p * j_p
            electrolyte = np.diff(ionic_faces) - reaction * grid.widths_m
            output = np.concatenate(
                (solid_n / current_scale, solid_p / current_scale, electrolyte / current_scale)
            )
            output[n_negative + n_positive] = phi_e[0] / 0.1
            return output

        solution = least_squares(
            residual,
            np.asarray(initial, dtype=float),
            xtol=p.nonlinear_tolerance,
            ftol=p.nonlinear_tolerance,
            gtol=p.nonlinear_tolerance,
            max_nfev=p.nonlinear_max_evaluations,
            x_scale="jac",
        )
        final_residual = residual(solution.x)
        residual_norm = float(np.linalg.norm(final_residual, ord=np.inf))
        acceptance = max(50.0 * p.nonlinear_tolerance, 1.0e-6)
        if not solution.success or residual_norm > acceptance:
            raise ConvergenceError(
                "P2D charge-conservation solve failed: "
                f"{solution.message}; infinity residual={residual_norm:.3e}"
            )
        phi_n = solution.x[:n_negative]
        phi_p = solution.x[n_negative : n_negative + n_positive]
        phi_e = solution.x[n_negative + n_positive :]
        j_n = self._butler_volmer(
            i0_n,
            phi_n - phi_e[grid.negative] - ocp_n,
            p.negative.charge_transfer_coefficient,
            temperature_K,
        )
        j_p = self._butler_volmer(
            i0_p,
            phi_p - phi_e[grid.positive] - ocp_p,
            p.positive.charge_transfer_coefficient,
            temperature_K,
        )
        return _PotentialResult(
            phi_s_negative=phi_n,
            phi_s_positive=phi_p,
            phi_electrolyte=phi_e,
            j_negative=j_n,
            j_positive=j_p,
            evaluations=int(solution.nfev),
            residual_norm=residual_norm,
        )

    def _exchange_current(
        self,
        electrode: ElectrodeParameters,
        electrolyte_concentration: np.ndarray,
        surface_concentration: np.ndarray,
        soc: float,
        temperature_K: float,
    ) -> np.ndarray:
        alpha = electrode.charge_transfer_coefficient
        maximum = electrode.maximum_concentration_mol_m3
        surface = np.clip(surface_concentration, 1.0e-12 * maximum, (1.0 - 1.0e-12) * maximum)
        electrolyte = np.maximum(electrolyte_concentration, 1.0e-12)
        return (
            FARADAY
            * electrode.reaction_rate(soc, temperature_K)
            * electrolyte**alpha
            * (maximum - surface) ** alpha
            * surface ** (1.0 - alpha)
        )

    @staticmethod
    def _butler_volmer(
        exchange_current: np.ndarray,
        overpotential_V: np.ndarray,
        alpha: float,
        temperature_K: float,
    ) -> np.ndarray:
        scaled = FARADAY * np.asarray(overpotential_V) / (GAS_CONSTANT * temperature_K)
        anodic = np.exp(np.clip(alpha * scaled, -80.0, 80.0))
        cathodic = np.exp(np.clip(-(1.0 - alpha) * scaled, -80.0, 80.0))
        return exchange_current * (anodic - cathodic)

    def _electrolyte_structure(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.parameters
        porosity = np.concatenate(
            (
                np.full(p.discretization.negative_cells, p.negative.porosity),
                np.full(p.discretization.separator_cells, p.separator.porosity),
                np.full(p.discretization.positive_cells, p.positive.porosity),
            )
        )
        bruggeman = np.concatenate(
            (
                np.full(p.discretization.negative_cells, p.negative.bruggeman),
                np.full(p.discretization.separator_cells, p.separator.bruggeman),
                np.full(p.discretization.positive_cells, p.positive.bruggeman),
            )
        )
        return porosity, bruggeman

    def _electrolyte_step(
        self,
        previous: np.ndarray,
        j_negative: np.ndarray,
        j_positive: np.ndarray,
        dt_s: float,
        soc: float,
        temperature_K: float,
    ) -> np.ndarray:
        p = self.parameters
        porosity, bruggeman = self._electrolyte_structure()
        diffusivity = evaluate_property(
            p.electrolyte.diffusivity_m2_s,
            soc,
            temperature_K,
            name="electrolyte.diffusivity_m2_s",
        )
        effective_diffusivity = diffusivity * porosity**bruggeman
        source = np.zeros_like(previous)
        source[self.grid.negative] = (
            (1.0 - p.electrolyte.transference_number)
            * p.negative.specific_surface_area_m2_m3
            * j_negative
            / FARADAY
        )
        source[self.grid.positive] = (
            (1.0 - p.electrolyte.transference_number)
            * p.positive.specific_surface_area_m2_m3
            * j_positive
            / FARADAY
        )
        return conservative_diffusion_step(
            previous,
            dt_s,
            self.grid.widths_m,
            porosity,
            effective_diffusivity,
            source,
        )

    def _temperature_step(
        self,
        old_state: P2DState,
        current_A: float,
        voltage_V: float,
        negative_particles: np.ndarray,
        positive_particles: np.ndarray,
        dt_s: float,
    ) -> float:
        p = self.parameters
        theta_n = float(
            np.mean(negative_particles[:, -1])
            / p.negative.maximum_concentration_mol_m3
        )
        theta_p = float(
            np.mean(positive_particles[:, -1])
            / p.positive.maximum_concentration_mol_m3
        )
        ocv = float(
            p.positive.ocp_V(theta_p, old_state.temperature_K)
            - p.negative.ocp_V(theta_n, old_state.temperature_K)
        )
        entropic = float(
            p.positive.entropic_coefficient_V_K(theta_p, old_state.temperature_K)
            - p.negative.entropic_coefficient_V_K(theta_n, old_state.temperature_K)
        )
        irreversible_heat_W = current_A * (ocv - voltage_V)
        reversible_heat_W = -current_A * old_state.temperature_K * entropic
        cooling_W = (
            p.thermal.heat_transfer_coefficient_W_m2_K
            * p.thermal.cooling_area_m2
            * (old_state.temperature_K - p.thermal.ambient_temperature_K)
        )
        thermal_capacity = (
            p.thermal.density_heat_capacity_J_m3_K
            * p.area_m2
            * p.thickness_m
        )
        if thermal_capacity <= 0.0:
            return old_state.temperature_K
        updated = old_state.temperature_K + dt_s * (
            irreversible_heat_W + reversible_heat_W - cooling_W
        ) / thermal_capacity
        if not np.isfinite(updated) or updated <= 0.0:
            raise ConvergenceError("P2D thermal update produced an invalid temperature")
        return float(updated)

    def _validate_concentrations(
        self,
        electrolyte: np.ndarray,
        negative: np.ndarray,
        positive: np.ndarray,
    ) -> None:
        p = self.parameters
        if np.min(electrolyte) <= p.minimum_concentration_mol_m3:
            raise ConvergenceError(
                "electrolyte concentration depleted; reduce the time step/current"
            )
        for name, values, maximum in (
            ("negative", negative, p.negative.maximum_concentration_mol_m3),
            ("positive", positive, p.positive.maximum_concentration_mol_m3),
        ):
            if np.min(values) <= 0.0 or np.max(values) >= maximum:
                raise ConvergenceError(
                    f"{name} particle concentration left (0, c_max); "
                    "reduce the time step/current or stop at a stoichiometric cutoff"
                )

    def _inventory_errors(
        self,
        old_electrolyte: np.ndarray,
        new_electrolyte: np.ndarray,
        old_negative: np.ndarray,
        new_negative: np.ndarray,
        old_positive: np.ndarray,
        new_positive: np.ndarray,
        j_negative: np.ndarray,
        j_positive: np.ndarray,
        dt_s: float,
    ) -> tuple[float, float, float]:
        p = self.parameters
        porosity, _ = self._electrolyte_structure()
        old_e = float(np.sum(porosity * self.grid.widths_m * old_electrolyte))
        new_e = float(np.sum(porosity * self.grid.widths_m * new_electrolyte))
        source = (
            (1.0 - p.electrolyte.transference_number)
            / FARADAY
            * (
                p.negative.specific_surface_area_m2_m3
                * np.sum(j_negative * self.grid.widths_m[self.grid.negative])
                + p.positive.specific_surface_area_m2_m3
                * np.sum(j_positive * self.grid.widths_m[self.grid.positive])
            )
        )
        error_e = abs(new_e - old_e - dt_s * source) / max(abs(old_e), 1.0e-30)

        def particle_error(
            old: np.ndarray,
            new: np.ndarray,
            radius: float,
            current: np.ndarray,
        ) -> float:
            old_inventory = spherical_inventory(old, radius)
            new_inventory = spherical_inventory(new, radius)
            expected_change = -dt_s * radius**2 * current / FARADAY
            mismatch = new_inventory - old_inventory - expected_change
            return float(
                np.max(np.abs(mismatch) / np.maximum(np.abs(old_inventory), 1.0e-30))
            )

        return (
            float(error_e),
            particle_error(
                old_negative,
                new_negative,
                p.negative.particle_radius_m,
                j_negative,
            ),
            particle_error(
                old_positive,
                new_positive,
                p.positive.particle_radius_m,
                j_positive,
            ),
        )


def _phase_current_faces(
    potential: np.ndarray,
    conductivity: np.ndarray,
    widths: np.ndarray,
    left_boundary_current_A_m2: float,
    right_boundary_current_A_m2: float,
) -> np.ndarray:
    faces = np.empty(len(potential) + 1, dtype=np.float64)
    faces[0] = left_boundary_current_A_m2
    faces[-1] = right_boundary_current_A_m2
    face_conductance = harmonic_face_coefficients(conductivity, widths)
    faces[1:-1] = -face_conductance * np.diff(potential)
    return faces


def _relative_change(new: np.ndarray, old: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(new)), float(np.linalg.norm(old)), 1.0)
    return float(np.linalg.norm(new - old) / scale)


__all__ = ["MaterialUpdate", "P2DModel"]
