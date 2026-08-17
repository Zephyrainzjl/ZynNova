"""Three-dimensional porous-electrode FEM with embedded spherical particles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...constants import FARADAY, GAS_CONSTANT
from ...core import DirichletBC, Mesh, SurfaceLoad, box_tetrahedral_mesh
from ...core.linalg import LinearSolveOptions, solve_linear
from ...exceptions import ConvergenceError
from ...fem import (
    LinearElasticityProblem,
    assemble_scalar,
    assemble_scalar_source,
)
from ...fem.assembly import (
    assemble_surface_load,
    cell_gradients,
    triangle_areas,
)
from ..properties import evaluate_property
from ..p2d.numerics import spherical_average, spherical_particle_step
from ..p2d.parameters import P2DParameters


def _sparse_modules() -> tuple[Any, Any]:
    try:
        import scipy.sparse as sparse
        import scipy.sparse.linalg as sparse_linalg
    except ImportError as exc:
        raise ImportError("3-D battery FEM requires SciPy; install zynnova[battery]") from exc
    return sparse, sparse_linalg


@dataclass(slots=True)
class Battery3DConfig:
    negative_region: int = 0
    separator_region: int = 1
    positive_region: int = 2
    negative_collector_boundary: str = "xmin"
    positive_collector_boundary: str = "xmax"
    particle_cells: int = 12
    potential_tolerance: float = 1.0e-8
    potential_max_evaluations: int = 400
    thermal_conductivity_W_m_K: tuple[float, float, float] = (1.2, 0.4, 1.2)
    young_modulus_Pa: tuple[float, float, float] = (1.0e9, 0.2e9, 1.5e9)
    poisson_ratio: tuple[float, float, float] = (0.30, 0.35, 0.30)
    chemical_expansion: tuple[float, float] = (0.06, 0.02)
    mechanics_boundary: str | None = None
    backend: str = "auto"

    def __post_init__(self) -> None:
        if len({self.negative_region, self.separator_region, self.positive_region}) != 3:
            raise ValueError("3-D battery region identifiers must be distinct")
        if self.particle_cells < 2:
            raise ValueError("particle_cells must be at least two")
        if self.potential_tolerance <= 0.0 or self.potential_max_evaluations < 1:
            raise ValueError("3-D nonlinear solver controls are invalid")


@dataclass(slots=True)
class Battery3DDiagnostics:
    converged: bool
    potential_evaluations: int
    potential_residual_norm: float
    electrolyte_inventory_error: float
    message: str


@dataclass(slots=True)
class Battery3DState:
    time_s: float
    electrolyte_concentration_mol_m3: np.ndarray
    electrolyte_potential_V: np.ndarray
    solid_potential_V: np.ndarray
    negative_particle_concentration_mol_m3: np.ndarray
    positive_particle_concentration_mol_m3: np.ndarray
    interfacial_current_A_m2: np.ndarray
    temperature_K: np.ndarray
    displacement_m: np.ndarray
    terminal_voltage_V: float
    current_A: float = 0.0
    diagnostics: Battery3DDiagnostics | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def copy(self) -> Battery3DState:
        return Battery3DState(
            time_s=float(self.time_s),
            electrolyte_concentration_mol_m3=(
                self.electrolyte_concentration_mol_m3.copy()
            ),
            electrolyte_potential_V=self.electrolyte_potential_V.copy(),
            solid_potential_V=self.solid_potential_V.copy(),
            negative_particle_concentration_mol_m3=(
                self.negative_particle_concentration_mol_m3.copy()
            ),
            positive_particle_concentration_mol_m3=(
                self.positive_particle_concentration_mol_m3.copy()
            ),
            interfacial_current_A_m2=self.interfacial_current_A_m2.copy(),
            temperature_K=self.temperature_K.copy(),
            displacement_m=self.displacement_m.copy(),
            terminal_voltage_V=float(self.terminal_voltage_V),
            current_A=float(self.current_A),
            diagnostics=self.diagnostics,
            metadata=self.metadata.copy(),
        )


def layered_battery_mesh(
    lengths_m: tuple[float, float, float],
    shape: tuple[int, int, int],
    layer_thicknesses_m: tuple[float, float, float],
    *,
    region_ids: tuple[int, int, int] = (0, 1, 2),
) -> Mesh:
    """Generate a box Tet4 mesh and label negative/separator/positive layers."""

    if not np.isclose(sum(layer_thicknesses_m), lengths_m[0], rtol=1.0e-10, atol=1.0e-15):
        raise ValueError("layer thicknesses must sum to the x length")
    if shape[0] < 3:
        raise ValueError("layered battery mesh needs at least three x-direction cells")
    remaining = shape[0] - 3
    proportions = np.asarray(layer_thicknesses_m, dtype=float) / sum(layer_thicknesses_m)
    allocation = proportions * remaining
    layer_cells = np.floor(allocation).astype(int) + 1
    unassigned = shape[0] - int(layer_cells.sum())
    for index in np.argsort(-(allocation - np.floor(allocation)))[:unassigned]:
        layer_cells[index] += 1
    x_edges = [0.0]
    offset = 0.0
    for thickness, count in zip(layer_thicknesses_m, layer_cells, strict=True):
        local = offset + np.linspace(0.0, thickness, int(count) + 1)[1:]
        x_edges.extend(local.tolist())
        offset += thickness
    mesh = box_tetrahedral_mesh(lengths_m, shape)
    uniform_levels = np.linspace(0.0, lengths_m[0], shape[0] + 1)
    remapped_nodes = mesh.nodes.copy()
    level_indices = np.argmin(
        np.abs(remapped_nodes[:, 0, None] - uniform_levels[None, :]), axis=1
    )
    remapped_nodes[:, 0] = np.asarray(x_edges)[level_indices]
    mesh = Mesh(
        nodes=remapped_nodes,
        cells=mesh.cells,
        boundary_faces=mesh.boundary_faces,
    )
    centers = mesh.cell_centers()[:, 0]
    first = layer_thicknesses_m[0]
    second = first + layer_thicknesses_m[1]
    regions = np.where(
        centers < first,
        region_ids[0],
        np.where(centers < second, region_ids[1], region_ids[2]),
    ).astype(np.int32)
    if any(not np.any(regions == region) for region in region_ids):
        raise ValueError("mesh resolution does not place cells in every battery layer")
    return Mesh(
        nodes=mesh.nodes,
        cells=mesh.cells,
        cell_regions=regions,
        boundary_faces=mesh.boundary_faces,
        metadata={"layer_thicknesses_m": tuple(map(float, layer_thicknesses_m))},
    )


class PorousElectrode3D:
    """Macrohomogeneous 3-D porous electrode with per-cell radial particles."""

    def __init__(
        self,
        mesh: Mesh,
        parameters: P2DParameters,
        config: Battery3DConfig | None = None,
    ) -> None:
        self.mesh = mesh
        self.parameters = parameters
        self.config = config or Battery3DConfig()
        self.negative_cells = mesh.region_cells(self.config.negative_region)
        self.separator_cells = mesh.region_cells(self.config.separator_region)
        self.positive_cells = mesh.region_cells(self.config.positive_region)
        if min(len(self.negative_cells), len(self.separator_cells), len(self.positive_cells)) < 1:
            raise ValueError("3-D mesh must contain negative, separator, and positive regions")
        self.negative_nodes = mesh.region_nodes(self.config.negative_region)
        self.positive_nodes = mesh.region_nodes(self.config.positive_region)
        overlap = np.intersect1d(self.negative_nodes, self.positive_nodes)
        if overlap.size:
            raise ValueError("negative and positive solid phases share mesh nodes")
        self.active_nodes = np.concatenate((self.negative_nodes, self.positive_nodes))
        self.active_lookup = np.full(mesh.n_nodes, -1, dtype=np.int64)
        self.active_lookup[self.active_nodes] = np.arange(len(self.active_nodes))
        for name in (
            self.config.negative_collector_boundary,
            self.config.positive_collector_boundary,
        ):
            mesh.boundary_nodes(name)

    def initialize(self, soc: float = 1.0, *, temperature_K: float | None = None) -> Battery3DState:
        if not 0.0 <= soc <= 1.0:
            raise ValueError("SOC must lie in [0, 1]")
        p = self.parameters
        temperature = p.initial_temperature_K if temperature_K is None else float(temperature_K)
        theta_n = p.negative.stoichiometry(soc)
        theta_p = p.positive.stoichiometry(soc)
        c_e = np.full(self.mesh.n_nodes, p.electrolyte.initial_concentration_mol_m3)
        c_n = np.full(
            (len(self.negative_cells), self.config.particle_cells),
            theta_n * p.negative.maximum_concentration_mol_m3,
        )
        c_p = np.full(
            (len(self.positive_cells), self.config.particle_cells),
            theta_p * p.positive.maximum_concentration_mol_m3,
        )
        solid = np.zeros(self.mesh.n_nodes)
        solid[self.negative_nodes] = float(p.negative.ocp_V(theta_n, temperature))
        solid[self.positive_nodes] = float(p.positive.ocp_V(theta_p, temperature))
        voltage = self._terminal_voltage(solid)
        return Battery3DState(
            time_s=0.0,
            electrolyte_concentration_mol_m3=c_e,
            electrolyte_potential_V=np.zeros(self.mesh.n_nodes),
            solid_potential_V=solid,
            negative_particle_concentration_mol_m3=c_n,
            positive_particle_concentration_mol_m3=c_p,
            interfacial_current_A_m2=np.zeros(self.mesh.n_cells),
            temperature_K=np.full(self.mesh.n_nodes, temperature),
            displacement_m=np.zeros((self.mesh.n_nodes, 3)),
            terminal_voltage_V=voltage,
            metadata={
                "initial_negative_stoichiometry": theta_n,
                "initial_positive_stoichiometry": theta_p,
            },
        )

    def step(self, state: Battery3DState, current_A: float, dt_s: float) -> Battery3DState:
        if dt_s <= 0.0 or not np.isfinite(current_A):
            raise ValueError("time step must be positive and current finite")
        p = self.parameters
        temperature = float(np.mean(state.temperature_K))
        soc = self._soc(state)
        potential = self._solve_potentials(state, current_A, soc, temperature)
        j_negative = potential[2][self.negative_cells]
        j_positive = potential[2][self.positive_cells]
        c_n = spherical_particle_step(
            state.negative_particle_concentration_mol_m3,
            dt_s,
            p.negative.particle_radius_m,
            p.negative.diffusivity(soc, temperature),
            j_negative,
            FARADAY,
        )
        c_p = spherical_particle_step(
            state.positive_particle_concentration_mol_m3,
            dt_s,
            p.positive.particle_radius_m,
            p.positive.diffusivity(soc, temperature),
            j_positive,
            FARADAY,
        )
        c_e, inventory_error = self._electrolyte_step(
            state.electrolyte_concentration_mol_m3,
            potential[2],
            dt_s,
            soc,
            temperature,
        )
        if (
            np.min(c_e) <= p.minimum_concentration_mol_m3
            or np.min(c_n) <= 0.0
            or np.max(c_n) >= p.negative.maximum_concentration_mol_m3
            or np.min(c_p) <= 0.0
            or np.max(c_p) >= p.positive.maximum_concentration_mol_m3
        ):
            raise ConvergenceError("3-D electrochemical concentration bounds were violated")
        temperature_field = self._thermal_step(
            state,
            potential[0],
            potential[1],
            potential[2],
            current_A,
            dt_s,
            soc,
            temperature,
        )
        displacement = self._mechanics_step(state, c_n, c_p)
        solid_full = np.zeros(self.mesh.n_nodes)
        solid_full[self.active_nodes] = potential[0]
        voltage = self._terminal_voltage(solid_full)
        return Battery3DState(
            time_s=state.time_s + dt_s,
            electrolyte_concentration_mol_m3=c_e,
            electrolyte_potential_V=potential[1],
            solid_potential_V=solid_full,
            negative_particle_concentration_mol_m3=c_n,
            positive_particle_concentration_mol_m3=c_p,
            interfacial_current_A_m2=potential[2],
            temperature_K=temperature_field,
            displacement_m=displacement,
            terminal_voltage_V=voltage,
            current_A=float(current_A),
            diagnostics=Battery3DDiagnostics(
                converged=True,
                potential_evaluations=potential[3],
                potential_residual_norm=potential[4],
                electrolyte_inventory_error=inventory_error,
                message="3-D FEM charge solve with embedded implicit particle diffusion",
            ),
            metadata=state.metadata.copy(),
        )

    def _solve_potentials(
        self,
        state: Battery3DState,
        current_A: float,
        soc: float,
        temperature_K: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
        try:
            from scipy.optimize import least_squares
        except ImportError as exc:
            raise ImportError("3-D charge solve requires SciPy") from exc
        p = self.parameters
        cell_porosity, bruggeman = self._cell_structure()
        kappa = evaluate_property(
            p.electrolyte.ionic_conductivity_S_m,
            soc,
            temperature_K,
            name="electrolyte.ionic_conductivity_S_m",
        )
        kappa_cells = kappa * cell_porosity**bruggeman
        electrolyte_k = assemble_scalar(
            self.mesh, kappa_cells, 1.0, backend=self.config.backend
        ).stiffness
        factor = evaluate_property(
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
            * factor
            / FARADAY
        )
        concentration_k = assemble_scalar(
            self.mesh, beta * kappa_cells, 1.0, backend=self.config.backend
        ).stiffness
        sigma_cells = np.zeros(self.mesh.n_cells)
        sigma_cells[self.negative_cells] = (
            p.negative.conductivity(soc, temperature_K)
            * p.negative.active_volume_fraction ** p.negative.bruggeman
        )
        sigma_cells[self.positive_cells] = (
            p.positive.conductivity(soc, temperature_K)
            * p.positive.active_volume_fraction ** p.positive.bruggeman
        )
        solid_full_k = assemble_scalar(
            self.mesh, sigma_cells, 1.0, backend=self.config.backend
        ).stiffness
        solid_k = solid_full_k[self.active_nodes][:, self.active_nodes]
        boundary_load = self._solid_boundary_load(current_A)[self.active_nodes]
        c_e_cell = state.electrolyte_concentration_mol_m3[self.mesh.cells].mean(axis=1)
        theta_n = np.clip(
            state.negative_particle_concentration_mol_m3[:, -1]
            / p.negative.maximum_concentration_mol_m3,
            1.0e-8,
            1.0 - 1.0e-8,
        )
        theta_p = np.clip(
            state.positive_particle_concentration_mol_m3[:, -1]
            / p.positive.maximum_concentration_mol_m3,
            1.0e-8,
            1.0 - 1.0e-8,
        )
        ocp_n = np.asarray(p.negative.ocp_V(theta_n, temperature_K))
        ocp_p = np.asarray(p.positive.ocp_V(theta_p, temperature_K))
        i0_n = _exchange_current(
            p.negative.reaction_rate(soc, temperature_K),
            p.negative.charge_transfer_coefficient,
            c_e_cell[self.negative_cells],
            theta_n * p.negative.maximum_concentration_mol_m3,
            p.negative.maximum_concentration_mol_m3,
        )
        i0_p = _exchange_current(
            p.positive.reaction_rate(soc, temperature_K),
            p.positive.charge_transfer_coefficient,
            c_e_cell[self.positive_cells],
            theta_p * p.positive.maximum_concentration_mol_m3,
            p.positive.maximum_concentration_mol_m3,
        )
        cell_volumes = self.mesh.cell_volumes()
        current_scale = max(abs(current_A), 1.0)
        initial = np.concatenate(
            (
                state.solid_potential_V[self.active_nodes],
                state.electrolyte_potential_V,
            )
        )
        log_concentration = np.log(
            np.maximum(
                state.electrolyte_concentration_mol_m3,
                p.minimum_concentration_mol_m3,
            )
        )

        def residual(unknown: np.ndarray) -> np.ndarray:
            phi_s = unknown[: len(self.active_nodes)]
            phi_e = unknown[len(self.active_nodes) :]
            j = self._reaction_currents(
                phi_s, phi_e, ocp_n, ocp_p, i0_n, i0_p, temperature_K
            )
            reaction_load = self._reaction_load(j)
            solid_residual = (
                solid_k @ phi_s
                + reaction_load[self.active_nodes]
                - boundary_load
            ) / current_scale
            electrolyte_residual = (
                electrolyte_k @ phi_e
                - reaction_load
                - concentration_k @ log_concentration
            ) / current_scale
            electrolyte_residual = np.asarray(electrolyte_residual).reshape(-1)
            electrolyte_residual[0] = phi_e[0] / 0.1
            return np.concatenate((np.asarray(solid_residual).reshape(-1), electrolyte_residual))

        solution = least_squares(
            residual,
            initial,
            xtol=self.config.potential_tolerance,
            ftol=self.config.potential_tolerance,
            gtol=self.config.potential_tolerance,
            max_nfev=self.config.potential_max_evaluations,
            x_scale="jac",
        )
        final = residual(solution.x)
        norm = float(np.linalg.norm(final, ord=np.inf))
        if not solution.success or norm > max(1.0e-6, 50.0 * self.config.potential_tolerance):
            raise ConvergenceError(
                f"3-D battery charge solve failed: {solution.message}; residual={norm:.3e}"
            )
        phi_s = solution.x[: len(self.active_nodes)]
        phi_e = solution.x[len(self.active_nodes) :]
        current = self._reaction_currents(
            phi_s, phi_e, ocp_n, ocp_p, i0_n, i0_p, temperature_K
        )
        return phi_s, phi_e, current, int(solution.nfev), norm

    def _reaction_currents(
        self,
        phi_s_active: np.ndarray,
        phi_e: np.ndarray,
        ocp_n: np.ndarray,
        ocp_p: np.ndarray,
        i0_n: np.ndarray,
        i0_p: np.ndarray,
        temperature_K: float,
    ) -> np.ndarray:
        p = self.parameters
        result = np.zeros(self.mesh.n_cells)

        def phase(
            cells: np.ndarray,
            ocp: np.ndarray,
            exchange: np.ndarray,
            alpha: float,
        ) -> np.ndarray:
            solid = np.asarray(
                [
                    np.mean(phi_s_active[self.active_lookup[self.mesh.cells[cell]]])
                    for cell in cells
                ]
            )
            electrolyte = phi_e[self.mesh.cells[cells]].mean(axis=1)
            scaled = FARADAY * (solid - electrolyte - ocp) / (
                GAS_CONSTANT * temperature_K
            )
            return exchange * (
                np.exp(np.clip(alpha * scaled, -80.0, 80.0))
                - np.exp(np.clip(-(1.0 - alpha) * scaled, -80.0, 80.0))
            )

        result[self.negative_cells] = phase(
            self.negative_cells,
            ocp_n,
            i0_n,
            p.negative.charge_transfer_coefficient,
        )
        result[self.positive_cells] = phase(
            self.positive_cells,
            ocp_p,
            i0_p,
            p.positive.charge_transfer_coefficient,
        )
        return result

    def _reaction_load(self, interfacial_current: np.ndarray) -> np.ndarray:
        p = self.parameters
        volumetric = np.zeros(self.mesh.n_cells)
        volumetric[self.negative_cells] = (
            p.negative.specific_surface_area_m2_m3
            * interfacial_current[self.negative_cells]
        )
        volumetric[self.positive_cells] = (
            p.positive.specific_surface_area_m2_m3
            * interfacial_current[self.positive_cells]
        )
        return assemble_scalar_source(self.mesh, volumetric)

    def _solid_boundary_load(self, current_A: float) -> np.ndarray:
        negative_faces = self.mesh.boundary_faces[
            self.config.negative_collector_boundary
        ]
        positive_faces = self.mesh.boundary_faces[
            self.config.positive_collector_boundary
        ]
        negative_area = float(np.sum(triangle_areas(self.mesh.nodes, negative_faces)))
        positive_area = float(np.sum(triangle_areas(self.mesh.nodes, positive_faces)))
        if negative_area <= 0.0 or positive_area <= 0.0:
            raise ValueError("current-collector boundary has zero area")
        result = assemble_surface_load(
            self.mesh,
            SurfaceLoad(negative_faces, np.asarray(current_A / negative_area)),
            vector=False,
        )
        result += assemble_surface_load(
            self.mesh,
            SurfaceLoad(positive_faces, np.asarray(-current_A / positive_area)),
            vector=False,
        )
        return result

    def _cell_structure(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.parameters
        porosity = np.empty(self.mesh.n_cells)
        bruggeman = np.empty(self.mesh.n_cells)
        porosity[self.negative_cells] = p.negative.porosity
        porosity[self.separator_cells] = p.separator.porosity
        porosity[self.positive_cells] = p.positive.porosity
        bruggeman[self.negative_cells] = p.negative.bruggeman
        bruggeman[self.separator_cells] = p.separator.bruggeman
        bruggeman[self.positive_cells] = p.positive.bruggeman
        return porosity, bruggeman

    def _electrolyte_step(
        self,
        previous: np.ndarray,
        interfacial_current: np.ndarray,
        dt_s: float,
        soc: float,
        temperature_K: float,
    ) -> tuple[np.ndarray, float]:
        sparse, sparse_linalg = _sparse_modules()
        p = self.parameters
        porosity, bruggeman = self._cell_structure()
        diffusivity = evaluate_property(
            p.electrolyte.diffusivity_m2_s,
            soc,
            temperature_K,
            name="electrolyte.diffusivity_m2_s",
        )
        operators = assemble_scalar(
            self.mesh,
            diffusivity * porosity**bruggeman,
            porosity,
            backend=self.config.backend,
        )
        source = np.zeros(self.mesh.n_cells)
        source[self.negative_cells] = (
            (1.0 - p.electrolyte.transference_number)
            * p.negative.specific_surface_area_m2_m3
            * interfacial_current[self.negative_cells]
            / FARADAY
        )
        source[self.positive_cells] = (
            (1.0 - p.electrolyte.transference_number)
            * p.positive.specific_surface_area_m2_m3
            * interfacial_current[self.positive_cells]
            / FARADAY
        )
        load = assemble_scalar_source(self.mesh, source)
        matrix = operators.mass + dt_s * operators.stiffness
        rhs = operators.mass @ previous + dt_s * load
        updated = np.asarray(sparse_linalg.spsolve(sparse.csr_matrix(matrix), rhs))
        old_inventory = float(np.sum(operators.mass @ previous))
        new_inventory = float(np.sum(operators.mass @ updated))
        expected = dt_s * float(np.sum(load))
        error = abs(new_inventory - old_inventory - expected) / max(
            abs(old_inventory), 1.0e-30
        )
        return updated, float(error)

    def _thermal_step(
        self,
        state: Battery3DState,
        phi_s_active: np.ndarray,
        phi_e: np.ndarray,
        interfacial_current: np.ndarray,
        current_A: float,
        dt_s: float,
        soc: float,
        temperature_K: float,
    ) -> np.ndarray:
        sparse, sparse_linalg = _sparse_modules()
        p = self.parameters
        conductivity = self._region_values(self.config.thermal_conductivity_W_m_K)
        capacity = np.full(
            self.mesh.n_cells, p.thermal.density_heat_capacity_J_m3_K
        )
        operators = assemble_scalar(
            self.mesh,
            conductivity,
            capacity,
            backend=self.config.backend,
        )
        solid_full = np.zeros(self.mesh.n_nodes)
        solid_full[self.active_nodes] = phi_s_active
        grad_s = cell_gradients(self.mesh, solid_full)
        grad_e = cell_gradients(self.mesh, phi_e)
        cell_porosity, bruggeman = self._cell_structure()
        kappa = evaluate_property(
            p.electrolyte.ionic_conductivity_S_m,
            soc,
            temperature_K,
            name="electrolyte.ionic_conductivity_S_m",
        )
        sigma = np.zeros(self.mesh.n_cells)
        sigma[self.negative_cells] = (
            p.negative.conductivity(soc, temperature_K)
            * p.negative.active_volume_fraction ** p.negative.bruggeman
        )
        sigma[self.positive_cells] = (
            p.positive.conductivity(soc, temperature_K)
            * p.positive.active_volume_fraction ** p.positive.bruggeman
        )
        heat = sigma * np.sum(grad_s * grad_s, axis=1)
        heat += kappa * cell_porosity**bruggeman * np.sum(grad_e * grad_e, axis=1)
        phi_s_cell = solid_full[self.mesh.cells].mean(axis=1)
        phi_e_cell = phi_e[self.mesh.cells].mean(axis=1)
        theta_n = state.negative_particle_concentration_mol_m3[:, -1] / (
            p.negative.maximum_concentration_mol_m3
        )
        theta_p = state.positive_particle_concentration_mol_m3[:, -1] / (
            p.positive.maximum_concentration_mol_m3
        )
        overpotential = np.zeros(self.mesh.n_cells)
        overpotential[self.negative_cells] = (
            phi_s_cell[self.negative_cells]
            - phi_e_cell[self.negative_cells]
            - p.negative.ocp_V(theta_n, temperature_K)
        )
        overpotential[self.positive_cells] = (
            phi_s_cell[self.positive_cells]
            - phi_e_cell[self.positive_cells]
            - p.positive.ocp_V(theta_p, temperature_K)
        )
        heat[self.negative_cells] += (
            p.negative.specific_surface_area_m2_m3
            * interfacial_current[self.negative_cells]
            * overpotential[self.negative_cells]
        )
        heat[self.positive_cells] += (
            p.positive.specific_surface_area_m2_m3
            * interfacial_current[self.positive_cells]
            * overpotential[self.positive_cells]
        )
        volume = float(np.sum(self.mesh.cell_volumes()))
        cooling = (
            p.thermal.heat_transfer_coefficient_W_m2_K
            * p.thermal.cooling_area_m2
            * (float(np.mean(state.temperature_K)) - p.thermal.ambient_temperature_K)
        )
        heat -= cooling / max(volume, 1.0e-30)
        load = assemble_scalar_source(self.mesh, heat)
        matrix = operators.mass + dt_s * operators.stiffness
        rhs = operators.mass @ state.temperature_K + dt_s * load
        updated = np.asarray(sparse_linalg.spsolve(sparse.csr_matrix(matrix), rhs))
        if np.min(updated) <= 0.0 or not np.all(np.isfinite(updated)):
            raise ConvergenceError("3-D thermal solve produced an invalid temperature")
        return updated

    def _mechanics_step(
        self,
        state: Battery3DState,
        negative_concentration: np.ndarray,
        positive_concentration: np.ndarray,
    ) -> np.ndarray:
        if self.config.mechanics_boundary is None:
            return state.displacement_m.copy()
        p = self.parameters
        young = self._region_values(self.config.young_modulus_Pa)
        poisson = self._region_values(self.config.poisson_ratio)
        eigenstrain = np.zeros((self.mesh.n_cells, 6))
        initial_n = float(state.metadata["initial_negative_stoichiometry"])
        initial_p = float(state.metadata["initial_positive_stoichiometry"])
        theta_n = spherical_average(negative_concentration) / (
            p.negative.maximum_concentration_mol_m3
        )
        theta_p = spherical_average(positive_concentration) / (
            p.positive.maximum_concentration_mol_m3
        )
        expansion_n = self.config.chemical_expansion[0] * (theta_n - initial_n)
        expansion_p = self.config.chemical_expansion[1] * (theta_p - initial_p)
        eigenstrain[self.negative_cells, :3] = expansion_n[:, None]
        eigenstrain[self.positive_cells, :3] = expansion_p[:, None]
        fixed = DirichletBC.vector(
            self.mesh.boundary_nodes(self.config.mechanics_boundary), value=0.0
        )
        solution = LinearElasticityProblem(
            self.mesh,
            young,
            poisson,
            dirichlet=(fixed,),
            eigenstrain=eigenstrain,
            backend=self.config.backend,
            linear_options=LinearSolveOptions(method="direct"),
        ).solve()
        return solution.displacement

    def _region_values(self, values: tuple[float, float, float]) -> np.ndarray:
        result = np.empty(self.mesh.n_cells)
        result[self.negative_cells] = values[0]
        result[self.separator_cells] = values[1]
        result[self.positive_cells] = values[2]
        return result

    def _terminal_voltage(self, solid_potential: np.ndarray) -> float:
        negative = self.mesh.boundary_nodes(self.config.negative_collector_boundary)
        positive = self.mesh.boundary_nodes(self.config.positive_collector_boundary)
        return float(np.mean(solid_potential[positive]) - np.mean(solid_potential[negative]))

    def _soc(self, state: Battery3DState) -> float:
        p = self.parameters
        theta = float(
            np.mean(spherical_average(state.negative_particle_concentration_mol_m3))
            / p.negative.maximum_concentration_mol_m3
        )
        return float(
            np.clip(
                (theta - p.negative.stoichiometry_at_soc0)
                / (
                    p.negative.stoichiometry_at_soc1
                    - p.negative.stoichiometry_at_soc0
                ),
                0.0,
                1.0,
            )
        )


def _exchange_current(
    reaction_rate: float,
    alpha: float,
    electrolyte_concentration: np.ndarray,
    surface_concentration: np.ndarray,
    maximum_concentration: float,
) -> np.ndarray:
    electrolyte = np.maximum(electrolyte_concentration, 1.0e-12)
    surface = np.clip(
        surface_concentration,
        1.0e-12 * maximum_concentration,
        (1.0 - 1.0e-12) * maximum_concentration,
    )
    return (
        FARADAY
        * reaction_rate
        * electrolyte**alpha
        * (maximum_concentration - surface) ** alpha
        * surface ** (1.0 - alpha)
    )


__all__ = [
    "Battery3DConfig",
    "Battery3DDiagnostics",
    "Battery3DState",
    "PorousElectrode3D",
    "layered_battery_mesh",
]
