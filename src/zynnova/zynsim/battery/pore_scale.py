"""Direct voxel-resolved electrochemistry for electrode microstructures.

This finite-volume model resolves solid/electrolyte potential, electrolyte
concentration, solid lithium occupancy, and Butler--Volmer reaction on every
solid--electrolyte face.  It complements the homogenized P2D and Battery3D
models and is designed for representative microstructure volumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np

from ..constants import FARADAY, GAS_CONSTANT


OCPFunction = Callable[[np.ndarray, float], np.ndarray]


def constant_cathode_ocp(theta: np.ndarray, temperature_K: float) -> np.ndarray:
    del temperature_K
    return np.full_like(np.asarray(theta, dtype=float), 4.0)


@dataclass(frozen=True, slots=True)
class PoreScaleElectrochemConfig:
    voxel_size_m: float | tuple[float, float, float] = 1.0e-7
    active_labels: tuple[int, ...] = (1,)
    electrolyte_labels: tuple[int, ...] = (2,)
    solid_conductivity_S_m: float = 10.0
    electrolyte_conductivity_S_m: float = 1.0
    solid_diffusivity_m2_s: float = 1.0e-14
    electrolyte_diffusivity_m2_s: float = 2.0e-10
    maximum_solid_concentration_mol_m3: float = 49_200.0
    initial_electrolyte_concentration_mol_m3: float = 1_000.0
    exchange_current_reference_A_m2: float = 3.0
    charge_transfer_coefficient: float = 0.5
    transference_number: float = 0.3
    temperature_K: float = 298.15
    solid_collector_axis: int = 0
    solid_collector_side: Literal["min", "max"] = "min"
    electrolyte_reference_axis: int = 0
    electrolyte_reference_side: Literal["min", "max"] = "max"
    nonlinear_iterations: int = 20
    nonlinear_tolerance: float = 1.0e-7
    relaxation: float = 0.5
    maximum_reaction_current_A_m2: float = 1.0e4
    linear_solver: Literal["direct", "cg"] = "cg"
    linear_tolerance: float = 1.0e-9
    linear_max_iterations: int = 20_000
    ocp_V: OCPFunction = field(default=constant_cathode_ocp, compare=False, repr=False)

    def __post_init__(self) -> None:
        spacing = _triple(self.voxel_size_m)
        object.__setattr__(self, "voxel_size_m", spacing)
        positive = (
            self.solid_conductivity_S_m,
            self.electrolyte_conductivity_S_m,
            self.solid_diffusivity_m2_s,
            self.electrolyte_diffusivity_m2_s,
            self.maximum_solid_concentration_mol_m3,
            self.initial_electrolyte_concentration_mol_m3,
            self.exchange_current_reference_A_m2,
            self.temperature_K,
            self.nonlinear_tolerance,
            self.maximum_reaction_current_A_m2,
            self.linear_tolerance,
        )
        if min(positive) <= 0.0:
            raise ValueError("pore-scale material/numerical values must be positive")
        if not 0.0 < self.charge_transfer_coefficient < 1.0 or not 0.0 < self.transference_number < 1.0:
            raise ValueError("charge-transfer/transference values must lie in (0,1)")
        if self.solid_collector_axis not in (0, 1, 2) or self.electrolyte_reference_axis not in (0, 1, 2):
            raise ValueError("boundary axes must be 0,1,2")
        if self.solid_collector_side not in {"min", "max"} or self.electrolyte_reference_side not in {"min", "max"}:
            raise ValueError("boundary sides must be 'min' or 'max'")
        if self.nonlinear_iterations < 1 or self.linear_max_iterations < 1:
            raise ValueError("iteration limits must be positive")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("relaxation must lie in (0,1]")


@dataclass(slots=True)
class PoreScaleElectrochemState:
    time_s: float
    solid_potential_V: np.ndarray
    electrolyte_potential_V: np.ndarray
    electrolyte_concentration_mol_m3: np.ndarray
    solid_occupancy: np.ndarray
    reaction_current_A_m2: np.ndarray
    terminal_current_A: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def copy(self) -> "PoreScaleElectrochemState":
        return PoreScaleElectrochemState(
            self.time_s,
            self.solid_potential_V.copy(),
            self.electrolyte_potential_V.copy(),
            self.electrolyte_concentration_mol_m3.copy(),
            self.solid_occupancy.copy(),
            self.reaction_current_A_m2.copy(),
            self.terminal_current_A,
            dict(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class PoreScaleStepDiagnostics:
    time_s: float
    terminal_current_A: float
    mean_solid_occupancy: float
    minimum_electrolyte_concentration_mol_m3: float
    maximum_reaction_current_A_m2: float
    nonlinear_iterations: int
    nonlinear_residual: float
    lithium_balance_error_mol: float


class PoreScaleElectrochemistry:
    """Voxel finite-volume direct numerical simulation of an electrode RVE."""

    def __init__(self, phase_labels: np.ndarray, config: PoreScaleElectrochemConfig | None = None) -> None:
        try:
            from scipy import sparse
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pore-scale electrochemistry requires scipy") from exc
        self.config = config or PoreScaleElectrochemConfig()
        self.labels = np.ascontiguousarray(phase_labels, dtype=np.int32)
        if self.labels.ndim != 3 or min(self.labels.shape) < 2:
            raise ValueError("phase_labels must be a 3-D array with extents >= 2")
        self.solid_mask = np.isin(self.labels, self.config.active_labels)
        self.electrolyte_mask = np.isin(self.labels, self.config.electrolyte_labels)
        if not np.any(self.solid_mask) or not np.any(self.electrolyte_mask):
            raise ValueError("both active solid and electrolyte phases are required")
        self._solid_ids, self._solid_laplacian = _masked_laplacian(self.solid_mask, self.config.voxel_size_m)
        self._electrolyte_ids, self._electrolyte_laplacian = _masked_laplacian(self.electrolyte_mask, self.config.voxel_size_m)
        self._solid_boundary = _boundary_unknowns(
            self.solid_mask, self._solid_ids, self.config.solid_collector_axis, self.config.solid_collector_side
        )
        self._electrolyte_boundary = _boundary_unknowns(
            self.electrolyte_mask, self._electrolyte_ids, self.config.electrolyte_reference_axis, self.config.electrolyte_reference_side
        )
        if not len(self._solid_boundary) or not len(self._electrolyte_boundary):
            raise ValueError("selected solid/electrolyte reference boundaries do not touch their phases")
        _require_all_components_touch_boundary(
            self.solid_mask,
            axis=self.config.solid_collector_axis,
            side=self.config.solid_collector_side,
            phase_name="active solid",
        )
        _require_all_components_touch_boundary(
            self.electrolyte_mask,
            axis=self.config.electrolyte_reference_axis,
            side=self.config.electrolyte_reference_side,
            phase_name="electrolyte",
        )
        self._solid_A = _dirichlet_matrix(self.config.solid_conductivity_S_m * self._solid_laplacian, self._solid_boundary)
        self._electrolyte_A = _dirichlet_matrix(self.config.electrolyte_conductivity_S_m * self._electrolyte_laplacian, self._electrolyte_boundary)
        self._active_face_ids, self._electrolyte_face_ids, self._face_area = _interface_faces(
            self.solid_mask, self.electrolyte_mask, self._solid_ids, self._electrolyte_ids, self.config.voxel_size_m
        )
        if not len(self._active_face_ids):
            raise ValueError("active solid and electrolyte have no shared faces")
        self._voxel_volume = float(np.prod(self.config.voxel_size_m))

    def initialize(
        self,
        *,
        solid_occupancy: float = 0.6,
        electrolyte_concentration_mol_m3: float | None = None,
        solid_potential_V: float = 4.0,
        electrolyte_potential_V: float = 0.0,
    ) -> PoreScaleElectrochemState:
        shape = self.labels.shape
        c_e = self.config.initial_electrolyte_concentration_mol_m3 if electrolyte_concentration_mol_m3 is None else float(electrolyte_concentration_mol_m3)
        return PoreScaleElectrochemState(
            time_s=0.0,
            solid_potential_V=np.where(self.solid_mask, float(solid_potential_V), np.nan),
            electrolyte_potential_V=np.where(self.electrolyte_mask, float(electrolyte_potential_V), np.nan),
            electrolyte_concentration_mol_m3=np.where(self.electrolyte_mask, c_e, 0.0),
            solid_occupancy=np.where(self.solid_mask, np.clip(float(solid_occupancy), 1.0e-8, 1.0 - 1.0e-8), 0.0),
            reaction_current_A_m2=np.zeros(len(self._active_face_ids), dtype=float),
        )

    def step(
        self,
        state: PoreScaleElectrochemState,
        *,
        applied_solid_potential_V: float,
        dt_s: float,
        electrolyte_reference_V: float = 0.0,
        temperature_K: float | None = None,
    ) -> tuple[PoreScaleElectrochemState, PoreScaleStepDiagnostics]:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        temperature = self.config.temperature_K if temperature_K is None else float(temperature_K)
        phi_s = state.solid_potential_V[self.solid_mask].copy()
        phi_e = state.electrolyte_potential_V[self.electrolyte_mask].copy()
        theta = state.solid_occupancy[self.solid_mask].copy()
        concentration = state.electrolyte_concentration_mol_m3[self.electrolyte_mask].copy()
        reaction = state.reaction_current_A_m2.copy()
        residual = np.inf
        iteration = 0
        for iteration in range(1, self.config.nonlinear_iterations + 1):
            theta_face = theta[self._active_face_ids]
            c_face = concentration[self._electrolyte_face_ids]
            ocp = np.asarray(self.config.ocp_V(theta_face, temperature), dtype=float)
            eta = phi_s[self._active_face_ids] - phi_e[self._electrolyte_face_ids] - ocp
            i0 = self.config.exchange_current_reference_A_m2 * np.sqrt(
                np.maximum(c_face / self.config.initial_electrolyte_concentration_mol_m3, 1.0e-12)
                * np.maximum(theta_face * (1.0 - theta_face), 1.0e-12)
            )
            argument = np.clip(
                self.config.charge_transfer_coefficient * FARADAY * eta / (GAS_CONSTANT * temperature),
                -30.0,
                30.0,
            )
            candidate_reaction = np.clip(
                2.0 * i0 * np.sinh(argument),
                -self.config.maximum_reaction_current_A_m2,
                self.config.maximum_reaction_current_A_m2,
            )
            reaction_new = (1.0 - self.config.relaxation) * reaction + self.config.relaxation * candidate_reaction
            solid_source = np.zeros_like(phi_s)
            electrolyte_source = np.zeros_like(phi_e)
            np.add.at(solid_source, self._active_face_ids, -reaction_new * self._face_area / self._voxel_volume)
            np.add.at(electrolyte_source, self._electrolyte_face_ids, reaction_new * self._face_area / self._voxel_volume)
            solid_rhs = solid_source.copy()
            electrolyte_rhs = electrolyte_source.copy()
            solid_rhs[self._solid_boundary] = float(applied_solid_potential_V)
            electrolyte_rhs[self._electrolyte_boundary] = float(electrolyte_reference_V)
            phi_s_new = _solve(self._solid_A, solid_rhs, phi_s, self.config)
            phi_e_new = _solve(self._electrolyte_A, electrolyte_rhs, phi_e, self.config)
            residual = max(
                _relative_change(phi_s_new, phi_s),
                _relative_change(phi_e_new, phi_e),
                _relative_change(reaction_new, reaction),
            )
            phi_s, phi_e, reaction = phi_s_new, phi_e_new, reaction_new
            if residual <= self.config.nonlinear_tolerance:
                break
        # Implicit diffusion with reaction transfer after converged potentials.
        solid_mol_rate = np.zeros_like(theta)
        electrolyte_mol_rate = np.zeros_like(concentration)
        np.add.at(solid_mol_rate, self._active_face_ids, -reaction * self._face_area / (FARADAY * self._voxel_volume))
        np.add.at(
            electrolyte_mol_rate,
            self._electrolyte_face_ids,
            (1.0 - self.config.transference_number) * reaction * self._face_area / (FARADAY * self._voxel_volume),
        )
        solid_matrix = _identity_plus(self._solid_laplacian, dt_s * self.config.solid_diffusivity_m2_s)
        electrolyte_matrix = _identity_plus(self._electrolyte_laplacian, dt_s * self.config.electrolyte_diffusivity_m2_s)
        theta_new = _solve(
            solid_matrix,
            theta + dt_s * solid_mol_rate / self.config.maximum_solid_concentration_mol_m3,
            theta,
            self.config,
        )
        concentration_new = _solve(
            electrolyte_matrix,
            concentration + dt_s * electrolyte_mol_rate,
            concentration,
            self.config,
        )
        theta_new = np.clip(theta_new, 1.0e-8, 1.0 - 1.0e-8)
        concentration_new = np.maximum(concentration_new, 1.0e-9)
        output = state.copy()
        output.time_s = state.time_s + dt_s
        output.solid_potential_V[self.solid_mask] = phi_s
        output.electrolyte_potential_V[self.electrolyte_mask] = phi_e
        output.solid_occupancy[self.solid_mask] = theta_new
        output.electrolyte_concentration_mol_m3[self.electrolyte_mask] = concentration_new
        output.reaction_current_A_m2 = reaction
        terminal_current = float(np.sum(reaction * self._face_area))
        output.terminal_current_A = terminal_current
        before = (
            np.sum(theta) * self.config.maximum_solid_concentration_mol_m3 * self._voxel_volume
            + np.sum(concentration) * self._voxel_volume
        )
        after = (
            np.sum(theta_new) * self.config.maximum_solid_concentration_mol_m3 * self._voxel_volume
            + np.sum(concentration_new) * self._voxel_volume
        )
        diagnostic = PoreScaleStepDiagnostics(
            time_s=output.time_s,
            terminal_current_A=terminal_current,
            mean_solid_occupancy=float(np.mean(theta_new)),
            minimum_electrolyte_concentration_mol_m3=float(np.min(concentration_new)),
            maximum_reaction_current_A_m2=float(np.max(np.abs(reaction))),
            nonlinear_iterations=iteration,
            nonlinear_residual=float(residual),
            lithium_balance_error_mol=float(after - before),
        )
        output.metadata.update({
            "nonlinear_iterations": iteration,
            "nonlinear_residual": float(residual),
            "applied_solid_potential_V": float(applied_solid_potential_V),
            "temperature_K": temperature,
        })
        return output, diagnostic



def _require_all_components_touch_boundary(mask: np.ndarray, *, axis: int, side: str, phase_name: str) -> None:
    """Reject electrically floating components before assembling a singular system.

    A direct pore-scale potential problem needs at least one Dirichlet reference
    in every connected component.  Silently regularising isolated particles or
    pores would change the physics, so the model fails early and asks the caller
    to repair, remove, or explicitly connect those components.
    """

    from scipy.ndimage import generate_binary_structure, label

    components, count = label(mask, structure=generate_binary_structure(3, 1))
    if count <= 1:
        return
    coordinate = 0 if side == "min" else mask.shape[axis] - 1
    selection = [slice(None)] * 3
    selection[axis] = coordinate
    touched = set(map(int, np.unique(components[tuple(selection)])))
    touched.discard(0)
    missing = sorted(set(range(1, count + 1)) - touched)
    if missing:
        voxel_counts = np.bincount(components.ravel(), minlength=count + 1)
        largest = sorted((int(voxel_counts[index]), index) for index in missing)[-5:]
        raise ValueError(
            f"{phase_name} contains {len(missing)} floating connected component(s) "
            f"that do not touch the selected {side} boundary on axis {axis}; "
            f"largest missing components (voxels, id)={largest}. Repair the "
            "segmentation/percolation or choose a boundary touching every component."
        )

def _masked_laplacian(mask: np.ndarray, spacing: tuple[float, float, float]):
    from scipy import sparse

    ids = np.full(mask.shape, -1, dtype=np.int64)
    ids[mask] = np.arange(np.count_nonzero(mask), dtype=np.int64)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    diagonal = np.zeros(np.count_nonzero(mask), dtype=float)
    for axis in range(3):
        coefficient = 1.0 / spacing[axis] ** 2
        left_slice = [slice(None)] * 3
        right_slice = [slice(None)] * 3
        left_slice[axis] = slice(0, -1)
        right_slice[axis] = slice(1, None)
        left = tuple(left_slice)
        right = tuple(right_slice)
        pair = mask[left] & mask[right]
        a = ids[left][pair]
        b = ids[right][pair]
        if len(a):
            rows.extend((a, b))
            cols.extend((b, a))
            data.extend((np.full(len(a), -coefficient), np.full(len(a), -coefficient)))
            np.add.at(diagonal, a, coefficient)
            np.add.at(diagonal, b, coefficient)
    index = np.arange(len(diagonal), dtype=np.int64)
    rows.append(index)
    cols.append(index)
    data.append(diagonal)
    matrix = sparse.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(len(diagonal), len(diagonal))).tocsr()
    return ids, matrix


def _boundary_unknowns(mask: np.ndarray, ids: np.ndarray, axis: int, side: str) -> np.ndarray:
    coordinate = 0 if side == "min" else mask.shape[axis] - 1
    selection = [slice(None)] * 3
    selection[axis] = coordinate
    values = ids[tuple(selection)]
    return np.asarray(values[values >= 0], dtype=np.int64)


def _dirichlet_matrix(matrix, boundary: np.ndarray):
    output = matrix.tolil(copy=True)
    for index in np.unique(boundary):
        output.rows[int(index)] = [int(index)]
        output.data[int(index)] = [1.0]
    return output.tocsr()


def _identity_plus(laplacian, scale: float):
    from scipy import sparse
    return sparse.eye(laplacian.shape[0], format="csr") + float(scale) * laplacian


def _interface_faces(
    solid: np.ndarray,
    electrolyte: np.ndarray,
    solid_ids: np.ndarray,
    electrolyte_ids: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active_ids: list[np.ndarray] = []
    electrolyte_face_ids: list[np.ndarray] = []
    areas: list[np.ndarray] = []
    for axis in range(3):
        left_slice = [slice(None)] * 3
        right_slice = [slice(None)] * 3
        left_slice[axis] = slice(0, -1)
        right_slice[axis] = slice(1, None)
        left, right = tuple(left_slice), tuple(right_slice)
        face_area = float(np.prod([spacing[d] for d in range(3) if d != axis]))
        mask = solid[left] & electrolyte[right]
        if np.any(mask):
            active_ids.append(solid_ids[left][mask])
            electrolyte_face_ids.append(electrolyte_ids[right][mask])
            areas.append(np.full(np.count_nonzero(mask), face_area))
        mask = electrolyte[left] & solid[right]
        if np.any(mask):
            active_ids.append(solid_ids[right][mask])
            electrolyte_face_ids.append(electrolyte_ids[left][mask])
            areas.append(np.full(np.count_nonzero(mask), face_area))
    if not active_ids:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    return np.concatenate(active_ids), np.concatenate(electrolyte_face_ids), np.concatenate(areas)


def _solve(matrix, rhs: np.ndarray, initial: np.ndarray, config: PoreScaleElectrochemConfig) -> np.ndarray:
    from scipy.sparse.linalg import cg, spsolve
    if config.linear_solver == "direct":
        return np.asarray(spsolve(matrix, rhs), dtype=float)
    solution, info = cg(
        matrix,
        rhs,
        x0=initial,
        rtol=config.linear_tolerance,
        atol=0.0,
        maxiter=config.linear_max_iterations,
    )
    if info != 0:
        raise RuntimeError(f"pore-scale CG failed with info={info}")
    return np.asarray(solution, dtype=float)


def _relative_change(new: np.ndarray, old: np.ndarray) -> float:
    return float(np.linalg.norm(new - old) / max(np.linalg.norm(new), np.linalg.norm(old), 1.0e-30))


def _triple(value: float | Sequence[float]) -> tuple[float, float, float]:
    result = (float(value),) * 3 if np.isscalar(value) else tuple(map(float, value))
    if len(result) != 3 or min(result) <= 0.0:
        raise ValueError("voxel_size_m must contain three positive values")
    return result  # type: ignore[return-value]


__all__ = [
    "PoreScaleElectrochemConfig",
    "PoreScaleElectrochemState",
    "PoreScaleElectrochemistry",
    "PoreScaleStepDiagnostics",
    "constant_cathode_ocp",
]
