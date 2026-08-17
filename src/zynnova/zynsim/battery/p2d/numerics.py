"""Conservative finite-volume operators used by the P2D solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sparse_modules():
    try:
        import scipy.sparse as sparse
        import scipy.sparse.linalg as sparse_linalg
    except ImportError as exc:
        raise ImportError("P2D integration requires SciPy; install zynnova[battery]") from exc
    return sparse, sparse_linalg


@dataclass(frozen=True, slots=True)
class ThroughCellGrid:
    centers_m: np.ndarray
    widths_m: np.ndarray
    negative: slice
    separator: slice
    positive: slice


def through_cell_grid(
    negative_thickness_m: float,
    separator_thickness_m: float,
    positive_thickness_m: float,
    negative_cells: int,
    separator_cells: int,
    positive_cells: int,
) -> ThroughCellGrid:
    widths = np.concatenate(
        (
            np.full(negative_cells, negative_thickness_m / negative_cells),
            np.full(separator_cells, separator_thickness_m / separator_cells),
            np.full(positive_cells, positive_thickness_m / positive_cells),
        )
    )
    edges = np.concatenate(([0.0], np.cumsum(widths)))
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_end = negative_cells
    s_end = n_end + separator_cells
    return ThroughCellGrid(
        centers_m=centers,
        widths_m=widths,
        negative=slice(0, n_end),
        separator=slice(n_end, s_end),
        positive=slice(s_end, len(widths)),
    )


def harmonic_face_coefficients(
    cell_values: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    values = np.asarray(cell_values, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    if values.shape != widths.shape or np.any(values <= 0.0) or np.any(widths <= 0.0):
        raise ValueError("face coefficients and widths must be aligned and positive")
    left_resistance = 0.5 * widths[:-1] / values[:-1]
    right_resistance = 0.5 * widths[1:] / values[1:]
    return 1.0 / (left_resistance + right_resistance)


def conservative_diffusion_step(
    previous: np.ndarray,
    dt_s: float,
    widths_m: np.ndarray,
    storage: np.ndarray,
    diffusivity_m2_s: np.ndarray,
    source_mol_m3_s: np.ndarray,
) -> np.ndarray:
    """Backward-Euler update for a nonuniform, no-flux 1-D finite volume grid."""

    sparse, sparse_linalg = _sparse_modules()
    previous = np.asarray(previous, dtype=np.float64)
    widths = np.asarray(widths_m, dtype=np.float64)
    storage = np.asarray(storage, dtype=np.float64)
    diffusivity = np.asarray(diffusivity_m2_s, dtype=np.float64)
    source = np.asarray(source_mol_m3_s, dtype=np.float64)
    if not (
        previous.shape == widths.shape == storage.shape == diffusivity.shape == source.shape
    ):
        raise ValueError("finite-volume arrays must have the same shape")
    if dt_s <= 0.0 or np.any(widths <= 0.0) or np.any(storage <= 0.0):
        raise ValueError("finite-volume time, widths, and storage must be positive")

    face_conductance = harmonic_face_coefficients(diffusivity, widths)
    size = len(previous)
    operator = sparse.lil_matrix((size, size), dtype=np.float64)
    for face, conductance in enumerate(face_conductance):
        left = face
        right = face + 1
        operator[left, left] -= conductance
        operator[left, right] += conductance
        operator[right, left] += conductance
        operator[right, right] -= conductance
    mass = storage * widths
    matrix = sparse.diags(mass) - dt_s * operator.tocsr()
    rhs = mass * previous + dt_s * source * widths
    updated = sparse_linalg.spsolve(matrix, rhs)
    return np.asarray(updated, dtype=np.float64)


def spherical_particle_step(
    previous: np.ndarray,
    dt_s: float,
    particle_radius_m: float,
    diffusivity_m2_s: float,
    outward_current_A_m2: np.ndarray,
    faraday_C_mol: float,
) -> np.ndarray:
    """Conservative implicit spherical diffusion for every macro electrode cell."""

    sparse, sparse_linalg = _sparse_modules()
    previous = np.asarray(previous, dtype=np.float64)
    currents = np.asarray(outward_current_A_m2, dtype=np.float64)
    if previous.ndim != 2 or currents.shape != (previous.shape[0],):
        raise ValueError("particle states must have shape (n_macro, n_radial)")
    if dt_s <= 0.0 or particle_radius_m <= 0.0 or diffusivity_m2_s <= 0.0:
        raise ValueError("particle time, radius, and diffusivity must be positive")
    radial_cells = previous.shape[1]
    edges = np.linspace(0.0, particle_radius_m, radial_cells + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    volumes = (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
    operator = sparse.lil_matrix((radial_cells, radial_cells), dtype=np.float64)
    for face in range(1, radial_cells):
        conductance = diffusivity_m2_s * edges[face] ** 2 / (
            centers[face] - centers[face - 1]
        )
        left = face - 1
        right = face
        operator[left, left] -= conductance
        operator[left, right] += conductance
        operator[right, left] += conductance
        operator[right, right] -= conductance
    matrix = sparse.diags(volumes) - dt_s * operator.tocsr()
    updated = np.empty_like(previous)
    outer_area_without_4pi = particle_radius_m**2
    for macro_cell in range(previous.shape[0]):
        outward_molar_flux = currents[macro_cell] / faraday_C_mol
        boundary = np.zeros(radial_cells)
        boundary[-1] = -outer_area_without_4pi * outward_molar_flux
        rhs = volumes * previous[macro_cell] + dt_s * boundary
        updated[macro_cell] = sparse_linalg.spsolve(matrix, rhs)
    return updated


def spherical_inventory(concentration: np.ndarray, particle_radius_m: float) -> np.ndarray:
    concentration = np.asarray(concentration, dtype=np.float64)
    radial_cells = concentration.shape[-1]
    edges = np.linspace(0.0, particle_radius_m, radial_cells + 1)
    shell_volumes_without_4pi = (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
    return concentration @ shell_volumes_without_4pi


def spherical_average(concentration: np.ndarray) -> np.ndarray:
    """Return volume-weighted averages over the final radial dimension."""

    values = np.asarray(concentration, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("spherical concentration needs a nonempty radial dimension")
    edges = np.linspace(0.0, 1.0, values.shape[-1] + 1)
    weights = edges[1:] ** 3 - edges[:-1] ** 3
    return np.sum(values * weights, axis=-1) / np.sum(weights)


__all__ = [
    "ThroughCellGrid",
    "conservative_diffusion_step",
    "harmonic_face_coefficients",
    "spherical_average",
    "spherical_inventory",
    "spherical_particle_step",
    "through_cell_grid",
]
