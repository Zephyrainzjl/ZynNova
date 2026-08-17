"""Global Tet4 assembly for transport and mechanics operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.boundary import SurfaceLoad
from ..core.mesh import Mesh
from ._backend import BackendName, native_module, resolve_backend
from .elements import (
    elastic_stiffness,
    isotropic_constitutive,
    normalize_tensor,
    strain_displacement,
    tet4_geometry,
)


def _sparse() -> Any:
    try:
        import scipy.sparse as sparse
    except ImportError as exc:
        raise ImportError("FEM assembly requires SciPy; install zynnova[simulation]") from exc
    return sparse


@dataclass(slots=True)
class ScalarOperators:
    mass: Any
    stiffness: Any
    backend: str


def _element_scalars(value: float | np.ndarray, count: int, name: str) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 0:
        result = np.full(count, float(raw), dtype=np.float64)
    elif raw.shape == (count,):
        result = raw
    else:
        raise ValueError(f"{name} must be scalar or have shape (n_cells,)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(result)


def _element_tensors(value: float | np.ndarray, count: int) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 0 or raw.shape == (3, 3):
        tensor = normalize_tensor(raw)
        return np.repeat(tensor[None, :, :], count, axis=0)
    if raw.shape == (count,):
        tensors = np.zeros((count, 3, 3), dtype=np.float64)
        tensors[:, np.arange(3), np.arange(3)] = raw[:, None]
    elif raw.shape == (count, 3, 3):
        tensors = raw.copy()
    else:
        raise ValueError(
            "transport coefficient must be scalar, (3,3), (n_cells,), or (n_cells,3,3)"
        )
    for tensor in tensors:
        normalize_tensor(tensor)
    return np.ascontiguousarray(tensors)


def _python_transport_coefficients(
    value: float | np.ndarray,
    count: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return either isotropic scalars or validated full tensors."""

    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 0:
        isotropic = np.full(count, float(raw), dtype=np.float64)
    elif raw.shape == (count,):
        isotropic = np.ascontiguousarray(raw)
    else:
        isotropic = None
    if isotropic is not None:
        if np.any(~np.isfinite(isotropic)) or np.any(isotropic < 0.0):
            raise ValueError(
                "isotropic transport coefficients must be finite and non-negative"
            )
        return isotropic, None

    if raw.shape == (3, 3):
        tensors = np.broadcast_to(raw, (count, 3, 3))
    elif raw.shape == (count, 3, 3):
        tensors = raw
    else:
        raise ValueError(
            "transport coefficient must be scalar, (3,3), (n_cells,), or "
            "(n_cells,3,3)"
        )
    if np.any(~np.isfinite(tensors)):
        raise ValueError("transport coefficient contains non-finite values")
    scale = np.maximum(np.max(np.abs(tensors), axis=(1, 2)), 1.0)
    symmetry_error = np.max(np.abs(tensors - np.swapaxes(tensors, 1, 2)), axis=(1, 2))
    if np.any(symmetry_error > 1.0e-12 * scale):
        raise ValueError("coefficient tensor must be symmetric")
    eigenvalues = np.linalg.eigvalsh(tensors)
    if np.any(eigenvalues[:, 0] < -1.0e-14 * scale):
        raise ValueError("coefficient tensor must be positive semidefinite")
    return None, np.ascontiguousarray(tensors)


def _batch_tet4_geometry(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Tet4 volumes and physical shape-function gradients."""

    coordinates = mesh.nodes[mesh.cells]
    jacobians = np.stack(
        (
            coordinates[:, 1] - coordinates[:, 0],
            coordinates[:, 2] - coordinates[:, 0],
            coordinates[:, 3] - coordinates[:, 0],
        ),
        axis=2,
    )
    determinants = np.linalg.det(jacobians)
    volumes = np.abs(determinants) / 6.0
    reference_gradients = np.asarray(
        (
            (-1.0, -1.0, -1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    gradients = np.einsum(
        "ij,ejk->eik",
        reference_gradients,
        np.linalg.inv(jacobians),
        optimize=True,
    )
    return volumes, gradients


def _coo_from_native(payload: dict[str, Any]) -> Any:
    sparse = _sparse()
    return sparse.coo_matrix(
        (
            np.asarray(payload["values"], dtype=np.float64),
            (
                np.asarray(payload["rows"], dtype=np.int64),
                np.asarray(payload["columns"], dtype=np.int64),
            ),
        ),
        shape=tuple(map(int, payload["shape"])),
    ).tocsr()


def assemble_scalar(
    mesh: Mesh,
    conductivity: float | np.ndarray,
    capacity: float | np.ndarray = 1.0,
    *,
    lumped_mass: bool = False,
    backend: BackendName = "auto",
) -> ScalarOperators:
    """Assemble ``M_ij=∫ capacity N_i N_j`` and ``K_ij=∫ gradN_i·D·gradN_j``."""

    sparse = _sparse()
    capacities = _element_scalars(capacity, mesh.n_cells, "capacity")
    if np.any(capacities < 0.0):
        raise ValueError("capacity must be non-negative")
    selected = resolve_backend(backend)
    if selected == "cpp":
        tensors = _element_tensors(conductivity, mesh.n_cells)
        native = native_module()
        assert native is not None
        payload = native.assemble_scalar(
            mesh.nodes,
            mesh.cells,
            tensors,
            capacities,
            bool(lumped_mass),
        )
        return ScalarOperators(
            mass=_coo_from_native(payload["mass"]),
            stiffness=_coo_from_native(payload["stiffness"]),
            backend="cpp",
        )

    isotropic, tensors = _python_transport_coefficients(
        conductivity,
        mesh.n_cells,
    )
    volumes, gradients = _batch_tet4_geometry(mesh)
    if isotropic is not None:
        local_stiffness = np.einsum(
            "e,eik,ejk->eij",
            volumes * isotropic,
            gradients,
            gradients,
            optimize=True,
        )
    else:
        assert tensors is not None
        local_stiffness = np.einsum(
            "e,eik,ekl,ejl->eij",
            volumes,
            gradients,
            tensors,
            gradients,
            optimize=True,
        )

    if lumped_mass:
        local_mass = np.zeros((mesh.n_cells, 4, 4), dtype=np.float64)
        local_mass[:, np.arange(4), np.arange(4)] = (
            capacities * volumes / 4.0
        )[:, None]
    else:
        local_mass = np.ones((mesh.n_cells, 4, 4), dtype=np.float64)
        local_mass[:, np.arange(4), np.arange(4)] = 2.0
        local_mass *= (capacities * volumes / 20.0)[:, None, None]

    rows = np.repeat(mesh.cells, 4, axis=1).reshape(-1)
    columns = np.tile(mesh.cells, (1, 4)).reshape(-1)
    mass_values = local_mass.reshape(-1)
    stiffness_values = local_stiffness.reshape(-1)
    shape = (mesh.n_nodes, mesh.n_nodes)
    return ScalarOperators(
        mass=sparse.coo_matrix((mass_values, (rows, columns)), shape=shape).tocsr(),
        stiffness=sparse.coo_matrix(
            (stiffness_values, (rows, columns)), shape=shape
        ).tocsr(),
        backend="python",
    )


def assemble_elasticity(
    mesh: Mesh,
    young_modulus: float | np.ndarray,
    poisson_ratio: float | np.ndarray,
    *,
    backend: BackendName = "auto",
) -> Any:
    """Assemble the small-strain isotropic 3-D elasticity tangent."""

    sparse = _sparse()
    young = _element_scalars(young_modulus, mesh.n_cells, "young_modulus")
    poisson = _element_scalars(poisson_ratio, mesh.n_cells, "poisson_ratio")
    selected = resolve_backend(backend)
    if selected == "cpp":
        native = native_module()
        assert native is not None
        return _coo_from_native(
            native.assemble_elasticity(mesh.nodes, mesh.cells, young, poisson)
        )

    entries = mesh.n_cells * 144
    rows = np.empty(entries, dtype=np.int64)
    columns = np.empty(entries, dtype=np.int64)
    values = np.empty(entries, dtype=np.float64)
    cursor = 0
    for element, cell in enumerate(mesh.cells):
        local = elastic_stiffness(mesh.nodes[cell], young[element], poisson[element])
        dofs = (3 * cell[:, None] + np.arange(3)[None, :]).reshape(-1)
        row_grid, column_grid = np.meshgrid(dofs, dofs, indexing="ij")
        block = slice(cursor, cursor + 144)
        rows[block] = row_grid.reshape(-1)
        columns[block] = column_grid.reshape(-1)
        values[block] = local.reshape(-1)
        cursor += 144
    size = 3 * mesh.n_nodes
    return sparse.coo_matrix((values, (rows, columns)), shape=(size, size)).tocsr()


def assemble_scalar_source(mesh: Mesh, source: float | np.ndarray) -> np.ndarray:
    """Assemble a volumetric scalar source.

    A vector with ``n_cells`` entries is interpreted as cellwise constant; a
    vector with ``n_nodes`` entries is consistently projected with the mass
    matrix.
    """

    raw = np.asarray(source, dtype=np.float64)
    if raw.ndim == 0:
        cell_values = np.full(mesh.n_cells, float(raw))
    elif raw.shape == (mesh.n_cells,):
        cell_values = raw
    elif raw.shape == (mesh.n_nodes,):
        unit_mass = assemble_scalar(mesh, 0.0, 1.0, backend="python").mass
        return np.asarray(unit_mass @ raw, dtype=np.float64)
    else:
        raise ValueError("scalar source must be scalar, cellwise, or nodal")
    weights = np.repeat(cell_values * mesh.cell_volumes() / 4.0, 4)
    return np.bincount(
        mesh.cells.reshape(-1),
        weights=weights,
        minlength=mesh.n_nodes,
    )


def assemble_body_force(mesh: Mesh, body_force: np.ndarray | float) -> np.ndarray:
    raw = np.asarray(body_force, dtype=np.float64)
    if raw.ndim == 0:
        cell_values = np.full((mesh.n_cells, 3), float(raw))
    elif raw.shape == (3,):
        cell_values = np.repeat(raw[None, :], mesh.n_cells, axis=0)
    elif raw.shape == (mesh.n_cells, 3):
        cell_values = raw
    else:
        raise ValueError("body force must be scalar, a 3-vector, or (n_cells, 3)")
    result = np.zeros((mesh.n_nodes, 3), dtype=np.float64)
    for cell, volume, value in zip(mesh.cells, mesh.cell_volumes(), cell_values, strict=True):
        result[cell] += volume * value[None, :] / 4.0
    return result.reshape(-1)


def triangle_areas(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    coordinates = nodes[np.asarray(faces, dtype=np.int64)]
    return 0.5 * np.linalg.norm(
        np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]),
        axis=1,
    )


def assemble_surface_load(mesh: Mesh, load: SurfaceLoad, *, vector: bool) -> np.ndarray:
    areas = triangle_areas(mesh.nodes, load.faces)
    if vector:
        value = np.asarray(load.value, dtype=np.float64)
        if value.size == 1:
            value = np.repeat(value, 3)
        result = np.zeros((mesh.n_nodes, 3), dtype=np.float64)
        for face, area in zip(load.faces, areas, strict=True):
            result[face] += area * value.reshape(1, 3) / 3.0
        return result.reshape(-1)
    if load.value.size != 1:
        raise ValueError("a scalar problem requires scalar surface flux")
    result = np.zeros(mesh.n_nodes, dtype=np.float64)
    for face, area in zip(load.faces, areas, strict=True):
        result[face] += area * float(load.value.reshape(-1)[0]) / 3.0
    return result


def assemble_eigenstrain_load(
    mesh: Mesh,
    young_modulus: float | np.ndarray,
    poisson_ratio: float | np.ndarray,
    eigenstrain: np.ndarray,
) -> np.ndarray:
    """Assemble ``∫ B.T C epsilon_star`` for engineering-Voigt eigenstrain."""

    young = _element_scalars(young_modulus, mesh.n_cells, "young_modulus")
    poisson = _element_scalars(poisson_ratio, mesh.n_cells, "poisson_ratio")
    strain = np.asarray(eigenstrain, dtype=np.float64)
    if strain.shape == (6,):
        strain = np.repeat(strain[None, :], mesh.n_cells, axis=0)
    if strain.shape != (mesh.n_cells, 6):
        raise ValueError("eigenstrain must have shape (6,) or (n_cells, 6)")
    result = np.zeros(3 * mesh.n_nodes, dtype=np.float64)
    for element, cell in enumerate(mesh.cells):
        volume, gradients = tet4_geometry(mesh.nodes[cell])
        b_matrix = strain_displacement(gradients)
        local = volume * b_matrix.T @ (
            isotropic_constitutive(young[element], poisson[element]) @ strain[element]
        )
        dofs = (3 * cell[:, None] + np.arange(3)[None, :]).reshape(-1)
        result[dofs] += local
    return result


def cell_gradients(mesh: Mesh, nodal_values: np.ndarray) -> np.ndarray:
    values = np.asarray(nodal_values, dtype=np.float64)
    if values.shape != (mesh.n_nodes,):
        raise ValueError("nodal scalar values must have shape (n_nodes,)")
    _, gradients = _batch_tet4_geometry(mesh)
    return np.einsum(
        "ei,eij->ej",
        values[mesh.cells],
        gradients,
        optimize=True,
    )


__all__ = [
    "ScalarOperators",
    "assemble_body_force",
    "assemble_eigenstrain_load",
    "assemble_elasticity",
    "assemble_scalar",
    "assemble_scalar_source",
    "assemble_surface_load",
    "cell_gradients",
    "triangle_areas",
]
