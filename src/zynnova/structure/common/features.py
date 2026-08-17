from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .elements import (
    covalent_radii,
    electronegativity,
    masses,
    periods_and_groups,
    van_der_waals_radii,
)
from .types import StructureData


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Controls deterministic atom and edge feature construction."""

    include_charge: bool = True
    include_mass: bool = True
    include_radii: bool = True
    include_period_group: bool = True
    include_electronegativity: bool = True
    include_edge_vector: bool = True
    include_unit_vector: bool = True
    include_periodic_shift: bool = True
    normalize_atomic_number: bool = True


def build_node_features(
    structure: StructureData,
    config: FeatureConfig,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, np.ndarray]]:
    z = structure.atomic_numbers
    columns: list[np.ndarray] = []
    names: list[str] = []
    attrs: dict[str, np.ndarray] = {}

    z_feature = z.astype(np.float64) / 118.0 if config.normalize_atomic_number else z
    columns.append(z_feature)
    names.append("atomic_number_scaled" if config.normalize_atomic_number else "atomic_number")

    mass = structure.masses if structure.masses is not None else masses(z)
    attrs["mass"] = mass.astype(np.float64)
    if config.include_mass:
        columns.append(mass / 300.0)
        names.append("atomic_mass_scaled")

    cov = covalent_radii(z)
    vdw = van_der_waals_radii(z)
    attrs["covalent_radius"] = cov
    attrs["vdw_radius"] = vdw
    if config.include_radii:
        columns.extend((cov / 3.0, vdw / 4.0))
        names.extend(("covalent_radius_scaled", "vdw_radius_scaled"))

    if config.include_period_group:
        period, group = periods_and_groups(z)
        attrs["period"] = period
        attrs["group"] = group
        columns.extend((period / 7.0, group / 18.0))
        names.extend(("period_scaled", "group_scaled"))

    if config.include_electronegativity:
        en, en_mask = electronegativity(z)
        attrs["electronegativity_pauling"] = en
        columns.extend((en / 4.0, en_mask))
        names.extend(("electronegativity_scaled", "electronegativity_present"))

    if config.include_charge:
        charge = (
            np.zeros(structure.num_atoms, dtype=np.float64)
            if structure.charges is None
            else structure.charges
        )
        attrs["charge"] = charge
        columns.append(charge)
        names.append("initial_charge")

    return np.column_stack(columns).astype(np.float32), tuple(names), attrs


def build_edge_features(
    edge_vec: np.ndarray,
    edge_dist: np.ndarray,
    edge_shift: np.ndarray,
    *,
    bond_order: np.ndarray | None,
    config: FeatureConfig,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, np.ndarray]]:
    edge_count = len(edge_dist)
    columns: list[np.ndarray] = [edge_dist[:, None]]
    names: list[str] = ["distance_angstrom"]
    attrs: dict[str, np.ndarray] = {}

    if config.include_edge_vector:
        columns.append(edge_vec)
        names.extend(("dx", "dy", "dz"))
    if config.include_unit_vector:
        unit = np.divide(
            edge_vec,
            edge_dist[:, None],
            out=np.zeros_like(edge_vec),
            where=edge_dist[:, None] > 1.0e-12,
        )
        columns.append(unit)
        names.extend(("ux", "uy", "uz"))
        attrs["unit_vector"] = unit
    if config.include_periodic_shift:
        columns.append(edge_shift.astype(np.float64))
        names.extend(("cell_shift_a", "cell_shift_b", "cell_shift_c"))

    periodic = np.any(edge_shift != 0, axis=1).astype(np.float64)
    columns.append(periodic[:, None])
    names.append("is_periodic_image")
    attrs["is_periodic_image"] = periodic.astype(bool)

    if bond_order is None:
        bond_order = np.zeros(edge_count, dtype=np.float64)
    else:
        bond_order = np.asarray(bond_order, dtype=np.float64).reshape(edge_count)
    columns.append(bond_order[:, None])
    names.append("bond_order")
    attrs["bond_order"] = bond_order

    return np.concatenate(columns, axis=1).astype(np.float32), tuple(names), attrs
