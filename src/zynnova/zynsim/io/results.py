"""Portable NPZ/JSON and meshio-backed VTU result writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np

from ..battery.p2d.protocol import P2DTrajectory
from ..battery.p2d.state import P2DState
from ..core.mesh import Mesh

if TYPE_CHECKING:
    from ..battery.aging import AgingTrajectory
    from ..battery.diagnostics import EISResult
    from ..battery.pack import PackTrajectory


_P2D_ARRAY_FIELDS = (
    "electrolyte_concentration_mol_m3",
    "negative_particle_concentration_mol_m3",
    "positive_particle_concentration_mol_m3",
    "electrolyte_potential_V",
    "negative_solid_potential_V",
    "positive_solid_potential_V",
    "negative_interfacial_current_A_m2",
    "positive_interfacial_current_A_m2",
)


def save_p2d_state(path: str | Path, state: P2DState) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(getattr(state, name)) for name in _P2D_ARRAY_FIELDS}
    arrays["scalars"] = np.asarray(
        [
            state.time_s,
            state.temperature_K,
            state.terminal_voltage_V,
            state.current_A,
        ]
    )
    arrays["metadata_json"] = np.asarray(json.dumps(state.metadata, default=_json_default))
    np.savez_compressed(target, **arrays)
    return target


def load_p2d_state(path: str | Path) -> P2DState:
    with np.load(Path(path), allow_pickle=False) as payload:
        scalars = np.asarray(payload["scalars"], dtype=float)
        if scalars.shape != (4,):
            raise ValueError("P2D state archive has an incompatible scalar block")
        metadata = json.loads(str(payload["metadata_json"]))
        fields = {name: np.asarray(payload[name]).copy() for name in _P2D_ARRAY_FIELDS}
    return P2DState(
        time_s=float(scalars[0]),
        temperature_K=float(scalars[1]),
        terminal_voltage_V=float(scalars[2]),
        current_A=float(scalars[3]),
        metadata=metadata,
        **fields,
    )


def save_p2d_trajectory(path: str | Path, trajectory: P2DTrajectory) -> Path:
    if not trajectory.states:
        raise ValueError("cannot serialize an empty P2D trajectory")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "time_s": trajectory.time_s,
        "voltage_V": trajectory.voltage_V,
        "current_A": trajectory.current_A,
        "temperature_K": trajectory.temperature_K,
        "segment_labels": np.asarray(trajectory.segment_labels, dtype=str),
        "termination_reason": np.asarray(trajectory.termination_reason),
    }
    for name in _P2D_ARRAY_FIELDS:
        payload[name] = np.stack([np.asarray(getattr(state, name)) for state in trajectory.states])
    np.savez_compressed(target, **payload)
    return target


def save_aging_trajectory(path: str | Path, trajectory: AgingTrajectory) -> Path:
    """Save electrochemical and degradation history without pickled objects."""

    if not trajectory.states:
        raise ValueError("cannot serialize an empty aging trajectory")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "time_s": trajectory.time_s,
        "voltage_V": trajectory.voltage_V,
        "current_A": trajectory.current_A,
        "soc": trajectory.soc,
        "cycle_index": trajectory.cycle_index,
        "capacity_retention": trajectory.capacity_retention,
        "segment_labels": np.asarray(trajectory.segment_labels, dtype=str),
        "termination_reason": np.asarray(trajectory.termination_reason),
        "available_capacity_Ah": np.asarray(
            [state.available_capacity_Ah for state in trajectory.states]
        ),
        "equivalent_full_cycles": np.asarray(
            [state.equivalent_full_cycles for state in trajectory.states]
        ),
        "sei_thickness_m": np.stack(
            [state.degradation.sei_thickness_m for state in trajectory.states]
        ),
        "plated_lithium_mol_m2": np.stack(
            [
                state.degradation.plated_lithium_mol_m2
                for state in trajectory.states
            ]
        ),
        "lost_lithium_C_m2": np.stack(
            [state.degradation.lost_lithium_C_m2 for state in trajectory.states]
        ),
    }
    np.savez_compressed(target, **payload)
    return target


def save_eis_result(path: str | Path, result: EISResult) -> Path:
    """Save complex impedance and diagnostic metadata to compressed NPZ."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        frequency_Hz=result.frequency_Hz,
        impedance_real_ohm=result.impedance_ohm.real,
        impedance_imaginary_ohm=result.impedance_ohm.imag,
        magnitude_ohm=result.magnitude_ohm,
        phase_deg=result.phase_deg,
        voltage_amplitude_V=result.voltage_amplitude_V,
        current_amplitude_A=result.current_amplitude_A,
        metadata_json=np.asarray(
            json.dumps(result.metadata, default=_json_default)
        ),
    )
    return target


def save_pack_trajectory(path: str | Path, trajectory: PackTrajectory) -> Path:
    """Save pack voltage plus per-cell SOC, temperature, and current histories."""

    if not trajectory.states:
        raise ValueError("cannot serialize an empty pack trajectory")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        time_s=trajectory.time_s,
        pack_voltage_V=trajectory.voltage_V,
        pack_current_A=trajectory.current_A,
        cell_soc=trajectory.soc,
        cell_temperature_K=trajectory.temperature_K,
        cell_current_A=np.stack(
            [state.cell_current_A for state in trajectory.states]
        ),
        group_voltage_V=np.stack(
            [state.group_voltage_V for state in trajectory.states]
        ),
    )
    return target


def write_vtu(
    path: str | Path,
    mesh: Mesh,
    *,
    point_data: Mapping[str, np.ndarray] | None = None,
    cell_data: Mapping[str, np.ndarray] | None = None,
) -> Path:
    try:
        import meshio
    except ImportError as exc:
        raise ImportError("VTU export requires meshio; install zynnova[zynsim-io]") from exc
    normalized_cell_data = {
        str(name): [np.asarray(values)] for name, values in (cell_data or {}).items()
    }
    normalized_cell_data.setdefault("region", [mesh.cell_regions])
    output = meshio.Mesh(
        points=mesh.nodes,
        cells=[("tetra", mesh.cells)],
        point_data={str(name): np.asarray(values) for name, values in (point_data or {}).items()},
        cell_data=normalized_cell_data,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output.write(target)
    return target


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


__all__ = [
    "load_p2d_state",
    "save_aging_trajectory",
    "save_eis_result",
    "save_p2d_state",
    "save_p2d_trajectory",
    "save_pack_trajectory",
    "write_vtu",
]
