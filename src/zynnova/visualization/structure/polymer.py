from __future__ import annotations

from typing import Any

import numpy as np

from .adapters import (
    coerce_structure,
    coerce_trajectory,
    is_polymer_record,
    polymer_atom_annotations,
)
from .backend import (
    render_nglview,
    render_nglview_trajectory,
    render_py3dmol,
    render_py3dmol_trajectory,
    resolve_backend,
)
from .types import ViewerConfig

_UNIT_PALETTE = (
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
)


def _unit_colors(residue_ids: np.ndarray) -> dict[int, str]:
    unique = sorted(set(int(value) for value in residue_ids.tolist()))
    return {value: _UNIT_PALETTE[index % len(_UNIT_PALETTE)] for index, value in enumerate(unique)}


def _render_unit_level(record: Any, config: ViewerConfig, *, state_id: str | None) -> Any:
    """Render repeat-unit/coarse-grained nodes with 3Dmol shape primitives."""
    import py3Dmol

    from zynnova.structure.polymer import Resolution

    states = record.spatial_states if state_id is None else [record.get_state(state_id)]
    frame = None
    for state in states:
        for candidate in state.frames:
            if candidate.resolution in {Resolution.REPEAT_UNIT, Resolution.COARSE_GRAINED}:
                frame = candidate
                break
        if frame is not None:
            break
    if frame is None:
        atomistic = coerce_structure(record, state_id=state_id)
        residues, _, _ = polymer_atom_annotations(record, atomistic, state_id=state_id)
        coordinates = []
        node_ids = []
        for residue in sorted(set(residues.tolist())):
            mask = residues == residue
            coordinates.append(atomistic.positions[mask].mean(axis=0))
            node_ids.append(record.architecture.nodes[residue - 1].id)
        coordinates_array = np.asarray(coordinates, dtype=float)
    else:
        coordinates_array = frame.coordinates
        node_ids = frame.node_ids

    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    node_map = {node.id: node for node in record.architecture.nodes}
    view = py3Dmol.view(width=config.width, height=config.height)
    view.setBackgroundColor(config.background)
    for index, (node_id, coordinate) in enumerate(zip(node_ids, coordinates_array, strict=True)):
        node = node_map.get(node_id)
        unit_id = "unit" if node is None else node.unit_id
        color = _UNIT_PALETTE[index % len(_UNIT_PALETTE)]
        view.addSphere(
            {
                "center": {
                    "x": float(coordinate[0]),
                    "y": float(coordinate[1]),
                    "z": float(coordinate[2]),
                },
                "radius": max(config.sphere_scale * 2.4, 0.35),
                "color": color,
            }
        )
        if config.atom_labels or config.atom_indices:
            label = unit_id if config.atom_labels else ""
            if config.atom_indices:
                label = f"{label}:{index}" if label else str(index)
            view.addLabel(
                label,
                {
                    "position": {
                        "x": float(coordinate[0]),
                        "y": float(coordinate[1]),
                        "z": float(coordinate[2]),
                    },
                    "fontSize": 12,
                    "inFront": True,
                    "backgroundOpacity": 0.55,
                },
            )
    for edge in record.architecture.edges:
        if edge.source not in node_index or edge.target not in node_index:
            continue
        source = coordinates_array[node_index[edge.source]]
        target = coordinates_array[node_index[edge.target]]
        view.addCylinder(
            {
                "start": {"x": float(source[0]), "y": float(source[1]), "z": float(source[2])},
                "end": {"x": float(target[0]), "y": float(target[1]), "z": float(target[2])},
                "radius": config.stick_radius,
                "color": "gray",
            }
        )
    view.zoomTo()
    if config.spin:
        view.spin(True)
    view.render()
    return view


def visualize_polymer(
    source: Any,
    *,
    backend: str = "auto",
    style: str = "ball_and_stick",
    resolution: str = "atomistic",
    state_id: str | None = None,
    frame_index: int = 0,
    trajectory: bool = False,
    color_by: str = "unit",
    width: int = 900,
    height: int = 560,
    background: str = "white",
    camera: str = "perspective",
    atom_labels: bool = False,
    atom_indices: bool = False,
    show_hydrogens: bool = True,
    show_bonds: bool = True,
    show_unit_cell: bool = True,
    spin: bool = False,
    interval: int = 100,
    loop: str = "forward",
    reps: int = 0,
    initial_frame: int = 0,
    **kwargs: Any,
) -> Any:
    """Display a polymer record, atomistic chain, coarse-grained chain, or trajectory."""
    config = ViewerConfig(
        width=width,
        height=height,
        background=background,
        style=style,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        camera=camera,  # type: ignore[arg-type]
        atom_labels=atom_labels,
        atom_indices=atom_indices,
        show_hydrogens=show_hydrogens,
        show_bonds=show_bonds,
        show_unit_cell=show_unit_cell,
        spin=spin,
        color_by=color_by,  # type: ignore[arg-type]
        **kwargs,
    )
    config.validate()

    if resolution in {"unit", "repeat_unit", "coarse_grained", "cg"}:
        if not is_polymer_record(source):
            raise TypeError("unit-level visualization requires a PolymerRecord")
        if resolve_backend(config.backend) != "py3dmol":
            raise ValueError("unit-level polymer visualization currently uses the py3dmol backend")
        return _render_unit_level(source, config, state_id=state_id)

    if trajectory:
        frames = coerce_trajectory(source, state_id=state_id)
        selected = resolve_backend(config.backend, trajectory=True)
        residues = names = chains = None
        if is_polymer_record(source):
            residues, names, chains = polymer_atom_annotations(
                source,
                frames[0],
                state_id=state_id,
                frame_index=frame_index,
            )
        if selected == "nglview":
            return render_nglview_trajectory(
                frames,
                config,
                initial_frame=initial_frame,
            )
        colors = None
        if residues is not None and color_by in {"unit", "role", "chain"}:
            colors = _unit_colors(residues)
        return render_py3dmol_trajectory(
            frames,
            config,
            residue_ids=residues,
            residue_names=names,
            chain_ids=chains,
            residue_colors=colors,
            interval=interval,
            loop=loop,
            reps=reps,
            initial_frame=initial_frame,
        )

    structure = coerce_structure(source, state_id=state_id, frame_index=frame_index)
    selected = resolve_backend(config.backend)
    if selected == "nglview":
        return render_nglview(structure, config)

    residues = names = chains = None
    colors = None
    if is_polymer_record(source):
        residues, names, chains = polymer_atom_annotations(
            source,
            structure,
            state_id=state_id,
            frame_index=frame_index,
        )
        if color_by in {"unit", "role", "chain"}:
            colors = _unit_colors(residues)
    return render_py3dmol(
        structure,
        config,
        title="Polymer",
        residue_ids=residues,
        residue_names=names,
        chain_ids=chains,
        residue_colors=colors,
    )
