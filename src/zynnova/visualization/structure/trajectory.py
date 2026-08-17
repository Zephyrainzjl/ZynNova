from __future__ import annotations

from typing import Any

import numpy as np

from zynnova.structure.common.types import StructureData

from .adapters import (
    coerce_trajectory,
    infer_structure_kind,
    is_polymer_record,
    polymer_atom_annotations,
)
from .backend import (
    render_nglview_trajectory,
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
    return {
        value: _UNIT_PALETTE[index % len(_UNIT_PALETTE)]
        for index, value in enumerate(unique)
    }


def _repeat_frames(
    frames: list[StructureData],
    supercell: tuple[int, int, int],
) -> list[StructureData]:
    if supercell == (1, 1, 1):
        return frames
    repeated: list[StructureData] = []
    for frame in frames:
        atoms = frame.to_ase().repeat(supercell)
        repeated.append(StructureData.from_ase(atoms, source=frame.source))
    return repeated


def visualize_trajectory(
    source: Any,
    *,
    kind: str = "auto",
    backend: str = "auto",
    style: str = "ball_and_stick",
    state_id: str | None = None,
    width: int = 900,
    height: int = 560,
    background: str = "white",
    camera: str = "perspective",
    atom_labels: bool = False,
    atom_indices: bool = False,
    show_hydrogens: bool = True,
    show_bonds: bool = True,
    show_unit_cell: bool = True,
    show_axes: bool = False,
    supercell: tuple[int, int, int] = (1, 1, 1),
    spin: bool = False,
    color_by: str = "element",
    interval: int = 100,
    loop: str = "forward",
    reps: int = 0,
    initial_frame: int = 0,
    format: str | None = None,
    **viewer_options: Any,
) -> Any:
    """Display a molecular, polymer, or crystal trajectory in Jupyter.

    ``trajectory`` is intentionally handled before structure-kind dispatch so
    animation-only arguments never leak into :class:`ViewerConfig`.
    """
    if interval <= 0:
        raise ValueError("interval must be positive")
    if reps < 0:
        raise ValueError("reps must be non-negative")
    if loop not in {"forward", "backward", "backAndForth"}:
        raise ValueError("loop must be 'forward', 'backward', or 'backAndForth'")

    frames = coerce_trajectory(source, state_id=state_id, format=format)
    if not frames:
        raise ValueError("trajectory contains no frames")

    atom_count = frames[0].num_atoms
    atomic_numbers = frames[0].atomic_numbers
    for index, frame in enumerate(frames[1:], start=1):
        if frame.num_atoms != atom_count:
            raise ValueError(
                f"trajectory frame {index} has {frame.num_atoms} atoms; "
                f"expected {atom_count}"
            )
        if not np.array_equal(frame.atomic_numbers, atomic_numbers):
            raise ValueError(
                f"trajectory frame {index} has a different atom ordering or composition"
            )

    resolved_kind = (
        infer_structure_kind(frames[0], format=format)
        if kind == "auto"
        else kind
    )
    if resolved_kind not in {"molecule", "polymer", "crystal"}:
        raise ValueError(f"unknown structure kind: {resolved_kind}")
    if resolved_kind == "crystal" and not np.any(frames[0].pbc):
        raise ValueError("crystal trajectory visualization requires periodic frames")

    residue_ids = residue_names = chain_ids = None
    residue_colors = None
    if resolved_kind == "polymer" and is_polymer_record(source):
        residue_ids, residue_names, chain_ids = polymer_atom_annotations(
            source,
            frames[0],
            state_id=state_id,
            frame_index=0,
        )
        if color_by in {"unit", "role", "chain"}:
            residue_colors = _unit_colors(residue_ids)

    repeat_count = int(np.prod(supercell))
    frames = _repeat_frames(frames, supercell)
    if repeat_count > 1 and residue_ids is not None:
        residue_ids = np.tile(residue_ids, repeat_count)
        residue_names = None if residue_names is None else residue_names * repeat_count
        chain_ids = None if chain_ids is None else chain_ids * repeat_count
    initial_frame = int(np.clip(initial_frame, 0, len(frames) - 1))

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
        show_axes=show_axes,
        # Frames have already been repeated above, so the renderer must not
        # replicate them a second time.
        supercell=(1, 1, 1),
        spin=spin,
        color_by=color_by,  # type: ignore[arg-type]
        **viewer_options,
    )
    config.validate()

    selected = resolve_backend(config.backend, trajectory=True)
    if selected == "nglview":
        return render_nglview_trajectory(
            frames,
            config,
            initial_frame=initial_frame,
        )
    return render_py3dmol_trajectory(
        frames,
        config,
        title=f"ZynNova {resolved_kind} trajectory",
        residue_ids=residue_ids,
        residue_names=residue_names,
        chain_ids=chain_ids,
        residue_colors=residue_colors,
        interval=interval,
        loop=loop,
        reps=reps,
        initial_frame=initial_frame,
    )
