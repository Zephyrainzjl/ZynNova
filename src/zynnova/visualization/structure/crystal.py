from __future__ import annotations

from typing import Any

import numpy as np

from .adapters import coerce_structure
from .backend import render_nglview, render_py3dmol, resolve_backend
from .types import ViewerConfig


def visualize_crystal(
    source: Any,
    *,
    backend: str = "auto",
    style: str = "ball_and_stick",
    supercell: tuple[int, int, int] = (1, 1, 1),
    show_unit_cell: bool = True,
    show_axes: bool = True,
    width: int = 800,
    height: int = 560,
    background: str = "white",
    camera: str = "orthographic",
    atom_labels: bool = False,
    atom_indices: bool = False,
    spin: bool = False,
    trajectory: bool = False,
    interval: int = 100,
    loop: str = "forward",
    reps: int = 0,
    initial_frame: int = 0,
    format: str | None = None,
    **kwargs: Any,
) -> Any:
    """Interactively display a periodic crystal, supercell, or trajectory."""
    if trajectory:
        from .trajectory import visualize_trajectory

        return visualize_trajectory(
            source,
            kind="crystal",
            backend=backend,
            style=style,
            supercell=supercell,
            show_unit_cell=show_unit_cell,
            show_axes=show_axes,
            width=width,
            height=height,
            background=background,
            camera=camera,
            atom_labels=atom_labels,
            atom_indices=atom_indices,
            spin=spin,
            interval=interval,
            loop=loop,
            reps=reps,
            initial_frame=initial_frame,
            format=format,
            **kwargs,
        )
    config = ViewerConfig(
        width=width,
        height=height,
        background=background,
        style=style,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        camera=camera,  # type: ignore[arg-type]
        atom_labels=atom_labels,
        atom_indices=atom_indices,
        spin=spin,
        show_unit_cell=show_unit_cell,
        show_axes=show_axes,
        supercell=supercell,
        **kwargs,
    )
    config.validate()
    structure = coerce_structure(source, kind="crystal", format=format)
    if not np.any(structure.pbc):
        raise ValueError("crystal visualization requires periodic boundary conditions")
    selected = resolve_backend(config.backend)
    if selected == "nglview":
        # NGLView does not expose the same simple supercell replication API;
        # replicate through ASE before rendering.
        if supercell != (1, 1, 1):
            atoms = structure.to_ase().repeat(supercell)
            structure = type(structure).from_ase(atoms)
        return render_nglview(structure, config)
    return render_py3dmol(structure, config, title=str(structure.info.get("name", "Crystal")))
