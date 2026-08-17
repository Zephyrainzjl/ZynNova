from __future__ import annotations

from typing import Any

from .adapters import coerce_structure
from .backend import render_nglview, render_py3dmol, resolve_backend
from .types import ViewerConfig


def visualize_molecule(
    source: Any,
    *,
    backend: str = "auto",
    style: str = "ball_and_stick",
    width: int = 800,
    height: int = 500,
    background: str = "white",
    atom_labels: bool = False,
    atom_indices: bool = False,
    show_hydrogens: bool = True,
    show_bonds: bool = True,
    camera: str = "perspective",
    spin: bool = False,
    trajectory: bool = False,
    interval: int = 100,
    loop: str = "forward",
    reps: int = 0,
    initial_frame: int = 0,
    format: str | None = None,
    **kwargs: Any,
) -> Any:
    """Interactively display a small molecule or molecular trajectory."""
    if trajectory:
        from .trajectory import visualize_trajectory

        return visualize_trajectory(
            source,
            kind="molecule",
            backend=backend,
            style=style,
            width=width,
            height=height,
            background=background,
            atom_labels=atom_labels,
            atom_indices=atom_indices,
            show_hydrogens=show_hydrogens,
            show_bonds=show_bonds,
            camera=camera,
            spin=spin,
            show_unit_cell=False,
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
        atom_labels=atom_labels,
        atom_indices=atom_indices,
        show_hydrogens=show_hydrogens,
        show_bonds=show_bonds,
        camera=camera,  # type: ignore[arg-type]
        spin=spin,
        show_unit_cell=False,
        **kwargs,
    )
    config.validate()
    structure = coerce_structure(source, kind="molecule", format=format)
    selected = resolve_backend(config.backend)
    if selected == "nglview":
        return render_nglview(structure, config)
    return render_py3dmol(structure, config, title=str(structure.info.get("name", "Molecule")))
