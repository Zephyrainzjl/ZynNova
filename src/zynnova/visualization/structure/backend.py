from __future__ import annotations

from collections.abc import Sequence
from importlib.util import find_spec
from typing import Any

import numpy as np

from zynnova.structure.common.types import StructureData

from .formats import structure_to_pdb, structure_to_sdf, structures_to_multimodel_pdb
from .types import ViewerConfig


class VisualizationDependencyError(ImportError):
    """Raised when an optional visualization backend is unavailable."""


def available_backends() -> tuple[str, ...]:
    output: list[str] = []
    if find_spec("py3Dmol") is not None:
        output.append("py3dmol")
    if find_spec("nglview") is not None:
        output.append("nglview")
    return tuple(output)


def resolve_backend(requested: str, *, trajectory: bool = False) -> str:
    if requested != "auto":
        if requested not in {"py3dmol", "nglview"}:
            raise ValueError(f"unknown visualization backend: {requested}")
        if requested not in available_backends():
            raise VisualizationDependencyError(
                f"{requested} is not installed; install zynnova[visualization]"
            )
        return requested
    installed = available_backends()
    if trajectory and "nglview" in installed:
        return "nglview"
    if "py3dmol" in installed:
        return "py3dmol"
    if "nglview" in installed:
        return "nglview"
    raise VisualizationDependencyError(
        "No notebook visualization backend is installed; run "
        "`pip install -e '.[visualization]'`"
    )


def _py3dmol_style(config: ViewerConfig, *, color: str | None = None) -> dict[str, Any]:
    common = {} if color is None else {"color": color}
    if color is None:
        common["colorscheme"] = "Jmol"
    if config.style == "sphere":
        return {"sphere": {**common, "scale": config.sphere_scale}}
    if config.style == "stick":
        return {"stick": {**common, "radius": config.stick_radius}}
    if config.style == "line":
        return {"line": {**common, "linewidth": config.line_width}}
    return {
        "stick": {**common, "radius": config.stick_radius},
        "sphere": {**common, "scale": config.sphere_scale},
    }


def _apply_py3dmol_labels(view: Any, structure: StructureData, config: ViewerConfig) -> None:
    if not (config.atom_labels or config.atom_indices):
        return
    from zynnova.structure.common.elements import symbols_from_numbers

    symbols = symbols_from_numbers(structure.atomic_numbers)
    for index, (symbol, position) in enumerate(zip(symbols, structure.positions, strict=True)):
        parts: list[str] = []
        if config.atom_labels:
            parts.append(symbol)
        if config.atom_indices:
            parts.append(str(index))
        view.addLabel(
            ":".join(parts),
            {
                "position": {
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                },
                "fontSize": 11,
                "inFront": True,
                "backgroundOpacity": 0.45,
            },
        )


def _add_axes(view: Any, structure: StructureData) -> None:
    if np.any(structure.pbc) and abs(float(np.linalg.det(structure.cell))) > 1e-14:
        vectors = structure.cell
    else:
        span = np.ptp(structure.positions, axis=0)
        length = max(float(np.max(span)), 1.0) * 0.35
        vectors = np.eye(3) * length
    colors = ("red", "green", "blue")
    for vector, color in zip(vectors, colors, strict=True):
        view.addArrow(
            {
                "start": {"x": 0.0, "y": 0.0, "z": 0.0},
                "end": {
                    "x": float(vector[0]),
                    "y": float(vector[1]),
                    "z": float(vector[2]),
                },
                "radius": 0.08,
                "color": color,
            }
        )


def render_py3dmol(
    structure: StructureData,
    config: ViewerConfig,
    *,
    title: str = "ZynNova",
    residue_ids: np.ndarray | None = None,
    residue_names: Sequence[str] | None = None,
    chain_ids: Sequence[str] | None = None,
    residue_colors: dict[int, str] | None = None,
) -> Any:
    import py3Dmol

    view = py3Dmol.view(width=config.width, height=config.height)
    use_sdf = (
        not np.any(structure.pbc)
        and structure.num_atoms <= 999
        and (structure.bonds is None or len(structure.bonds) <= 999)
        and residue_ids is None
    )
    if use_sdf:
        view.addModel(structure_to_sdf(structure, title=title), "sdf")
        model_index = 0
    else:
        view.addModel(
            structure_to_pdb(
                structure,
                title=title,
                residue_ids=residue_ids,
                residue_names=residue_names,
                chain_ids=chain_ids,
            ),
            "pdb",
        )
        model_index = 0

    view.setBackgroundColor(config.background)
    if residue_colors and residue_ids is not None:
        view.setStyle({}, {})
        for residue, color in residue_colors.items():
            view.setStyle({"resi": int(residue)}, _py3dmol_style(config, color=color))
    else:
        view.setStyle({}, _py3dmol_style(config))
    if not config.show_hydrogens:
        view.setStyle({"elem": "H"}, {})
    if not config.show_bonds:
        sphere = {"scale": config.sphere_scale, "colorscheme": "Jmol"}
        view.setStyle({}, {"sphere": sphere})
    if config.style == "surface":
        surface_type = getattr(py3Dmol, "VDW", 1)
        view.addSurface(surface_type, {"opacity": config.surface_opacity, "color": "white"})
    if config.show_unit_cell and np.any(structure.pbc):
        view.addUnitCell(model_index, {"box": {"color": "gray", "radius": 0.05}})
    if np.any(np.asarray(config.supercell) > 1):
        view.replicateUnitCell(*config.supercell, model_index, True)
    if config.show_axes:
        _add_axes(view, structure)
    _apply_py3dmol_labels(view, structure, config)
    view.zoomTo()
    if config.zoom != 1.0:
        view.zoom(config.zoom)
    if config.camera == "orthographic":
        view.setProjection("orthographic")
    if config.spin:
        view.spin(True)
    view.render()
    return view


def render_py3dmol_trajectory(
    frames: Sequence[StructureData],
    config: ViewerConfig,
    *,
    title: str = "ZynNova trajectory",
    residue_ids: np.ndarray | None = None,
    residue_names: Sequence[str] | None = None,
    chain_ids: Sequence[str] | None = None,
    residue_colors: dict[int, str] | None = None,
    interval: int = 100,
    loop: str = "forward",
    reps: int = 0,
    initial_frame: int = 0,
) -> Any:
    import py3Dmol

    if not frames:
        raise ValueError("trajectory contains no frames")

    view = py3Dmol.view(width=config.width, height=config.height)
    view.addModelsAsFrames(
        structures_to_multimodel_pdb(
            frames,
            title=title,
            residue_ids=residue_ids,
            residue_names=residue_names,
            chain_ids=chain_ids,
        ),
        "pdb",
    )
    model_index = 0
    view.setBackgroundColor(config.background)

    if residue_colors and residue_ids is not None:
        view.setStyle({}, {})
        for residue, color in residue_colors.items():
            view.setStyle(
                {"resi": int(residue)},
                _py3dmol_style(config, color=color),
            )
    else:
        view.setStyle({}, _py3dmol_style(config))

    if not config.show_hydrogens:
        view.setStyle({"elem": "H"}, {})
    if not config.show_bonds:
        view.setStyle(
            {},
            {"sphere": {"scale": config.sphere_scale, "colorscheme": "Jmol"}},
        )
    if config.show_unit_cell and np.any(frames[0].pbc):
        view.addUnitCell(model_index, {"box": {"color": "gray", "radius": 0.05}})
    if np.any(np.asarray(config.supercell) > 1):
        view.replicateUnitCell(*config.supercell, model_index, True)
    if config.show_axes:
        _add_axes(view, frames[0])

    # Labels are intentionally omitted for animations.  py3Dmol labels are
    # static scene objects and would otherwise stay at frame-zero coordinates.
    view.animate({"loop": loop, "reps": reps, "interval": interval})
    if initial_frame:
        view.setFrame(initial_frame)
    view.zoomTo()
    if config.zoom != 1.0:
        view.zoom(config.zoom)
    if config.camera == "orthographic":
        view.setProjection("orthographic")
    if config.spin:
        view.spin(True)
    view.render()
    return view

def _configure_nglview(view: Any, structure: StructureData, config: ViewerConfig) -> Any:
    view.clear_representations()
    style = config.style
    if style == "sphere":
        view.add_spacefill(radius_scale=config.sphere_scale)
    elif style == "stick":
        view.add_licorice(radius=config.stick_radius)
    elif style == "line":
        view.add_line()
    elif style == "surface":
        view.add_ball_and_stick()
        view.add_surface(opacity=config.surface_opacity)
    else:
        view.add_ball_and_stick(aspect_ratio=max(config.stick_radius * 10, 1.0))
    if not config.show_hydrogens:
        # NGL selection uses boolean expressions.
        view.representations = [
            {**item, "params": {**item.get("params", {}), "sele": "not hydrogen"}}
            for item in view.representations
        ]
    if config.show_unit_cell and np.any(structure.pbc):
        view.add_unitcell()
    view.background = config.background
    view.camera = config.camera
    view.center()
    return view


def render_nglview(structure: StructureData, config: ViewerConfig) -> Any:
    try:
        import nglview as nv
    except ImportError as exc:
        raise VisualizationDependencyError(
            "nglview is not installed; install zynnova[visualization]"
        ) from exc
    view = nv.show_ase(structure.to_ase(), default=False)
    return _configure_nglview(view, structure, config)


def render_nglview_trajectory(
    frames: Sequence[StructureData],
    config: ViewerConfig,
    *,
    initial_frame: int = 0,
) -> Any:
    try:
        import nglview as nv
    except ImportError as exc:
        raise VisualizationDependencyError(
            "nglview is not installed; install zynnova[trajectory-visualization]"
        ) from exc
    if not frames:
        raise ValueError("trajectory contains no frames")
    ase_frames = [frame.to_ase() for frame in frames]
    view = nv.show_asetraj(ase_frames, default=False)
    view = _configure_nglview(view, frames[0], config)
    view.frame = int(np.clip(initial_frame, 0, len(frames) - 1))
    return view

