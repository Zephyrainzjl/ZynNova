"""meshio exchange for ZynSim mixed first-order finite-element meshes."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from ..core.general_mesh import ElementBlock, GeneralMesh


_MESHIO_TO_ZYNSIM = {
    "vertex": "point1",
    "line": "line2",
    "triangle": "tri3",
    "quad": "quad4",
    "tetra": "tet4",
    "pyramid": "pyramid5",
    "wedge": "wedge6",
    "hexahedron": "hex8",
}
_ZYNSIM_TO_MESHIO = {value: key for key, value in _MESHIO_TO_ZYNSIM.items()}


def read_general_mesh(
    path: str | Path,
    *,
    entity_data_name: str | None = None,
) -> GeneralMesh:
    """Read any meshio-supported first-order volume/boundary mesh."""

    try:
        import meshio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("mesh exchange requires meshio; install zynnova[zynsim-io]") from exc
    source = meshio.read(Path(path))
    blocks: list[ElementBlock] = []
    cell_data = source.cell_data.get(entity_data_name, []) if entity_data_name else []
    for index, cell_block in enumerate(source.cells):
        element_type = _MESHIO_TO_ZYNSIM.get(cell_block.type)
        if element_type is None:
            # High-order cells are not silently linearized because that can
            # invalidate geometry and material selections.
            continue
        entities = None
        if index < len(cell_data):
            values = np.asarray(cell_data[index]).reshape(-1)
            if len(values) == len(cell_block.data):
                entities = values.astype(np.int32, copy=False)
        blocks.append(
            ElementBlock(
                element_type,
                np.asarray(cell_block.data, dtype=np.int64),
                entities,
                name=f"{cell_block.type}_{index}",
            )
        )
    if not blocks:
        raise ValueError("input contains no supported first-order cells")
    return GeneralMesh(
        nodes=np.asarray(source.points[:, :3], dtype=float),
        blocks=blocks,
        metadata={
            "source_path": str(Path(path)),
            "meshio_field_data": {key: np.asarray(value).tolist() for key, value in source.field_data.items()},
        },
    )


def write_general_mesh(
    path: str | Path,
    mesh: GeneralMesh,
    *,
    entity_data_name: str = "geometric_entity",
    binary: bool | None = None,
) -> Path:
    """Write a mixed mesh to VTU, XDMF, Exodus, Gmsh, Abaqus, and more."""

    try:
        import meshio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("mesh exchange requires meshio; install zynnova[zynsim-io]") from exc
    cells = []
    entities = []
    for block in mesh.blocks:
        cells.append((_ZYNSIM_TO_MESHIO[block.element_type], block.connectivity))
        entities.append(np.asarray(block.entity_ids, dtype=np.int32))
    kwargs = {} if binary is None else {"binary": bool(binary)}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(
        target,
        meshio.Mesh(
            points=mesh.nodes,
            cells=cells,
            cell_data={entity_data_name: entities},
        ),
        **kwargs,
    )
    return target


__all__ = ["read_general_mesh", "write_general_mesh"]
