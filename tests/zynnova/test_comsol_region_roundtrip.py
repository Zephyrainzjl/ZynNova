from __future__ import annotations

import numpy as np

from zynnova.geometry import VolumeMesh
from zynnova.zynmorph import export_comsol_mphtxt, load_comsol_tet4_mphtxt


def _five_region_mesh() -> VolumeMesh:
    # Five disconnected positive-volume tetrahedra are sufficient to exercise
    # the region/entity mapping without introducing shared-face topology.
    nodes = []
    tets = []
    for region in range(5):
        x = 2.0 * region
        start = len(nodes)
        nodes.extend(
            [
                (x, 0.0, 0.0),
                (x + 1.0, 0.0, 0.0),
                (x, 1.0, 0.0),
                (x, 0.0, 1.0),
            ]
        )
        tets.append((start, start + 1, start + 2, start + 3))
    return VolumeMesh(
        np.asarray(nodes, dtype=np.float64),
        np.asarray(tets, dtype=np.int64),
        np.arange(5, dtype=np.int32),
        {i: f"phase_{i}" for i in range(5)},
    )


def test_comsol_roundtrip_restores_original_zero_based_material_ids(tmp_path):
    mesh = _five_region_mesh()
    path = tmp_path / "five_regions.mphtxt"
    report = export_comsol_mphtxt(path, mesh, include_boundaries=False)
    assert report.region_entity_map == {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}

    text = path.read_text(encoding="utf-8")
    assert "# ZYNNOVA_REGION_ENTITY_MAP" in text
    assert '"0":1' in text

    loaded = load_comsol_tet4_mphtxt(path)
    assert loaded.n_nodes == mesh.n_nodes
    assert loaded.n_cells == mesh.n_cells
    assert np.array_equal(loaded.cell_regions, mesh.cell_regions)
    assert loaded.region_names == mesh.region_names
    assert loaded.metadata["restored_original_regions"] is True


def test_external_legacy_mphtxt_without_metadata_keeps_entity_ids(tmp_path):
    mesh = _five_region_mesh()
    path = tmp_path / "legacy.mphtxt"
    export_comsol_mphtxt(path, mesh, include_boundaries=False)
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("# ZYNNOVA_REGION_")
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    loaded = load_comsol_tet4_mphtxt(path)
    assert tuple(np.unique(loaded.cell_regions)) == (1, 2, 3, 4, 5)
    assert loaded.metadata["restored_original_regions"] is False
