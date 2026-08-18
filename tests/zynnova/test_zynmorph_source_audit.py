from __future__ import annotations

import json
from pathlib import Path

import zynnova.zynmorph as zm


def test_source_audit_declares_both_audited_frameworks_and_license_boundary():
    path = Path(zm.__file__).parent / "microstructure" / "SOURCE_AUDIT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = {entry["name"]: entry for entry in payload["sources"]}
    assert "MCS-CICE ElectrodeGenerationAlgorithm" in sources
    assert "MCRpy" in sources
    assert "clean-room" in sources["MCS-CICE ElectrodeGenerationAlgorithm"]["integration_mode"]
    assert sources["MCRpy"]["license_status"] == "Apache-2.0"


def test_unified_top_level_api_exports_generation_characterization_and_reconstruction():
    for name in (
        "generate_particle_electrode",
        "characterize",
        "reconstruct",
        "match",
        "merge",
        "merge_directional",
        "interpolate",
        "mesh_complex_regions",
        "load_comsol_tet4_mphtxt",
    ):
        assert hasattr(zm, name), name
