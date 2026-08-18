from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PIN = "c039698cf4cce5c671b281c003dbc6cd8e58acc3"


def test_tetgen_source_pin_license_and_build_contract_are_explicit() -> None:
    root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    cpp_cmake = (ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    lock = json.loads(
        (ROOT / "cpp" / "third_party" / "tetgen" / "SOURCE_LOCK.json").read_text(
            encoding="utf-8"
        )
    )
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert "ZYNNOVA_BUILD_TETGEN" in root_cmake
    assert "ZYNNOVA_TETGEN_SOURCE_DIR" in root_cmake
    assert PIN in cpp_cmake
    assert "TETLIBRARY" in cpp_cmake
    assert "tetgen.cxx" in cpp_cmake and "predicates.cxx" in cpp_cmake
    assert lock["source_revision"] == PIN
    assert lock["license"] == "AGPL-3.0-or-later"
    assert "AGPL-3.0-or-later" in notices


def test_pybind_adapter_uses_regions_facets_quality_and_local_size_callback() -> None:
    binding = (ROOT / "cpp" / "bindings" / "zynmorph_tetgen_module.cpp").read_text(
        encoding="utf-8"
    )
    for token in (
        "regionlist",
        "facetconstraintlist",
        "tetunsuitable",
        "radius_edge_ratio",
        "minimum_dihedral_degrees",
        "numberoftetrahedronattributes",
        "trifacemarkerlist",
    ):
        assert token in binding
    assert 'switches << "D"' in binding
    assert 'switches << "C"' in binding
    assert 'switches << "u"' in binding


def test_vendor_helper_is_pinned_atomic_and_does_not_spawn_a_shell() -> None:
    helper = (ROOT / "scripts" / "vendor_tetgen.py").read_text(encoding="utf-8")
    assert PIN in helper
    assert "--accept-agpl" in helper
    assert "VENDOR_MANIFEST.json" in helper
    assert "shell=True" not in helper
    assert "os.system" not in helper
    assert "subprocess.run" in helper


def test_tetgen_install_contract_does_not_require_optional_readmes() -> None:
    cpp_cmake = (ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    integration_readme = (
        ROOT / "cpp" / "third_party" / "tetgen" / "README.md"
    ).read_text(encoding="utf-8")

    # vendor_tetgen.py replaces source/ with the real upstream source tree, so
    # no source/README.md placeholder may be assumed to survive vendoring.
    assert 'third_party/tetgen/source/README.md' not in cpp_cmake
    assert '_zynnova_tetgen_optional_distribution_files' in cpp_cmake
    assert 'if(EXISTS "${_optional_file}" AND NOT IS_DIRECTORY "${_optional_file}")' in cpp_cmake
    assert 'VENDOR_MANIFEST.json' in cpp_cmake
    assert 'vendor_tetgen.py' in integration_readme
