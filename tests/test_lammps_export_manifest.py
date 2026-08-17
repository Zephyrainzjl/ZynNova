from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from zynnova.ml.zivar import lammps as zivar_lammps


def _fake_lammps(path: Path) -> Path:
    path.write_text(
        '''#!/usr/bin/env python3
print("""Large-scale Atomic/Molecular Massively Parallel Simulator - 30 Mar 2026 - Development
Git info (develop / test-g012345)
Compiler: GNU C++ test
MPI v1.0: LAMMPS MPI STUBS
KOKKOS package API: CUDA OpenMP
KOKKOS package precision: double
KOKKOS package view layout: legacy
Kokkos library version: 5.1.99

Installed packages:
KOKKOS ML-IAP PYTHON

List of individual style options included in this LAMMPS executable
* Fix styles
external external/kk
""")
''',
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _model() -> SimpleNamespace:
    return SimpleNamespace(
        atomic_numbers=(3, 8),
        backbone_manifest={"kind": "convolution", "contract": "test"},
        config=SimpleNamespace(spin=SimpleNamespace(mode="none")),
        _checkpoint_capabilities={"production_validated": False},
        _release_validation=None,
        _release_evidence=None,
    )


def test_reference_bundle_binds_and_records_exact_lammps_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = _fake_lammps(tmp_path / "lmp-audited")
    checkpoint = tmp_path / "input.pt"
    checkpoint.write_bytes(b"zivar-test-checkpoint")
    monkeypatch.setattr(zivar_lammps, "load_zivar", lambda *args, **kwargs: _model())

    bundle = zivar_lammps.export_zivar_lammps_bundle(
        checkpoint,
        tmp_path / "bundle",
        (3, 8),
        config=zivar_lammps.LAMMPSConfig(
            lammps_binary=str(binary),
            require_release_evidence=False,
        ),
    )
    manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))

    assert manifest["schema"] == "zivar-lammps-reference-bundle-0.2.0"
    assert manifest["deployment_class"] == "reference_development"
    assert manifest["backend"] == "python_fix_external_reference_development"
    assert manifest["native"] is False
    assert manifest["production_native_ready"] is False
    assert manifest["limitations"]["full_zivar_native_kokkos"] is False

    identity = manifest["lammps"]
    assert identity["binary"] == str(binary.resolve())
    assert identity["binary_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert identity["version"] == "30 Mar 2026 - Development"
    assert identity["packages"] == ["KOKKOS", "ML-IAP", "PYTHON"]
    assert identity["kokkos"] == {
        "api": ["CUDA", "OpenMP"],
        "compiled": True,
        "library_version": "5.1.99",
        "precision": "double",
        "view_layout": "legacy",
    }
    assert identity["fix_external_compiled"] is True

    driver = bundle.driver.read_text(encoding="utf-8")
    assert "from lammps import lammps" not in driver
    assert "install_reference_callback" in driver
    assert "not a LAMMPS launcher" in driver
    assert "native full-ZIVAR Kokkos implementation is not provided" in driver


def test_production_export_rejects_unbound_lammps_binary(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires LAMMPSConfig.lammps_binary"):
        zivar_lammps.export_zivar_lammps_bundle(
            tmp_path / "checkpoint.pt",
            tmp_path / "bundle",
            (3,),
            config=zivar_lammps.LAMMPSConfig(
                lammps_binary=None,
                require_release_evidence=True,
            ),
        )


def test_default_binary_is_the_documented_nompi_gpu_build() -> None:
    assert zivar_lammps.LAMMPSConfig().lammps_binary == str(
        Path("~/software/lammps-mliap-gpu-nompi/bin/lmp").expanduser().resolve()
    )

