from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_split_editable_native_namespace_can_find_second_zynnova_tree(tmp_path: Path) -> None:
    root = _project_root()
    fake = tmp_path / "installed"
    native = fake / "zynnova" / "_native"
    native.mkdir(parents=True)
    (native / "_zynmorph_tetgen_native.py").write_text(
        "tetgen_version='TetGen 1.6.0'\n"
        "tetgen_license='AGPL-3.0-or-later'\n"
        "tetgen_binding_abi=2\n"
        "def tetrahedralize(*args, **kwargs): return {}\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(fake)))
    code = (
        "import zynnova; "
        "from zynnova.zynmorph import tetgen_native_status; "
        "s=tetgen_native_status(); "
        "print(list(zynnova.__path__)); print(s); "
        "assert s.available; assert 'installed' in str(s.module_path)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_mingw_tetgen_runtime_packaging_contract() -> None:
    text = (_project_root() / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "-static-libgcc" in text
    assert "-static-libstdc++" in text
    assert "libwinpthread-1.dll" in text
    assert "-print-file-name=${_runtime_name}" in text
    assert "DESTINATION zynnova/_native" in text


def test_native_diagnostic_script_exists() -> None:
    path = _project_root() / "scripts" / "diagnose_tetgen_native.py"
    assert path.is_file()
