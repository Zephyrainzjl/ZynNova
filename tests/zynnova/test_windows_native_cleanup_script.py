from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    script = Path(__file__).parents[2] / "scripts" / "cleanup_zynnova_native_windows.py"
    spec = importlib.util.spec_from_file_location("cleanup_zynnova_native_windows", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cleanup_finder_only_targets_known_zynnova_native_rollback_names(tmp_path):
    module = _load_script()
    package = tmp_path / "zynnova"
    package.mkdir()
    expected = {package / "~.ative", package / "~native", package / "~_native"}
    for path in expected:
        path.mkdir()
    (package / "~unrelated").mkdir()

    found = set(module.find_stale_native_directories([tmp_path]))
    assert found == expected
