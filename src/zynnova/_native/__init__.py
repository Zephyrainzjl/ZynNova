"""Native-extension namespace for ZynNova.

Editable scikit-build-core installs can split Python sources and CMake-installed
extension modules across different ``zynnova`` directories.  This module joins
all ``_native`` directories found under ``zynnova.__path__`` and, on Windows,
registers only trusted package/Conda/compiler runtime directories for dependent
DLL resolution.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_parent = sys.modules.get("zynnova")
_paths: list[str] = []
for _root in getattr(_parent, "__path__", ()):  # root path is extended in zynnova.__init__
    _candidate = Path(_root) / "_native"
    if _candidate.is_dir():
        _value = str(_candidate.resolve())
        if _value not in _paths:
            _paths.append(_value)
__path__ = _paths

_DLL_DIRECTORY_HANDLES: list[object] = []


def _register_windows_dll_directory(path: str | os.PathLike[str]) -> bool:
    """Register *path* for extension-module dependencies on Python >= 3.8."""

    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return False
    candidate = Path(path)
    if not candidate.is_dir():
        return False
    resolved = str(candidate.resolve())
    if any(getattr(handle, "path", None) == resolved for handle in _DLL_DIRECTORY_HANDLES):
        return True
    try:
        handle = os.add_dll_directory(resolved)
    except OSError:
        return False
    # The handle must stay alive for the directory to remain registered.
    try:
        setattr(handle, "path", resolved)
    except Exception:
        pass
    _DLL_DIRECTORY_HANDLES.append(handle)
    return True


def _bootstrap_windows_runtime_dirs() -> tuple[str, ...]:
    """Add high-confidence DLL locations used by Conda/MinGW development installs."""

    if os.name != "nt":
        return ()
    candidates: list[Path] = [Path(value) for value in __path__]
    prefix = Path(sys.prefix)
    candidates.extend((prefix / "DLLs", prefix / "Library" / "bin", prefix / "bin"))
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        cp = Path(conda_prefix)
        candidates.extend((cp / "DLLs", cp / "Library" / "bin", cp / "bin"))
    for exe in ("c++.exe", "g++.exe", "gcc.exe"):
        found = shutil.which(exe)
        if found:
            candidates.append(Path(found).resolve().parent)
    registered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            value = str(candidate.resolve())
        except OSError:
            continue
        if value in seen:
            continue
        seen.add(value)
        if _register_windows_dll_directory(value):
            registered.append(value)
    return tuple(registered)


WINDOWS_RUNTIME_DIRECTORIES = _bootstrap_windows_runtime_dirs()

__all__ = ["WINDOWS_RUNTIME_DIRECTORIES"]
