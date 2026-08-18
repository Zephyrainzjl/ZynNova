"""Remove stale pip rollback directories left by a locked ZynNova native extension.

Windows does not allow an in-use ``.pyd``/DLL to be deleted. During an
upgrade, pip first renames an installed directory to a rollback name such as
``~.ative``. If a Jupyter kernel, VS Code Python process, or another interpreter
still has a ZynNova native module loaded, cleanup can fail and leave that
rollback directory behind.

Run this script from a *fresh* terminal after closing every Python/Jupyter
process that imported ``zynnova._native``.
"""

from __future__ import annotations

import argparse
import shutil
import site
import sys
from pathlib import Path


def _site_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        roots.extend(Path(value) for value in site.getsitepackages())
    except AttributeError:
        pass
    user = site.getusersitepackages()
    if user:
        roots.append(Path(user))
    # Keep deterministic ordering and remove duplicates.
    return list(dict.fromkeys(path.resolve() for path in roots if path))


def find_stale_native_directories(roots: list[Path] | None = None) -> list[Path]:
    """Return pip rollback directories specifically associated with ZynNova."""

    found: list[Path] = []
    for root in roots or _site_roots():
        package = root / "zynnova"
        if not package.is_dir():
            continue
        for name in ("~.ative", "~native", "~_native"):
            candidate = package / name
            if candidate.exists():
                found.append(candidate)
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remove",
        action="store_true",
        help="actually remove stale rollback directories; default is inspection only",
    )
    parser.add_argument(
        "--site-packages",
        action="append",
        default=[],
        help="override site-packages root (mainly useful for tests)",
    )
    args = parser.parse_args()

    loaded = [
        name
        for name in sys.modules
        if name == "zynnova._native" or name.startswith("zynnova._native.")
    ]
    if loaded:
        print("Refusing cleanup: this Python process has native ZynNova modules loaded:")
        for name in loaded:
            print(f"  {name}")
        print("Start a fresh terminal/interpreter before cleanup.")
        return 3

    roots = [Path(value) for value in args.site_packages] if args.site_packages else None
    candidates = find_stale_native_directories(roots)
    if not candidates:
        print("No stale ZynNova native rollback directories found.")
        return 0

    print("Stale ZynNova native rollback directories:")
    for path in candidates:
        print(f"  {path}")

    if not args.remove:
        print("Inspection only. Re-run with --remove after closing Jupyter/VS Code Python kernels.")
        return 0

    failed: list[tuple[Path, BaseException]] = []
    for path in candidates:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"Removed: {path}")
        except (PermissionError, OSError) as exc:
            failed.append((path, exc))

    if failed:
        print("\nCould not remove one or more directories. A Windows process is likely still")
        print("holding a .pyd/DLL open. Close Jupyter kernels, VS Code Python terminals,")
        print("and other python.exe/pythonw.exe processes that imported ZynNova, then retry.")
        for path, exc in failed:
            print(f"  {path}: {type(exc).__name__}: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
