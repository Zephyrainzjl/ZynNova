#!/usr/bin/env python3
"""Copy the pinned TetGen 1.6.0 C++ source into the ZynNova source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = "https://github.com/pyvista/tetgen.git"
REVISION = "c039698cf4cce5c671b281c003dbc6cd8e58acc3"
REQUIRED = ("tetgen.cxx", "tetgen.h", "predicates.cxx", "tetgen-license")


def _run(arguments: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}\n"
            f"{completed.stdout[-4000:]}"
        )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkout(checkout: Path) -> Path:
    source = checkout / "src"
    missing = [name for name in REQUIRED if not (source / name).is_file()]
    if missing:
        raise RuntimeError(f"TetGen checkout is incomplete; missing: {missing}")
    if (checkout / ".git").exists():
        actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
        if actual != REVISION:
            raise RuntimeError(
                f"TetGen checkout revision mismatch: expected {REVISION}, received {actual}"
            )
    else:
        raise RuntimeError(
            "--source-checkout must be a git checkout so the pinned revision can be verified"
        )
    header = (source / "tetgen.h").read_text(encoding="utf-8", errors="replace")
    if "Version 1.6" not in header:
        raise RuntimeError("tetgen.h does not identify the expected TetGen 1.6 source")
    license_text = (source / "tetgen-license").read_text(
        encoding="utf-8", errors="replace"
    ).lower()
    if "affero" not in license_text or "general public license" not in license_text:
        raise RuntimeError("tetgen-license does not contain the expected AGPL notice")
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-agpl",
        action="store_true",
        help="acknowledge that TetGen and linked/distributed binaries are AGPL-covered",
    )
    parser.add_argument(
        "--source-checkout",
        type=Path,
        help="existing pyvista/tetgen git checkout at the pinned revision",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing vendor tree")
    args = parser.parse_args()
    if not args.accept_agpl:
        parser.error("--accept-agpl is required before copying TetGen source")

    repository_root = Path(__file__).resolve().parents[1]
    destination = repository_root / "cpp" / "third_party" / "tetgen" / "source"
    existing_sources = [destination / name for name in REQUIRED]
    if any(path.exists() for path in existing_sources) and not args.force:
        raise RuntimeError(
            f"TetGen source already exists at {destination}; pass --force to replace it"
        )

    with tempfile.TemporaryDirectory(prefix="zynnova-tetgen-") as temporary_name:
        temporary = Path(temporary_name)
        if args.source_checkout is None:
            checkout = temporary / "checkout"
            _run(["git", "clone", "--no-checkout", REPOSITORY, str(checkout)])
            _run(["git", "checkout", "--detach", REVISION], cwd=checkout)
        else:
            checkout = args.source_checkout.expanduser().resolve()
            if not checkout.is_dir():
                raise RuntimeError(f"source checkout does not exist: {checkout}")
        source = _verify_checkout(checkout)

        staged = temporary / "source"
        staged.mkdir()
        files: dict[str, dict[str, object]] = {}
        for name in REQUIRED:
            target = staged / name
            shutil.copy2(source / name, target)
            files[name] = {"bytes": target.stat().st_size, "sha256": _sha256(target)}
        manifest = {
            "schema": "zynnova.tetgen-vendor.v1",
            "component": "TetGen",
            "algorithm_version": "1.6.0",
            "repository": REPOSITORY,
            "revision": REVISION,
            "license": "AGPL-3.0-or-later",
            "copied_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": files,
        }
        (staged / "VENDOR_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        backup = destination.with_name(destination.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            destination.rename(backup)
        try:
            shutil.copytree(staged, destination)
        except BaseException:
            if destination.exists():
                shutil.rmtree(destination)
            if backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)

    print(f"Vendored TetGen 1.6.0 at {destination}")
    print(f"Pinned revision: {REVISION}")
    print("License: AGPL-3.0-or-later")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
