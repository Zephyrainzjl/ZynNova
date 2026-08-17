"""Clone an audited ZynNova backend and record the immutable commit.

This helper intentionally does not install dependencies or download weights. Those
operations must follow the selected upstream repository's official instructions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _run(argv: Sequence[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(argv),
        cwd=None if cwd is None else str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sources() -> dict[str, dict[str, object]]:
    path = _root() / "src/zynnova/SOURCE_LOCK.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["id"]): item for item in payload["sources"]}


def build_parser() -> argparse.ArgumentParser:
    sources = _sources()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=sorted(sources))
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--revision", help="tag, branch, or commit; default is upstream default branch")
    parser.add_argument(
        "--accept-upstream-terms",
        action="store_true",
        help="required acknowledgement; this is not a legal determination",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.accept_upstream_terms:
        raise SystemExit("refusing to clone without --accept-upstream-terms")
    source = _sources()[args.source]
    destination = args.destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"destination is not empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    clone = ["git", "clone", "--recursive", str(source["repository"]), str(destination)]
    _run(clone)
    revision = args.revision or str(source.get("default_branch") or "main")
    _run(["git", "checkout", revision], cwd=destination)
    _run(["git", "submodule", "update", "--init", "--recursive"], cwd=destination)
    commit = _run(["git", "rev-parse", "HEAD"], cwd=destination)
    record = {
        "schema": "zynnova.checked-out-source/1.0",
        "id": args.source,
        "repository": source["repository"],
        "requested_revision": revision,
        "commit": commit,
        "license_summary": source["license"],
        "path": str(destination),
    }
    record_path = destination / ".zynnova-source.json"
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(record_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
