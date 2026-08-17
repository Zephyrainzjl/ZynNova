"""Run deterministic ZynNova validation and emit a machine-readable report."""

from __future__ import annotations

import argparse
import compileall
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(argv: Sequence[str], *, root: Path) -> dict[str, object]:
    started = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        list(argv),
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "argv": list(argv),
        "returncode": result.returncode,
        "elapsed_s": time.perf_counter() - started,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/zynnova_validation.json"),
    )
    args = parser.parse_args(argv)
    root = _root()
    started = time.perf_counter()
    compile_ok = compileall.compile_dir(root / "src/zynnova", quiet=1, force=True)
    tests = _run([sys.executable, "-m", "pytest", "-q", "tests/zynnova"], root=root)
    status = _run([sys.executable, "-m", "zynnova", "status"], root=root)
    examples = _run(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "examples/zynnova",
            "scripts/zynnova",
        ],
        root=root,
    )
    report = {
        "schema": "zynnova.validation/1.0",
        "python": sys.version,
        "platform": platform.platform(),
        "compile_package": compile_ok,
        "targeted_tests": tests,
        "backend_status_command": status,
        "compile_examples_and_scripts": examples,
        "elapsed_s": time.perf_counter() - started,
        "passed": bool(
            compile_ok
            and tests["returncode"] == 0
            and status["returncode"] == 0
            and examples["returncode"] == 0
        ),
        "scope": (
            "deterministic local pipelines and external-backend contracts; "
            "large model inference is not executed without repositories/weights"
        ),
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
