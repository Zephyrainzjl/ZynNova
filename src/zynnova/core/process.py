"""Safe process execution for external repositories and isolated environments."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .exceptions import BackendExecutionError


@dataclass(frozen=True, slots=True)
class ProcessResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    cwd: str | None


def run_process(
    argv: Sequence[str | os.PathLike[str]],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
    check: bool = True,
    stdin: str | None = None,
) -> ProcessResult:
    """Run an argv sequence without a shell, preserving stdout/stderr for audit."""

    command = tuple(str(item) for item in argv)
    if not command:
        raise ValueError("argv cannot be empty")
    resolved_cwd = None if cwd is None else str(Path(cwd).resolve())
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(key): str(value) for key, value in env.items()})
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=resolved_cwd,
            env=merged_env,
            input=stdin,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BackendExecutionError(f"could not execute {format_argv(command)}: {exc}") from exc
    result = ProcessResult(
        argv=command,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        elapsed_s=time.perf_counter() - started,
        cwd=resolved_cwd,
    )
    if check and result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).splitlines()[-20:])
        raise BackendExecutionError(
            f"command failed with exit code {result.returncode}: {format_argv(command)}\n{tail}"
        )
    return result


def format_argv(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in argv)


__all__ = ["ProcessResult", "format_argv", "run_process"]
