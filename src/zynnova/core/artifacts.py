"""Content-addressed run artifacts and reproducibility manifests."""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .serialization import dump_json


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 without loading a potentially large asset into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """One materialized file together with its identity and semantic role."""

    path: str
    role: str
    media_type: str = "application/octet-stream"
    sha256: str | None = None
    bytes: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        role: str,
        media_type: str = "application/octet-stream",
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        return cls(
            path=str(source),
            role=str(role),
            media_type=str(media_type),
            sha256=sha256_file(source),
            bytes=source.stat().st_size,
            metadata=dict(metadata or {}),
        )


@dataclass(slots=True)
class RunManifest:
    """Audit record shared by all four ZynNova subframeworks."""

    workflow: str
    backend: str
    configuration: Mapping[str, Any]
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_unix_s: float = field(default_factory=time.time)
    status: str = "running"
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    events: list[Mapping[str, Any]] = field(default_factory=list)
    software: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.software:
            self.software = runtime_fingerprint()
        self.configuration = dict(self.configuration)
        self.software = dict(self.software)
        self.provenance = dict(self.provenance)

    def add_artifact(
        self,
        path: str | Path,
        *,
        role: str,
        media_type: str = "application/octet-stream",
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        record = ArtifactRecord.from_path(
            path,
            role=role,
            media_type=media_type,
            metadata=metadata,
        )
        self.artifacts.append(record)
        return record

    def event(self, name: str, **payload: Any) -> None:
        self.events.append({"name": str(name), "unix_s": time.time(), **payload})

    def finish(self, *, status: str = "completed") -> None:
        self.status = str(status)
        self.event("finish", status=self.status)

    def save(self, path: str | Path) -> Path:
        return dump_json(path, self)


def runtime_fingerprint() -> Mapping[str, Any]:
    """Collect stable runtime information without importing optional libraries."""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "git_commit": _git_commit(),
    }


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def collect_artifacts(
    paths: Sequence[str | Path],
    *,
    role: str,
    media_type: str = "application/octet-stream",
) -> list[ArtifactRecord]:
    return [
        ArtifactRecord.from_path(path, role=role, media_type=media_type)
        for path in paths
    ]


__all__ = [
    "ArtifactRecord",
    "RunManifest",
    "collect_artifacts",
    "runtime_fingerprint",
    "sha256_file",
]
